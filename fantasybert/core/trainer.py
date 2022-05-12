import os
from torch.utils.data import DataLoader
import warnings
from pkg_resources import parse_version
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import time, datetime
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.cuda.amp import autocast as autocast, GradScaler
from fantasybert.models.modeling_utils import build_bert_optim_parameters
from fantasybert.core.tester import Tester
from fantasybert.core.trainer_utils import _build_args, _get_func_signature, _model_contains_inner_module, _save_model
from fantasybert.core.utils import _move_dict_value_to_device
from fantasybert.core.data_utils import BatchIter, TrainerState
from fantasybert.core.callback import CallbackManager, Callback, CallbackException
from fantasybert.core.logger_utils import logger
from fantasybert.core.loss_main import prepare_losser, LossBase
from fantasybert.core.metrics import _prepare_metrics, MetricBase
from fantasybert.core.optimization import _prepare_optimizers, get_scheduler, Lookahead
warnings.filterwarnings("ignore")
from typing import Union, Optional, Any, List


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 args: Union[TrainerState, Any],
                 train_data: Union[Dataset, BatchIter, DataLoader],
                 tester: Optional[Tester] = None,
                 loss: Union[LossBase, str] = 'cross_entropy',
                 eval_loss: Union[LossBase, str] = 'as_train',
                 optimizer: Union[Optimizer, str] = 'AdamW',
                 scheduler: Union[LambdaLR, str] = None,
                 dev_data: Union[Dataset, BatchIter, DataLoader] = None,
                 test_data: Union[Dataset, BatchIter, DataLoader] = None,
                 callbacks: Union[List[Callback], List[str], Callback, str] = 'default',
                 stay_default_callback: bool = False, # set stay_default_callback = True only when we pass new callbacks, therefore callbacks!='default'
                 metrics: Union[List[MetricBase], List[str], MetricBase, str] = None,
                 collate_fn=None,
                 train_collate_fn=None,
                 eval_collate_fn=None,
                 train_shuffle: bool = True,
                 eval_shuffle: bool = False,
                 use_tqdm: bool = True,
                 validate_every: int = -1,
                 print_every: int = -1,
                 show_train_result=True,
                 **kwargs,
                 ):
        super(Trainer, self).__init__()

        self.args = args
        if collate_fn is not None and train_collate_fn is None and eval_collate_fn is None:
            self.train_collate_fn = collate_fn
            self.eval_collate_fn = collate_fn
        else:
            self.train_collate_fn = train_collate_fn
            self.eval_collate_fn = eval_collate_fn
        self.train_shuffle = train_shuffle
        self.eval_shuffle = eval_shuffle

        self.batch_size = self.args.train_batch_size
        self.fp16 = self.args.fp16
        self.n_epochs = self.args.n_epochs
        self.use_block_shuffle = self.args.use_block_shuffle

        self.validate_every = validate_every
        self.show_train_result = show_train_result

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.use_tqdm = use_tqdm
        self.stay_default_callback = stay_default_callback

        self.best_metric_indicator = None
        self.best_dev_epoch = None
        self.best_dev_step = None
        self.best_dev_perf = None

        self.device = self.args.device
        self.model = model
        self.model.to(self.device)

        self.pin_memory = kwargs.get('pin_memory', False if parse_version(torch.__version__)==parse_version('1.9') else True)
        self.print_every = print_every if print_every != -1 else self.validate_every
        self.set_grad_to_none = kwargs.get('set_grad_to_none', True)

        self.scaler = GradScaler()

        # model
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be torch.nn.Module, got {type(model)}.")

        # data
        self.worker_init_fn = kwargs.get('worker_init_fn')
        self.batch_sampler = kwargs.get('batch_sampler')
        if isinstance(train_data, Dataset):
            self.data_iterator = BatchIter(dataset=train_data, batch_size=self.batch_size, sampler=self.args.sampler,
                                             num_workers=self.args.num_workers, drop_last=self.args.drop_last,
                                            collate_fn=self.train_collate_fn, shuffle=self.train_shuffle,
                                                use_block_shuffle=self.use_block_shuffle,**kwargs)
        elif isinstance(train_data, DataLoader):
            self.data_iterator = train_data
        elif isinstance(train_data, BatchIter):
            self.data_iterator = train_data
        else:
            raise TypeError("train_data type {} not support".format(type(train_data)))

        if _model_contains_inner_module(self.model):
            self._forward_func = self.model.module.forward
        else:
            self._forward_func = self.model.forward

        # gradient_accumulation_steps
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        assert self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be no less than 1."

        self.n_steps = (len(self.data_iterator) // self.gradient_accumulation_steps) * self.n_epochs
        # TODO：到底用不用除？

        # optimizer
        self.optimizer_clss = _prepare_optimizers(optimizer)
        optimizer_grouped_parameters = build_bert_optim_parameters(self.model, self.args.weight_decay)
        self.optimizer = self.optimizer_clss(optimizer_grouped_parameters, lr=self.args.learning_rate)
        if self.args.use_lookahead:
            self.optimizer = Lookahead(optimizer=self.optimizer, **self.args.optim_args)

        # scheduler
        self.warmup_steps = int(self.n_steps * self.args.warmup_ratios)
        if isinstance(scheduler, str):
            self.scheduler = get_scheduler(name=scheduler,
                                           optimizer=self.optimizer,
                                           num_warmup_steps=self.warmup_steps,
                                           num_training_steps=self.n_steps)
        elif isinstance(scheduler, LambdaLR):
            self.scheduler = scheduler
        else:
            self.scheduler = None

        # loss
        self.losser = prepare_losser(loss)
        if eval_loss == 'as_train':
            self.eval_loss = loss
        else:
            self.eval_loss = eval_loss

        # metrics
        self.metrics = _prepare_metrics(metrics)

        # metric_key
        self.metric_key = self.args.metric_key
        if self.metric_key is not None:
            self.metric_key = self.metric_key[1:] if self.metric_key[0] == "+" or self.metric_key[0] == "-" else self.metric_key
        else:
            self.metric_key = None


        # tester
        if self.dev_data is not None:
            if tester is None:
                self.tester = Tester(model=self.model,
                                 args=self.args,
                                 loss=self.eval_loss,
                                 data=self.dev_data,
                                 metrics=self.metrics,
                                 collate_fn=self.eval_collate_fn,
                                 shuffle=self.eval_shuffle,
                                 use_tqdm=self.args.test_use_tqdm)
            else:
                self.tester = tester

        # save_path
        self.save_path = self.args.save_path
        if not (self.args.save_path is None or isinstance(self.args.save_path, str)):
            raise ValueError("save_path can only be None or `str`.")

        # callback
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callback_manager = CallbackManager(env={"trainer": self},
                                                callbacks=callbacks)


    def _compute_loss(self, predict, truth, batch_data):
        return self.losser(predict, truth, batch_data, self.args)


    def train(self, load_best_model=True, on_exception='auto', **kwargs):
        results = {}

        if self.n_epochs <= 0:
            logger.info(f"training epoch is {self.n_epochs}, nothing was done.")
            results['seconds'] = 0.
            return results
        try:
            self.model.train()
            self._load_best_model = load_best_model
            self.start_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            start_time = time.time()
            logger.info("training epochs started " + self.start_time)
            self.step = 0
            self.epoch = 1

            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()

            except BaseException as e:
                self.callback_manager.on_exception(e)
                if on_exception == 'auto':
                    if not isinstance(e, (CallbackException, KeyboardInterrupt)):
                        raise e
                elif on_exception == 'raise':
                    raise e


            if self.dev_data is not None and self.best_dev_perf is not None and load_best_model:
                model_name = "best_" + "_".join([self.model.__class__.__name__, self.metric_key])
                load_succeed = self._load_model(self.model, model_name)
                if load_succeed:
                    logger.info("Reloaded the best model.")
                else:
                    logger.info("Fail to reload best model.")

            if self.dev_data is None and self.save_path is not None:
                model_name = "_".join([self.model.__class__.__name__])
                _save_model(self.model, model_name, self.save_path, self.device, self.args.save_only_param)

        finally:
            if self.dev_data is not None and self.best_dev_perf is not None:
                print(self.best_dev_perf)
                logger.info(
                    "\nIn Epoch:{}/Step:{}, got best dev performance:".format(self.best_dev_epoch, self.best_dev_step))
                logger.info(self.tester._format_eval_results(self.best_dev_perf))
                results['best_eval'] = self.best_dev_perf
                results['best_epoch'] = self.best_dev_epoch
                results['best_step'] = self.best_dev_step

        results['seconds'] = round(time.time() - start_time, 2)

        return results

    def _train(self):
        if not self.use_tqdm:
            from .trainer_utils import _pseudo_tqdm as inner_tqdm
        else:
            inner_tqdm = tqdm
        start = time.time()
        with inner_tqdm(total=self.n_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                        initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            self.batch_per_epoch = len(self.data_iterator)
            for epoch in range(self.epoch, self.n_epochs + 1):
                self.epoch = epoch
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
                # early stopping
                self.callback_manager.on_epoch_begin()
                for batch in self.data_iterator:
                    self.step += 1
                    batch_x, batch_y = self._get_batch_xy(batch)
                    if batch_y is not None:
                        _move_dict_value_to_device(batch_x, batch_y, device=self.device)
                    else:
                        _move_dict_value_to_device(batch_x, device=self.device)
                    _move_dict_value_to_device(batch, device=self.device)

                    # negative sampling; replace unknown; re-weight batch_y
                    self.callback_manager.on_batch_begin(batch_x, batch_y)
                    prediction = self._data_forward(self.model, batch)

                    # edit prediction
                    self.callback_manager.on_loss_begin(batch_y, prediction)
                    loss = self._compute_loss(prediction, batch_y, batch_x).mean()
                    loss = loss / self.gradient_accumulation_steps
                    avg_loss += loss.item()

                    # Is loss NaN or inf? requires_grad = False
                    self.callback_manager.on_backward_begin(loss)
                    self._grad_backward(loss)

                    self.callback_manager.on_backward_end(batch_x, batch_y)

                    self._update()
                    self.callback_manager.on_step_end()
                    self._gradient_zero_grad()

                    if self.step % self.print_every == 0:
                        avg_loss = float(avg_loss) / self.print_every
                        if self.use_tqdm:
                            print_output = "loss:{:<6.5f}".format(avg_loss)
                            pbar.update(self.print_every)
                        else:
                            end = time.time()
                            diff = round(end - start)
                            print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                                epoch, self.step, avg_loss, diff)
                        pbar.set_postfix_str(print_output)
                        avg_loss = 0
                    self.callback_manager.on_batch_end()

                    if (self.validate_every > 0 and self.step % self.validate_every == 0) \
                            and self.dev_data is not None:
                        eval_res = self._do_validation(epoch=epoch, step=self.step)
                        eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(epoch, self.n_epochs, self.step,
                                                                                    self.n_steps)
                        # pbar.write(eval_str + '\n')
                        logger.info(eval_str)
                        logger.info(self.tester._format_eval_results(eval_res)+'\n')
                # ================= mini-batch end ==================== #
                if self.validate_every<0 and self.dev_data is not None:  # 在epoch结束之后的evaluate
                    eval_res = self._do_validation(epoch=epoch, step=self.step)
                    eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(epoch, self.n_epochs, self.step,
                                                                                       self.n_steps)
                    # pbar.write(eval_str + '\n')
                    logger.info(eval_str)
                    logger.info(self.tester._format_eval_results(eval_res) + '\n')
                # lr decay; early stopping
                self.callback_manager.on_epoch_end()
            # =============== epochs end =================== #
            if self.dev_data is not None and (self.validate_every>0 and self.n_steps%self.validate_every!=0):
                eval_res = self._do_validation(epoch=epoch, step=self.step)
                eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(epoch, self.n_epochs, self.step,
                                                                                   self.n_steps)
                # pbar.write(eval_str + '\n')
                logger.info(eval_str)
                logger.info(self.tester._format_eval_results(eval_res) + '\n')
            pbar.close()
            self.pbar = None
        # ============ tqdm end ============== #

    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        if self.fp16:
            # 前向过程(model + loss)开启 autocast
            with autocast():
                outputs = network(**x)
        else:
            outputs = network(**x)
        if not isinstance(outputs, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(outputs)}.")
        return outputs

    def _grad_backward(self, loss):
        if self.fp16:
            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大
            self.scaler.scale(loss).backward()
        else:
            loss.backward()  #  gradient_accumulation_steps？？

    def _gradient_zero_grad(self):
        if self.step % self.gradient_accumulation_steps == 0:
            self._clear_grad(self.optimizer, self.set_grad_to_none)

    def _clear_grad(self, optimizer, set_to_none=True):
        param_groups = optimizer.param_groups
        for group in param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def _update(self):
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
            # scaler.step() unscale之前放大后的梯度，但是scale太多可能出现inf或NaN
            # 故其会判断是否出现了inf/NaN
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 如果检测到出现了inf或者NaN，就跳过这次梯度更新，同时动态调整scaler的大小
            self.scaler.step(self.optimizer)
            # 查看是否要更新scaler
            self.scaler.update()
        else:
            self.optimizer.step()

    def _do_validation(self, epoch, step):
        self.callback_manager.on_valid_begin()
        res = self.tester.test()

        is_better_eval = False
        if self._better_eval_result(res):
            if self.save_path is not None:
                _save_model(self.model, "best_" + "_".join([self.model.__class__.__name__, self.metric_key]),
                            self.save_path, self.device, self.args.save_only_param)
            elif self._load_best_model:
                self._best_model_states = {name: param.cpu().clone() for name, param in self.model.state_dict().items()}
            self.best_dev_perf = res
            self.best_dev_epoch = epoch
            self.best_dev_step = step
            is_better_eval = True
        # get validation results; adjust optimizer
        self.callback_manager.on_valid_end(res, self.metric_key, self.optimizer, is_better_eval)
        return res

    def test(self, model_name):
        load_succeed = self._load_model(self.model, model_name)
        if load_succeed:
            logger.info("Successfully loaded the best model.")
        else:
            logger.info("Failed to load best model.")
        tester = Tester(model=self.model,
               args=self.args,
               data=self.test_data,
               metrics=self.metrics,
               collate_fn=self.eval_collate_fn,
               shuffle=self.eval_shuffle,
               use_tqdm=True)
        res = tester.test()
        for k,v in res.items():
            print(v)


    def _better_eval_result(self, metrics):
        r"""Check if the current epoch yields better validation results.

        :return bool value: True means current results on dev set is the best.
        """
        indicator, indicator_val = _check_eval_results(metrics, self.metric_key, self.metrics)
        if self.metric_key is None:
            self.metric_key = indicator
        is_better = True
        if self.best_metric_indicator is None:
            # first-time validation
            self.best_metric_indicator = indicator_val
        else:
            if indicator != 'loss':
                if indicator_val > self.best_metric_indicator:
                    self.best_metric_indicator = indicator_val
                else:
                    is_better = False
            else:
                if indicator_val < self.best_metric_indicator:
                    self.best_metric_indicator = indicator_val
                else:
                    is_better = False
        return is_better


    def _load_model(self, model, model_name, only_param=False):
        # 返回bool值指示是否成功reload模型
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if only_param:
                states = torch.load(model_path)
            else:
                states = torch.load(model_path)
                #states = torch.load(model_path).state_dict()
            if _model_contains_inner_module(model):
                model.module.load_state_dict(states)
            else:
                model.load_state_dict(states)
        elif hasattr(self, "_best_model_states"):
            model.load_state_dict(self._best_model_states)
        else:
            return False
        return True


    def _get_batch_xy(self, batch):
        batch_x, batch_y = {}, None
        for name, item in batch.items():
            if name == 'label':
                batch_y = {}
                batch_y['target'] = batch['label']
            else:
                batch_x[name] = item
        return batch_x, batch_y


def _check_eval_results(metrics, metric_key, metric_list):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    # metric_list: 多个用来做评价的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    indicator_val, indicator = None, None
    if isinstance(metrics, dict):

        if metric_key is None:
            metric_dict = list(metrics.values())[0]  # 取第一个metric
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            metric_dict = list(metrics.values())
            # [{'accuracy': 0.073444}, {'f1': 0.073448, 'precision': 0.073451, 'recall': 0.073444}, {'loss': 3.644089}]
            if not any(metric_key in _.keys() for _ in metric_dict):
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")

            for item in metric_dict:
                if metric_key in item.keys():
                    indicator_val = item[metric_key]
                    indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))

    return indicator, indicator_val