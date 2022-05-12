import torch.nn as nn
from functools import partial
import warnings

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
from torch.cuda.amp import autocast as autocast
from fantasybert.core.trainer_utils import _build_args, _get_func_signature
from fantasybert.core.utils import _move_dict_value_to_device
from fantasybert.core.data_utils import BatchIter, TrainerState
from fantasybert.core.trainer_utils import _model_contains_inner_module, _data_parallel_wrapper
from fantasybert.core.logger_utils import logger
from fantasybert.core.loss_main import prepare_losser, LossBase
from fantasybert.core.metrics import _prepare_metrics, MetricBase
warnings.filterwarnings("ignore")
from typing import Union, Any


class Tester(object):
    def __init__(self,
                 model: nn.Module,
                 args: Union[TrainerState, Any],
                 data: Union[Dataset, BatchIter],
                 loss: Union[LossBase, str] = 'cross_entropy',
                 metrics: Union[MetricBase, str] = None,
                 collate_fn = None,
                 shuffle: bool = False,
                 use_tqdm: bool = False,
                 **kwargs):

        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")

        self.args = args
        self.data = data
        self.device = self.args.device
        self._model = model.to(self.device)
        self.batch_size = self.args.eval_batch_size
        self.use_tqdm = use_tqdm
        self.logger = logger
        self.pin_memory = kwargs.get('pin_memory', True)

        self.metrics = _prepare_metrics(metrics)
        self.eval_loss = loss
        if loss != 'none':
            self.losser = prepare_losser(loss)

        # data
        if isinstance(self.data, Dataset):
            self.data_iterator = BatchIter(dataset=self.data, batch_size=self.batch_size, sampler=self.args.sampler,
                                           num_workers=self.args.num_workers, drop_last=self.args.drop_last,
                                           collate_fn=collate_fn, shuffle=shuffle,
                                           use_block_shuffle=self.args.eval_use_block_shuffle, **kwargs)
        elif isinstance(self.data, BatchIter):
            self.data_iterator = self.data
            self.data = self.data.dataset
        else:
            raise TypeError("train_data type {} not support".format(type(self.data)))

        # check predict
        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                            self._model.device_ids,
                                                                            self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict  # 用于匹配参数
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict  # 用于调用
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
        else:
            if _model_contains_inner_module(model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward


    def _compute_loss(self, predict, truth, batch_data):
        return self.losser(predict, truth, batch_data, self.args)


    def test(self):
        r"""开始进行验证，并返回验证结果。

        :return Dict[Dict]: dict的二层嵌套结构，dict的第一层是metric的名称; 第二层是这个metric的指标。一个AccuracyMetric的例子为{'AccuracyMetric': {'acc': 1.0}}。
        """
        # turn on the testing mode; clean up the history
        self._model.eval()
        data_iterator = self.data_iterator
        eval_results = {}
        avg_loss = 0

        with torch.no_grad():
            if not self.use_tqdm:
                from fantasybert.core.trainer_utils import _pseudo_tqdm as inner_tqdm
            else:
                inner_tqdm = tqdm
            with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Test")

                start_time = time.time()
                for batch in self.data_iterator:
                    batch_x, batch_y = self._get_batch_xy(batch)

                    _move_dict_value_to_device(batch_x, batch_y, device=self.device)
                    _move_dict_value_to_device(batch, device=self.device)

                    pred_dict = self._data_forward(self._predict_func, batch)

                    if self.eval_loss != 'none':
                        loss = self._compute_loss(pred_dict, batch_y, batch_x).mean()
                        avg_loss += loss.item()

                    if not isinstance(pred_dict, dict):
                        raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                        f"must be `dict`, got {type(pred_dict)}.")
                    for metric in self.metrics:
                        metric(pred_dict, batch_y)

                    if self.use_tqdm:
                        pbar.update()

                avg_loss = float(avg_loss) / len(self.data_iterator)
                for metric in self.metrics:
                    eval_result = metric.get_metric()
                    if not isinstance(eval_result, dict):
                        raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                        f"`dict`, got {type(eval_result)}")
                    metric_name = metric.get_metric_name()
                    eval_results[metric_name] = eval_result
                if self.eval_loss != 'none':
                    eval_results['loss'] = {'loss': round(avg_loss,7)}
                pbar.close()
                end_time = time.time()
                test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                self.logger.info(test_str)

        return eval_results

    def _get_batch_xy(self, batch):
        batch_x, batch_y = {}, None
        for name, item in batch.items():
            if name == 'label':
                batch_y = {}
                batch_y['target'] = batch['label']
            else:
                batch_x[name] = item
        return batch_x, batch_y

    def _data_forward(self, func, x):
        x = _build_args(func, **x)
        if self.args.fp16:
            with autocast():
                outputs = self._predict_func_wrapper(**x)
        else:
            outputs = self._predict_func_wrapper(**x)
        return outputs

    def _format_eval_results(self, results):
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]
