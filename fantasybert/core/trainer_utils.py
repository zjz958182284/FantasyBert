import os
import torch
import inspect
import torch.nn as nn
from fantasybert.core.logger_utils import logger
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report as sl_classification_report

def compute_cls_metrics(preds, labels):
    assert len(preds) == len(labels)
    results = {}

    classification_report_dict = classification_report(labels, preds, output_dict=True)

    for key0, val0 in classification_report_dict.items():
        if key0 == 'weighted avg':
            if isinstance(val0, dict):
                for key1, val1 in val0.items():
                    if key1 == 'recall' or key1 == 'precision' or key1 == 'f1-score':
                        results[key0 + "__" + key1] = val1
            else:
                results[key0] = val0

    accuracy = accuracy_score(labels, preds)
    results['accuracy'] = accuracy
    return results


def compute_sequence_labeling_metrics(preds, labels):
    assert len(preds) == len(labels)
    results = {}

    classification_report_dict = sl_classification_report(labels, preds, output_dict=True)
    # macro avg, weighted avg
    for key0, val0 in classification_report_dict.items():
        if key0 == 'weighted avg':
            if isinstance(val0, dict):
                for key1, val1 in val0.items():
                    if key1 == 'recall' or key1 == 'precision' or key1 == 'f1-score':
                        results[key0 + "__" + key1] = val1
            else:
                results[key0] = val0

    return results


def _build_args(func, **kwargs):
    r"""
    根据func的初始化参数，从kwargs中选择func需要的参数

    :param func: callable
    :param kwargs: 参数
    :return:dict. func中用到的参数
    """
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output


def _get_func_signature(func):
    r"""

    Given a function or method, return its signature.
    For example:

    1 function::

        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'

    2 method::

        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'

    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = '(self, '
        else:
            _self = '(self'
        signature_str = class_name + '.' + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


def _data_parallel_wrapper(func_name, device_ids, output_device):
    r"""
    这个函数是用于对需要多卡执行的函数的wrapper函数。参考的nn.DataParallel的forward函数

    :param str, func_name: 对network中的这个函数进行多卡运行
    :param device_ids: nn.DataParallel中的device_ids
    :param output_device: nn.DataParallel中的output_device
    :return:
    """

    def wrapper(network, *inputs, **kwargs):
        inputs, kwargs = scatter_kwargs(inputs, kwargs, device_ids, dim=0)
        if len(device_ids) == 1:
            return getattr(network, func_name)(*inputs[0], **kwargs[0])
        replicas = replicate(network, device_ids[:len(inputs)])
        outputs = parallel_apply(replicas, func_name, inputs, kwargs, device_ids[:len(replicas)])
        return gather(outputs, output_device)

    return wrapper


class _pseudo_tqdm:
    r"""
    当无法引入tqdm，或者Trainer中设置use_tqdm为false的时候，用该方法打印数据
    """

    def __init__(self, **kwargs):
        self.logger = logger

    def write(self, info):
        self.logger.info(info)

    def set_postfix_str(self, info):
        self.logger.info(info)

    def __getattr__(self, item):
        def pass_func(*args, **kwargs):
            pass

        return pass_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

def _model_contains_inner_module(model):
    r"""

    :param nn.Module model: 模型文件，判断是否内部包含model.module, 多用于check模型是否是nn.DataParallel,
        nn.parallel.DistributedDataParallel。主要是在做形参匹配的时候需要使用最内部的model的function。
    :return: bool
    """
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return True
    return False


def _get_model_device(model):
    r"""
    传入一个nn.Module的模型，获取它所在的device

    :param model: nn.Module
    :return: torch.device,None 如果返回值为None，说明这个模型没有任何参数。
    """
    # TODO 这个函数存在一定的风险，因为同一个模型可能存在某些parameter不在显卡中，比如BertEmbedding. 或者跨显卡
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device

def _save_model(model, model_name, save_dir, device=None, only_param=False):
    r""" 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param save_dir: 保存的directory
    :param only_param:
    :return:
    """
    if device is None:
        device = _get_model_device(model)
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if _model_contains_inner_module(model):
        model = model.module
    if only_param:
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, model_path)
    else:
        model.cpu()
        torch.save(model, model_path)
        model.to(device)

