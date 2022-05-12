import torch.nn as nn
import inspect
from collections import defaultdict

import torch
import torch.nn.functional as F
from fantasybert.core.utils import seq_len_to_mask, _check_function_or_method
from fantasybert.core.losses import FocalLoss, RDropLoss, DiceLoss
from fantasybert.core.utils import Const
from fantasybert.core.trainer_utils import _build_args, _get_func_signature

class LossBase(object):
    def __init__(self):
        self._param_map = {}  # key是fun的参数，value是以该值从传入的dict取出value

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _init_param_map(self, **kwargs):
        r"""perform checking
        """
        value_counter = defaultdict(set)

        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")
        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.get_loss)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.get_loss)}. Please check the "
                    f"initialization parameters, or change its signature.")


    def __call__(self, pred_dict, target_dict, batch_data, args, check=True):
        batch_dict = {}
        batch_dict['batch_dict'] = batch_data
        if check:
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.get_loss)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        mapped_pred_dict = {}
        mapped_target_dict = {}
        mapped_batch_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if target_dict is not None:
                if input_arg in target_dict:
                    mapped_target_dict[mapped_arg] = target_dict[input_arg]
            if input_arg in batch_dict:
                mapped_batch_dict[mapped_arg] = batch_dict[input_arg]

        refined_args = _build_args(self.get_loss, **mapped_pred_dict, **mapped_target_dict, **mapped_batch_dict, **args.to_dict)

        loss = self.get_loss(**refined_args)
        return loss


class LossInForward(LossBase):
    r"""
    从forward()函数返回结果中获取loss
    """

    def __init__(self, loss_key=Const.LOSS):
        r"""

        :param str loss_key: 在forward函数中loss的键名，默认为loss
        """
        super().__init__()
        if not isinstance(loss_key, str):
            raise TypeError(f"Only str allowed for loss_key, got {type(loss_key)}.")
        self.loss_key = loss_key

    def get_loss(self, **kwargs):
        if self.loss_key not in kwargs:
            raise AttributeError(f'loss_key:{self.loss_key} not found in kwargs')
        return kwargs[self.loss_key]

    def __call__(self, pred_dict, target_dict, batch_dict, args, check=False):

        loss = self.get_loss(**pred_dict)

        if not (isinstance(loss, torch.Tensor) and len(loss.size()) == 0):
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss excepted to be a torch.Tensor, got {type(loss)}")
            loss = torch.sum(loss) / (loss.view(-1)).size(0)
            # raise RuntimeError(f"The size of loss excepts to be torch.Size([]), got {loss.size()}")

        return loss


class LossFunc(LossBase):
    r"""
    提供给用户使用自定义损失函数的类

    :param func: 用户自行定义的损失函数，应当为一个函数。
    :param dict key_map: 参数映射表。键为Model/DataSet参数名，值为损失函数参数名。
                         fastNLP的trainer将在训练时从模型返回值或者训练数据DataSet的target=True的field中
                         找到相对应的参数名为value的参数，并传入func中作为参数名为key的参数
    :param kwargs: 除了参数映射表以外可以用key word args的方式设置参数映射关系

    使用方法::

        import torch.nn.functional as F
        loss_func = LossFunc(F.cross_entropy, input="pred", target="label")
        # 这表示构建了一个损失函数类，由func计算损失函数，其中将从模型返回值或者DataSet的target=True的field
        # 当中找到一个参数名为`pred`的参数传入func一个参数名为`input`的参数；找到一个参数名为`label`的参数
        # 传入func作为一个名为`target`的参数

    """

    def __init__(self, func, **kwargs):

        super(LossFunc, self).__init__()
        _check_function_or_method(func)
        self.get_loss = func
        self._init_param_map(**kwargs)


class _CrossEntropyLoss(LossBase):
    def __init__(self, pred=None, target=None, seq_len=None, class_in_dim=-1, ignore_idx=-100, reduction='mean',
                 epsilon=1.0,**kwargs):
        super(_CrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        ignore_idx = kwargs.pop('padding_idx', ignore_idx)
        self.ignore_idx = ignore_idx

        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.epsilon = epsilon
        self.class_in_dim = class_in_dim

    def get_loss(self, pred, target, batch_dict, seq_len=None, num_labels=None, add_polynominal=False):
        if seq_len is not None and target.dim() > 1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).eq(False)
            target = target.masked_fill(mask, self.ignore_idx)

        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):  # 有可能顺序替换了
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.transpose(-1, self.class_in_dim)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, reduction=self.reduction)
        ce_loss = loss_fct(input=pred, target=target)

        if add_polynominal:
            poly1 = torch.sum(F.one_hot(target, num_labels).float() * F.softmax(pred), dim=-1)
            loss = ce_loss + self.epsilon * (1 - poly1)
        else:
            loss = ce_loss

        return loss


class _FocalLoss(LossBase):
    def __init__(self, pred=None, target=None, seq_len=None, class_in_dim=-1, ignore_idx=-100, reduction='mean',
                 epsilon=1.0,**kwargs):
        super(_FocalLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        ignore_idx = kwargs.pop('padding_idx', ignore_idx)
        self.ignore_idx = ignore_idx
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.epsilon = epsilon
        self.class_in_dim = class_in_dim

    def get_loss(self, pred, target, batch_dict, seq_len=None, num_labels=None, add_polynominal=False):
        if seq_len is not None and target.dim() > 1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).eq(False)
            target = target.masked_fill(mask, self.ignore_idx)

        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):  # 有可能顺序替换了
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.transpose(-1, self.class_in_dim)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)

        loss_fct = FocalLoss(ignore_index=self.ignore_idx, reduction=self.reduction)
        focal_loss = loss_fct(input=pred, target=target)

        if add_polynominal:
            p = torch.nn.functional.sigmoid(pred)
            labels = torch.nn.functional.one_hot(target, num_labels)
            labels = torch.tensor(labels, dtype=torch.float32)
            poly1 = labels * p + (1 - labels) * (1 - p)
            loss = focal_loss + torch.mean(self.epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
        else:
            loss = focal_loss

        return loss


class _DiceLoss(LossBase):
    def __init__(self, pred=None, target=None, seq_len=None, class_in_dim=-1, ignore_idx=-100, reduction='mean',
                 epsilon=1.0, **kwargs):
        super(_DiceLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        ignore_idx = kwargs.pop('padding_idx', ignore_idx)
        self.ignore_idx = ignore_idx
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.epsilon = epsilon
        self.class_in_dim = class_in_dim

    def get_loss(self, pred, target, batch_dict, seq_len=None, num_labels=None, add_polynominal=False):
        if seq_len is not None and target.dim() > 1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).eq(False)
            target = target.masked_fill(mask, self.ignore_idx)

        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):  # 有可能顺序替换了
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.transpose(-1, self.class_in_dim)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)

        loss_fct = DiceLoss(ignore_index=self.ignore_idx, reduction=self.reduction)
        dice_loss = loss_fct(input=pred, target=target)

        if add_polynominal:
            p = torch.nn.functional.sigmoid(pred)
            labels = torch.nn.functional.one_hot(target, num_labels)
            labels = torch.tensor(labels, dtype=torch.float32)
            poly1 = labels * p + (1 - labels) * (1 - p)
            loss = dice_loss + torch.mean(self.epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
        else:
            loss = dice_loss

        return loss


class _RDropLoss(LossBase):
    def __init__(self, pred=None, target=None, seq_len=None, class_in_dim=-1, ignore_idx=-100, reduction='mean',alpha=4,
                 **kwargs):
        super(_RDropLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        ignore_idx = kwargs.pop('padding_idx', ignore_idx)
        self.ignore_idx = ignore_idx
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.class_in_dim = class_in_dim
        alpha = kwargs.pop('alpha', alpha)
        self.alpha = alpha

    def get_loss(self, pred, target, batch_dict, seq_len=None, num_labels=None, add_polynominal=False):
        if seq_len is not None and target.dim() > 1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).eq(False)
            target = target.masked_fill(mask, self.ignore_idx)

        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):  # 有可能顺序替换了
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.transpose(-1, self.class_in_dim)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)

        loss_fct = RDropLoss(_CrossEntropyLoss(), alpha=self.alpha)
        loss = loss_fct(input=pred, target=target)

        return loss

class _InfoNCELoss(LossBase):
    def __init__(self, pred=None, target=None, seq_len=None, **kwargs):
        super(_InfoNCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)


    def get_loss(self, pred, batch_dict, target=None, seq_len=None, num_labels=None):
        query, key = pred
        query = torch.div(query, torch.norm(query, dim=1).reshape(-1, 1))
        key = torch.div(key, torch.norm(key, dim=1).reshape(-1, 1))
        N, D = query.shape[0], query.shape[1]
        # calculate positive similarity
        batch_pos = torch.exp(torch.div(torch.bmm(query.view(N, 1, D), key.view(N, D, 1)).view(N, 1), 0.05))
        # calculate inner_batch all similarity
        batch_all = torch.sum(torch.exp(torch.div(torch.mm(query.view(N, D), torch.t(key)), 0.05)), dim=1)
        loss = torch.mean(-torch.log(torch.div(batch_pos, batch_all)))
        return loss

        # ori, pos = pred
        # bs = ori.size(0)
        # y_true = torch.arange(0, bs, device=ori.device)
        #
        # similarities = F.cosine_similarity(ori.unsqueeze(1), pos.unsqueeze(0), dim=2)
        # similarities = similarities - torch.eye(bs, device=ori.device) * 1e12
        # similarities = similarities / 0.05
        #
        # loss = F.cross_entropy(similarities, y_true)
        # return torch.mean(loss)


class _MultiLabelLoss(LossBase):
    def __init__(self, pred=None, target=None, seq_len=None, **kwargs):
        super(_MultiLabelLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def get_loss(self, pred, target, batch_dict, seq_len=None, num_labels=None):
        batch_size, num_labels = target.shape[:2]
        y_true = target.reshape(batch_size * num_labels, -1)  # (batch_size, num_labels, seq_len, seq_len)
        y_pred = pred.reshape(batch_size * num_labels, -1)  # (batch_size, num_labels, seq_len, seq_len)
        # y_pred, y_true = pred, target
        loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
        return loss


def prepare_losser(loss):
    if isinstance(loss, LossBase):
        return loss
    elif isinstance(loss, str):
        if loss == 'cross_entropy':
            return _CrossEntropyLoss()
        elif loss == 'focal':
            return _FocalLoss()
        elif loss == 'dice':
            return _DiceLoss()
        elif loss == 'rdrop':
            return _RDropLoss()
        elif loss == 'infoNCE':
            return _InfoNCELoss()
        elif loss == 'multi_label_loss':
            return _MultiLabelLoss()
        elif loss == 'loss_in_forward':
            return LossInForward()
        else:
            raise TypeError(f"Type of loss should be `cross_entropy, loss_in_forward, focal, dice, rdrop`, got {loss}")
    elif loss is None:
        return _CrossEntropyLoss()
    else:
        raise TypeError(f"Type of loss should be `LossBase` or `str`, got {type(loss)}")