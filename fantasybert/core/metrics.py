import inspect
import torch
import warnings
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from collections import defaultdict
from fantasybert.core.utils import compute_corrcoef
from fantasybert.core.trainer_utils import _build_args, _get_func_signature
from fantasybert.core.utils import seq_len_to_mask


class MetricBase(object):

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def set_metric_name(self, name: str):
        self._metric_name = name
        return self

    def get_metric_name(self):
        r"""
        返回metric的名称

        :return:
        """
        return self._metric_name

    def _init_param_map(self, **kwargs):
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
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def __call__(self, pred_dict, target_dict):
        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return


class AccuracyMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None, seq_len=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.total = 0
        self.acc_count = 0

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        # TODO 这里报错需要更改，因为pred是啥用户并不知道。需要告知用户真实的value
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None

        if pred.dim() == target.dim():
            if torch.numel(pred) != torch.numel(target):
                raise RuntimeError(
                    f"In {_get_func_signature(self.evaluate)}, when pred have same dimensions with target, they should have same element numbers. while target have "
                    f"element numbers:{torch.numel(target)}, pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(False), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {'accuracy': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


class ClassifyFPreRecMetric(MetricBase):
    r"""
    分类问题计算FPR值的Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）

    最后得到的metric结果为::

        {
            'f': xxx, # 这里使用f考虑以后可以计算f_beta值
            'pre': xxx,
            'rec':xxx
        }

    若only_gross=False, 即还会返回各个label的metric统计值::

        {
            'f': xxx,
            'pre': xxx,
            'rec':xxx,
            'f-label': xxx,
            'pre-label': xxx,
            'rec-label':xxx,
            ...
        }

    """

    def __init__(self, tag_vocab=None, pred=None, target=None, seq_len=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        r"""

        :param tag_vocab: 标签的 :class:`~fastNLP.Vocabulary` . 默认值为None。若为None则使用数字来作为标签内容，否则使用vocab来作为标签内容。
        :param str pred: 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用 `pred` 取数据
        :param str target: 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用 `target` 取数据
        :param str seq_len: 用该key在evaluate()时从传入dict中取出sequence length数据。为None，则使用 `seq_len` 取数据。
        :param list ignore_labels: str 组成的list. 这个list中的class不会被用于计算。例如在POS tagging时传入['NN']，则不会计算'NN'个label
        :param bool only_gross: 是否只计算总的f1, precision, recall的值；如果为False，不仅返回总的f1, pre, rec, 还会返回每个label的f1, pre, rec
        :param str f_type: `micro` 或 `macro` . `micro` :通过先计算总体的TP，FN和FP的数量，再计算f, precision, recall; `macro` : 分布计算每个类别的f, precision, recall，然后做平均（各类别f的权重相同）
        :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` . 常用为 `beta=0.5, 1, 2` 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
        """

        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._tp, self._fp, self._fn = defaultdict(int), defaultdict(int), defaultdict(int)
        # tp: truth=T, classify=T; fp: truth=T, classify=F; fn: truth=F, classify=T

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        # TODO 这里报错需要更改，因为pred是啥用户并不知道。需要告知用户真实的value
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = torch.ones_like(target).long().to(target.device)

        masks = masks.eq(1)

        if pred.dim() == target.dim():
            if torch.numel(pred) != torch.numel(target):
                raise RuntimeError(
                    f"In {_get_func_signature(self.evaluate)}, when pred have same dimensions with target, they should have same element numbers. while target have "
                    f"element numbers:{torch.numel(target)}, pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        target = target.masked_select(masks)
        pred = pred.masked_select(masks)
        target_idxes = set(target.reshape(-1).tolist())
        for target_idx in target_idxes:
            self._tp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target != target_idx, 0)).item()
            self._fp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target == target_idx, 0)).item()
            self._fn[target_idx] += torch.sum((pred != target_idx).long().masked_fill(target != target_idx, 0)).item()

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._fn.keys())
            tags.update(set(self._fp.keys()))
            tags.update(set(self._tp.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                if self.tag_vocab is not None:
                    tag_name = self.tag_vocab.to_word(tag)
                else:
                    tag_name = int(tag)
                tp = self._tp[tag]
                fn = self._fn[tag]
                fp = self._fp[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f1-{}'.format(tag_name)
                    pre_key = 'precision-{}'.format(tag_name)
                    rec_key = 'recall-{}'.format(tag_name)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f1'] = f_sum / len(tags)
                evaluate_result['precision'] = pre_sum / len(tags)
                evaluate_result['recall'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._tp.values()),
                                             sum(self._fn.values()),
                                             sum(self._fp.values()))
            evaluate_result['f1'] = f
            evaluate_result['precision'] = pre
            evaluate_result['recall'] = rec

        if reset:
            self._tp = defaultdict(int)
            self._fp = defaultdict(int)
            self._fn = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result


def _bmes_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['S-song', 'B-singer', 'M-singer', 'E-singer', 'S-moive', 'S-actor']。
    返回[('song', (0, 1)), ('singer', (1, 4)), ('moive', (4, 5)), ('actor', (5, 6))] (左闭右开区间)
    也可以是单纯的['S', 'B', 'M', 'E', 'B', 'M', 'M',...]序列

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bmeso_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bioes_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bioes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bio_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


def _get_encoding_type_from_tag_vocab(tag_vocab: dict) -> str:
    r"""
    给定Vocabulary自动判断是哪种类型的encoding, 支持判断bmes, bioes, bmeso, bio

    :param tag_vocab: 支持传入tag Vocabulary; 或者传入形如{0:"O", 1:"B-tag1"}，即index在前，tag在后的dict。
    :return:
    """
    tag_set = set()
    unk_token = '<UNK>'
    pad_token = '<PAD>'

    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support "
                       "'bio', 'bmes', 'bmeso', 'bioes' type.")


def _check_tag_vocab_and_encoding_type(tag_vocab: dict, encoding_type: str):
    r"""
    检查vocab中的tag是否与encoding_type是匹配的

    :param tag_vocab: 支持传入tag Vocabulary; 或者传入形如{0:"O", 1:"B-tag1"}，即index在前，tag在后的dict。
    :param encoding_type: bio, bmes, bioes, bmeso
    :return:
    """
    tag_set = set()
    unk_token = '<UNK>'
    pad_token = '<PAD>'
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.replace(tag, '')  # 删除该值
    if tags:  # 如果不为空，说明出现了未使用的tag
        warnings.warn(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")


class SpanFPreRecMetric(MetricBase):
    r"""
    在序列标注问题中，以span的方式计算F, pre, rec.
    比如中文Part of speech中，会以character的方式进行标注，句子 `中国在亚洲` 对应的POS可能为(以BMES为例)
    ['B-NN', 'E-NN', 'S-DET', 'B-NN', 'E-NN']。该metric就是为类似情况下的F1计算。
    最后得到的metric结果为::

        {
            'f': xxx, # 这里使用f考虑以后可以计算f_beta值
            'pre': xxx,
            'rec':xxx
        }

    若only_gross=False, 即还会返回各个label的metric统计值::

        {
            'f': xxx,
            'pre': xxx,
            'rec':xxx,
            'f-label': xxx,
            'pre-label': xxx,
            'rec-label':xxx,
            ...
        }
    """

    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
        r"""

        :param tag_vocab: 标签的 :class:`~fastNLP.Vocabulary` 。支持的标签为"B"(没有label)；或"B-xxx"(xxx为某种label，比如POS中的NN)，
            在解码时，会将相同xxx的认为是同一个label，比如['B-NN', 'E-NN']会被合并为一个'NN'.
        :param str pred: 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用 `pred` 取数据
        :param str target: 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用 `target` 取数据
        :param str seq_len: 用该key在evaluate()时从传入dict中取出sequence length数据。为None，则使用 `seq_len` 取数据。
        :param str encoding_type: 目前支持bio, bmes, bmeso, bioes。默认为None，通过tag_vocab自动判断.
        :param list ignore_labels: str 组成的list. 这个list中的class不会被用于计算。例如在POS tagging时传入['NN']，则不会计算'NN'个label
        :param bool only_gross: 是否只计算总的f1, precision, recall的值；如果为False，不仅返回总的f1, pre, rec, 还会返回每个label的f1, pre, rec
        :param str f_type: `micro` 或 `macro` . `micro` :通过先计算总体的TP，FN和FP的数量，再计算f, precision, recall; `macro` : 分布计算每个类别的f, precision, recall，然后做平均（各类别f的权重相同）
        :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` . 常用为 `beta=0.5, 1, 2` 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
        """
        #
        # if not isinstance(tag_vocab, Vocabulary):
        #     raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        if encoding_type:
            encoding_type = encoding_type.lower()
            _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)
            self.encoding_type = encoding_type
        else:
            self.encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)

        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso', 'bioes' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len=None):
        r"""evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param pred: [batch, seq_len] 或者 [batch, seq_len, len(tag_vocab)], 预测的结果
        :param target: [batch, seq_len], 真实值
        :param seq_len: [batch] 文本长度标记
        :return:
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        # if not isinstance(seq_len, torch.Tensor):
        #     raise TypeError(f"`seq_len` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
        #                     f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        seq_len = pred.size(1)

        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len)]
            gold_tags = target[i][:int(seq_len)]
            # pred_tags = pred[i][:int(seq_len[i])]
            # gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab[tag] for tag in pred_tags]
            gold_str_tags = [self.tag_vocab[tag] for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        r"""get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f1-{}'.format(tag)
                    pre_key = 'precision-{}'.format(tag)
                    rec_key = 'recall-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f1'] = f_sum / len(tags)
                evaluate_result['precision'] = pre_sum / len(tags)
                evaluate_result['recall'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._true_positives.values()),
                                             sum(self._false_negatives.values()),
                                             sum(self._false_positives.values()))
            evaluate_result['f1'] = f
            evaluate_result['precision'] = pre
            evaluate_result['recall'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


class SpearmanMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.spearman = 0
        self.batch_num = 0

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred[0], torch.Tensor):
            raise TypeError(f"value in `pred` in must be torch.Tensor, got {type(pred[0])}.")

        vec_a, vec_b = pred[:2]
        sims = F.cosine_similarity(vec_a, vec_b).cpu().numpy()
        labels = target.cpu().numpy()
        spearman_correlation = compute_corrcoef(sims, labels)
        self.spearman += spearman_correlation
        self.batch_num += 1

    def get_metric(self, reset=True):
        evaluate_result = {'spearman_correlation': round(float(self.spearman) / (self.batch_num + 1e-12), 6)}
        if reset:
            self.spearman = 0
            self.batch_num = 0
        return evaluate_result


def _prepare_metrics(metrics):
    r"""

    Prepare list of Metric based on input
    :param metrics:
    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                    _metrics.append(metric)

                elif isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)

                elif isinstance(metric, str):
                    if metric == 'acc':
                        metric = AccuracyMetric()
                    elif metric == 'fpr':
                        metric = ClassifyFPreRecMetric()
                    elif metrics == 'spearman':
                        _metrics = SpearmanMetric()
                    elif metric == 'spanfpr':
                        metric = SpanFPreRecMetric()  # 要传入vocab
                    else:
                        raise TypeError(f'expected to be str among acc, fpr, spearman_correlation and spanfpr,'
                                        f' got str {metrics}')
                    _metrics.append(metric)

                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `MetricBase` or str, not `{type(metric)}`.")
        elif isinstance(metrics, str):
            if metrics == 'acc':
                _metrics = AccuracyMetric()
            elif metrics == 'fpr':
                _metrics = ClassifyFPreRecMetric()
            elif metrics == 'spearman':
                _metrics = SpearmanMetric()
            elif metrics == 'spanfpr':
                _metrics = SpanFPreRecMetric()
            else:
                raise TypeError(f'expected to be str among acc, fpr and spanfpr, got str {metrics}')

        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics

