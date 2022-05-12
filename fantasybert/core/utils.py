#! -*- coding: utf-8 -*-

import os, sys, six, re, json
from collections import OrderedDict, UserDict
from dataclasses import fields
import inspect
import torch
import time
import functools
import torch.nn.functional as F
import json
import os
import torch.nn as nn
from enum import Enum
import random
import numpy as np
from collections import defaultdict
import torch
from dataclasses import dataclass, field, asdict
import inspect
from fantasybert.core.logger_utils import logger
_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str


@dataclass
class Const:
    INPUT = 'words'
    CHAR_INPUT = 'chars'
    INPUT_LEN = 'seq_len'
    OUTPUT = 'pred'
    TARGET = 'target'
    LOSS = 'loss'
    RAW_WORD = 'raw_words'
    RAW_CHAR = 'raw_chars'

    @staticmethod
    def INPUTS(i):
        r"""得到第 i 个 ``INPUT`` 的命名"""
        i = int(i) + 1
        return Const.INPUT + str(i)

    @staticmethod
    def CHAR_INPUTS(i):
        r"""得到第 i 个 ``CHAR_INPUT`` 的命名"""
        i = int(i) + 1
        return Const.CHAR_INPUT + str(i)

    @staticmethod
    def RAW_WORDS(i):
        r"""得到第 i 个 ``RAW_WORDS`` 的命名"""
        i = int(i) + 1
        return Const.RAW_WORD + str(i)

    @staticmethod
    def RAW_CHARS(i):
        r"""得到第 i 个 ``RAW_CHARS`` 的命名"""
        i = int(i) + 1
        return Const.RAW_CHAR + str(i)

    @staticmethod
    def INPUT_LENS(i):
        r"""得到第 i 个 ``INPUT_LEN`` 的命名"""
        i = int(i) + 1
        return Const.INPUT_LEN + str(i)

    @staticmethod
    def OUTPUTS(i):
        r"""得到第 i 个 ``OUTPUT`` 的命名"""
        i = int(i) + 1
        return Const.OUTPUT + str(i)

    @staticmethod
    def TARGETS(i):
        r"""得到第 i 个 ``TARGET`` 的命名"""
        i = int(i) + 1
        return Const.TARGET + str(i)

    @staticmethod
    def LOSSES(i):
        r"""得到第 i 个 ``LOSS`` 的命名"""
        i = int(i) + 1
        return Const.LOSS + str(i)


def seq_len_to_mask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask



# calculate running time for the function
def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('{} took {} ms'.format(func.__name__, (end - start) * 1000))
        return res

    return wrapper


def batch_sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def longest_common_substring(source, target):
    """最长公共子串（source和target的最长公共切片区间）
    返回：子串长度, 所在区间（四元组）
    注意：最长公共子串可能不止一个，所返回的区间只代表其中一个。
    """
    c, l, span = defaultdict(int), 0, (0, 0, 0, 0)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
                if c[i, j] > l:
                    l = c[i, j]
                    span = (i - l, i, j - l, j)
    return l, span


def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    import scipy.stats
    return scipy.stats.spearmanr(x, y).correlation


class DataAugment(object):
    def __init__(self, ):
        self.PUNCTUATIONS = ['.', ',', ' ', '。', '，', '、', '？']

    def insert_punctuation_marks(self, sentence, punc_ratio=0.3):
        import jieba
        words = [word for word in jieba.cut(sentence)]  # jieba分词，往词中间随机加标点
        new_line = []
        q = random.randint(1, int(punc_ratio * len(words) + 1))
        qs = random.sample(range(0, len(words)), q)

        for j, word in enumerate(words):
            if j in qs:
                new_line.append(self.PUNCTUATIONS[random.randint(0, len(self.PUNCTUATIONS) - 1)])
                new_line.append(word)
            else:
                new_line.append(word)
        new_line = ' '.join(new_line)
        return new_line

    def aeda_zh(self, input_text, punc_ratio=0.3):
        if isinstance(input_text, list) and len(input_text) > 1:
            total = []
            for text in input_text:
                total.append(self.insert_punctuation_marks(text, punc_ratio))
            return total

        elif isinstance(input_text, tuple) and len(input_text) > 1:
            total = []
            for text in input_text:
                total.append(self.insert_punctuation_marks(text, punc_ratio))
            return total

        elif isinstance(input_text, str):
            return self.insert_punctuation_marks(input_text, punc_ratio)

        else:
            raise AttributeError('Can only get input type of list, tuple and str')

    @classmethod
    def token_level_cutoff(self, input_ids, attention_mask, cutoff_rate=0.15, cutoff_type='token'):
        bsz, seq_len = input_ids.shape
        _input_ids = []
        cutoff_attention_mask = []

        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            _input_id = None
            # num_tokens = sample_mask.sum().int().item()
            num_tokens = sample_mask.sum().item()  # 当前句子中的token数
            cur_input_id = input_ids[bsz_id]  # 当前的input_ids

            if cutoff_type == 'span':
                sample_len = max(int(num_tokens * (1 - cutoff_rate)),
                                 1)  # if true_len is 32, cutoff_rate is 0.15 then sample_len is 27
                start_id = np.random.randint(1, high=num_tokens - sample_len + 1)  # 从1开始选择span，避免删除CLS, 同时也要避免超过句子长度
                cutoff_mask = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    cutoff_mask[idx] = 0  # 这些位置是0，bool之后就变成了False，而masked_fill是选择True的位置替换为value的

                cutoff_mask[0] = 0  # 避免CLS被替换
                cutoff_mask[num_tokens - 1] = 0  # 避免SEP被替换

            elif cutoff_type == 'token':
                cutoff_token_num = max(int(num_tokens * cutoff_rate), 1)
                cutoff_mask = [0] * seq_len
                idxs = random.sample([idx for idx in range(1, num_tokens - 1)], cutoff_token_num)
                # 不重复的选取cutoff_token_num个token的idx，同时避免选取CLS，SEP
                for idx in idxs:
                    cutoff_mask[idx] = 1  # 1对应True，masked_fill会把对应True的位置替换为0

            else:
                raise TypeError('Please type the right cutoff type!!!')

            cutoff_mask = torch.ByteTensor(cutoff_mask).bool().cuda()
            _input_id = cur_input_id.masked_fill(cutoff_mask, value=0).cuda()
            sample_mask = sample_mask.masked_fill(cutoff_mask, value=0).cuda()

            _input_id = _input_id.view(1, -1)  # 增加一个维度
            sample_mask = sample_mask.view(1, -1)

            _input_ids.append(_input_id)
            cutoff_attention_mask.append(sample_mask)

        _input_ids = torch.cat(_input_ids, dim=0)
        cutoff_attention_mask = torch.cat(cutoff_attention_mask, dim=0)
        return _input_ids, cutoff_attention_mask

    @classmethod
    def vec_level_cutoff(self, embeddings, cutoff_rate=0.05):
        bs, hidden_size = embeddings.shape
        cutoff_embed_num = max(int(hidden_size * cutoff_rate), 1)
        cut_embeddings = []
        for bs_id in range(bs):
            cutoff_mask = [1] * hidden_size
            embed = embeddings[bs_id]
            idxs = random.sample([idx for idx in range(hidden_size)], cutoff_embed_num)
            for idx in idxs:
                cutoff_mask[idx] = 0  # 1原来的向量值不变，0则相乘为0

            cutoff_mask = torch.tensor(cutoff_mask).cuda()
            embed = embed * cutoff_mask
            embed = embed.view(1, -1)
            cut_embeddings.append(embed)

        cut_embeddings = torch.cat(cut_embeddings, dim=0)
        return cut_embeddings

    # TODO: eng有点问题，应该空格分开,另添加标点符号，token cutoff等
    def word_repetition_eng(self, input_text, dup_rate):
        def repetition_for_one(text, dup_rate):
            actual_len = len(text)
            dup_len = random.randint(a=0, b=max(
                2, int(dup_rate * actual_len)))
            dup_word_index = random.sample(
                list(range(1, actual_len)), k=dup_len)

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            return dup_text

        if isinstance(input_text, list) and len(input_text) > 1:
            total = []
            for text in input_text:
                total.append(repetition_for_one(text, dup_rate))
            return total

        elif isinstance(input_text, tuple) and len(input_text) > 1:
            total = []
            for text in input_text:
                total.append(repetition_for_one(text, dup_rate))
            return total

        elif isinstance(input_text, str):
            return repetition_for_one(input_text, dup_rate)

        else:
            raise AttributeError('Can only get input type of list, tuple and str')

    @classmethod
    def word_repetition_chinese(self, input_text, dup_rate):
        ''' span duplicated for chinese
        '''
        import jieba
        def repetition_for_one(text, dup_rate):
            ori_text = text
            cut_text = jieba.cut(text, cut_all=False)
            text = list(cut_text)
            actual_len = len(text)
            if actual_len <= 4:
                return ori_text

            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            dup_word_index = random.sample(list(range(1, actual_len)), k=dup_len)

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            return dup_text

        if isinstance(input_text, list) and len(input_text) > 1:
            total = []
            for text in input_text:
                total.append(repetition_for_one(text, dup_rate))
            return total

        elif isinstance(input_text, tuple) and len(input_text) > 1:
            total = []
            for text in input_text:
                total.append(repetition_for_one(text, dup_rate))
            return total

        elif isinstance(input_text, str):
            return repetition_for_one(input_text, dup_rate)

        else:
            raise AttributeError('Can only get input type of list, tuple and str')


def get_kw(cls, kwargs):
    '''保留排除cls的入参后的kwargs
    '''
    kwargs_new = {}
    for k in kwargs:
        if k not in set(inspect.getargspec(cls)[0]):
            kwargs_new[k] = kwargs[k]
    return kwargs_new



def kl_distance(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), -1)
    return torch.mean(_kl)



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


def _move_dict_value_to_device(*args, device: torch.device, non_blocking=False):
    r"""

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param non_blocking: bool, 是否异步将数据转移到cpu, 需要tensor使用pin_memory()
    :param args:
    :return:
    """
    if not torch.cuda.is_available() or device is None:
        return

    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device, non_blocking=non_blocking)
        else:
            raise TypeError(f"Only support `dict` type right now got type {type(arg)}")


class ExplicitEnum(Enum):
    @classmethod  # classmethod修饰符对应的函数不需要实例化类，不需要self参数，但第一个参数需要是表示自身类的 cls 参数
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )



def _is_function_or_method(func):
    r"""

    :param func:
    :return:
    """
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        return False
    return True


def _check_function_or_method(func):
    if not _is_function_or_method(func):
        raise TypeError(f"{type(func)} is not a method or function.")