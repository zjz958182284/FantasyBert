from enum import Enum
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, Sequence, Any
from collections import OrderedDict, UserDict
import unicodedata

class ExplicitEnum(Enum):
    """
    Enum：python枚举类
    枚举类中不能存在相同的标签名
    枚举成员为单例，不可实例化，不可更改
    枚举是可迭代的
    使用__members__获取属性（如下面的LONGEST, MAX_LENGTH...）
    """
    @classmethod  # classmethod修饰符对应的函数不需要实例化类，不需要self参数，但第一个参数需要是表示自身类的 cls 参数
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"
    BATCH_LONGEST = 'batch_longest'


class TensorType(ExplicitEnum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class TruncationStrategy(ExplicitEnum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


@dataclass(frozen=True, eq=True)
class AddedToken:
    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__


class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens"]
    def __init__(self, verbose=True, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(isinstance(t, str) for t in value), "One of the tokens is not a string"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"special token {key} has to be either str or AddedToken but got: {type(value)}")

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, List[str], AddedToken]]) -> int:
        """
        增加新的特殊token，该token不会被split
        Examples:
            special_tokens_dict = {'additional_special_tokens': '[X]'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            # 注意: 新增token后需要resize_token_embeddings！！！
            model.resize_token_embeddings(len(tokenizer))
            assert tokenizer.additional_special_tokens == '[X]'
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            setattr(self, key, value)
            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(value, (str, AddedToken)), f"Token {value} for key {key} should be a str or an AddedToken instance"
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        """
        向vocabulary中增加新tokens
        注意: 新增token后需要resize_token_embeddings！！！
        model.resize_token_embeddings(len(tokenizer))

        Args:
            new_tokens 新增的token，只有当vocabulary中没有的才会被加入
            special_tokens 指定新增的token是否为特殊的
        """
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property  # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加（）, 如tokenizer.bos_token,且该属性不可更改
    def bos_token(self) -> str:
        """
        句子开始的token
        """
        return str(self._bos_token) if self._bos_token is not None else None

    @property
    def eos_token(self) -> str:
        """
        句子结束的token
        """
        return str(self._bos_token) if self._bos_token is not None else None

    @property
    def unk_token(self) -> str:
        """
        Unknown token
        """
        return str(self._unk_token) if self._unk_token is not None else None

    @property
    def sep_token(self) -> str:
        """
        句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有多句话作为输入时，此标记作为分隔符、最后一句话的结束符
        """
        return str(self._sep_token) if self._sep_token is not None else None

    @property
    def pad_token(self) -> str:
        """
        填充句子长度的 token
        """
        return str(self._pad_token) if self._pad_token is not None else None

    @property
    def cls_token(self) -> str:
        """
        分类token，位于序列第一个，可用该token向量代替整句向量
        """
        return str(self._cls_token) if self._cls_token is not None else None

    @property
    def mask_token(self) -> str:
        """
        MLM时的[MASK] token
        """
        return str(self._mask_token) if self._mask_token is not None else None

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        新增的特殊token
        """
        return [str(tok) for tok in self._additional_special_tokens] if self._additional_special_tokens is not None else None

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_tokens_to_ids(value)

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_tokens_to_ids(value)

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_tokens_to_ids(value)

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_tokens_to_ids(value)

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_tokens_to_ids(value)

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_tokens_to_ids(value)

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_tokens_to_ids(value)

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_tokens_to_ids(value) for value in values]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        special_tokens字典 {名称：值}
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[Union[str, AddedToken]]:
        """
        所有特殊的token
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        所有特殊的token对应的id
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids


class BatchEncoding(UserDict):
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[Any, Sequence[Any]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False):
        super().__init__(data)
        self._encodings = encoding
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    def __getitem__(self, item: Union[int, str]) -> Union[Any, Any]:
        """
        如果key是str，则返回对应的dict ('input_ids', 'attention_mask',etc.).
        如果可以是int，则返回encoding
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError("Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers")

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    @property
    def encodings(self) -> Optional[List[Any]]:
        """
        :obj:`Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns
        :obj:`None` if the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False):
        if tensor_type is None:
            return self
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        if tensor_type == TensorType.TENSORFLOW:
            import tensorflow as tf
            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            import torch
            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = lambda x: isinstance(x, np.ndarray)

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    使用二分查找的方法，将一个token插入到排序好的list中
    """
    import bisect
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # new_token is in token_list, don't add
        return
    else:
        token_list.insert(insertion_idx, new_token)


def _is_end_of_word(text):
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def string_matching(s, keywords):
    """判断s是否至少包含keywords中的至少一个字符串
    """
    for k in keywords:
        if re.search(k, s):
            return True
    return False


def _is_torch(x):
    import torch
    return isinstance(x, torch.Tensor)


def to_py_obj(obj):
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif _is_torch(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

#
# required_input = encoded_inputs["input_ids"]

# if not required_input:
#     if return_attention_mask:
#         encoded_inputs["attention_mask"] = []
#     return encoded_inputs

# first_element = required_input[0]
# if isinstance(first_element, (list, tuple)):
#     # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
#     index = 0
#     while len(required_input[index]) == 0:
#         index += 1
#     if index < len(required_input):
#         first_element = required_input[index][0]

# if not isinstance(first_element, (int, list, tuple)):
#     if _is_tensorflow(first_element):
#         return_tensors = "tf" if return_tensors is None else return_tensors
#     elif _is_torch(first_element):
#         return_tensors = "pt" if return_tensors is None else return_tensors
#     elif isinstance(first_element, np.ndarray):
#         return_tensors = "np" if return_tensors is None else return_tensors
#     else:
#         raise ValueError(f"type of {first_element} unknown: {type(first_element)},Should be one of a python, numpy, pytorch or tensorflow object.")
#
#     for key, value in encoded_inputs.items():
#         encoded_inputs[key] = to_py_obj(value)

