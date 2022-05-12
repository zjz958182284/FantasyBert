import copy
import numpy as np
import itertools
import unicodedata
import re
from fantasybert.tokenization.tokenization_utils import to_py_obj
from typing import Dict, List, Optional, Tuple, Union
from fantasybert.tokenization.tokenization_utils import PaddingStrategy, TensorType, TruncationStrategy, _is_control,\
    SpecialTokensMixin, BatchEncoding, AddedToken, _insert_one_token_to_ordered_list, _is_end_of_word, _is_start_of_word

TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER


class TokenizerBase(SpecialTokensMixin):
    def __init__(self, **kwargs):
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)

        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []

        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        super().__init__(**kwargs)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)


        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:              # Or on the newly added tokens
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))

        return len(tokens_to_add)

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def do_lower_case(self) -> bool:
        raise NotImplementedError

    def __len__(self):
        return self.vocab_size + len(self.added_tokens_encoder)

    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        all_special_tokens_extended = dict((str(t), t) for t in self.all_special_tokens if isinstance(t, AddedToken))

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        # Try to avoid splitting on token
                        if (
                                i < len(split_text) - 1
                                and not _is_end_of_word(sub_text)
                                and not _is_start_of_word(split_text[i + 1])
                        ):
                            # Don't extract the special token
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result.append(full_word)
                            full_word = ""
                            continue
                    # Strip white spaces on the right
                    if tok_extended.rstrip and i > 0:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        sub_text = sub_text.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                    if i > 0:
                        sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_no_split_tokens else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def _tokenize(self, text):
        raise NotImplementedError

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:

        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> List[int]:

        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs)
        return encoded_inputs["input_ids"]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        # 统计token个数
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=False, max_length=None, **kwargs
    ):
        """
        获取 padding 和 truncation的方式
        """
        padding_strategy, truncation_strategy = None, None

        if max_length is not None and padding is False and truncation is False:
            # 未给定最大句长和不进行padding时，则必须要进行截断
            truncation_strategy = (TruncationStrategy.LONGEST_FIRST)  # 当输入多个句子时，对最长的句子进行截断

        # Get padding strategy
        if padding is False:
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST  # 若未给出最大句长，则默认根据batch中最长句子长度进行padding
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):  # padding不是PaddingStrategy类型,则为str，传入PaddingStrategy
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:

                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length
        return padding_strategy, truncation_strategy, max_length, kwargs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,  # 是否添加[CLS],[SEP]等特殊token
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充 可填入的值为True/False/longest/batch_longest/max_length
        truncation: Union[bool, str, TruncationStrategy] = False,  # 是否进行截断 可填入的值为True/False/only_first/only_second/longest_first
        max_length: Optional[int] = None,  # 句子的最大长度，用于短填长切
        stride: int = 0,  #截断时使用的滑窗
        is_split_into_words: bool = False,  #  输入是否已经分好词了
        return_tensors: Optional[Union[str, TensorType]] = None,  # True返回tensor else List
        return_overflowing_tokens: bool = False,   # 是否返回截断超出的tokens
        return_special_tokens_mask: bool = False,  # 针对特殊token的mask，对应位置为1
        return_offsets_mapping: bool = False,    # 是否返回 word位置:token位置  (英文一个word可能会被拆分为几个token(subword))
        **kwargs) -> BatchEncoding:

        # 检查输入格式错误
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple))
            and (
                len(text) == 0
                or (
                    isinstance(text[0], str)
                    or (isinstance(text[0], (list, tuple)) and (len(text[0]) == 0 or isinstance(text[0][0], str)))
                )
            )
        ), (
            "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples)."
        )

        assert (
            text_pair is None
            or isinstance(text_pair, str)
            or (
                isinstance(text_pair, (list, tuple))
                and (
                    len(text_pair) == 0
                    or (
                        isinstance(text_pair[0], str)
                        or (
                            isinstance(text_pair[0], (list, tuple))
                            and (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str))
                        )
                    )
                )
            )
        ), ("text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple)))
            or (
                is_split_into_words and isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
            )
        )

        if is_batched:  # 判断输入的是一个batch还是单个序列
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                return_tensors=return_tensors,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                return_tensors=return_tensors,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                **kwargs,
            )

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ) -> BatchEncoding:

        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

        first_offsets_mapping, second_offsets_mapping, offsets_mapping = None, None, None
        if return_offsets_mapping:
            first_offsets_mapping = self.rematch(text, self._tokenize(text))
            second_offsets_mapping = self.rematch(text_pair, self._tokenize(text_pair)) if text_pair is not None else None
        offsets_mapping = ((first_offsets_mapping, second_offsets_mapping)) if second_offsets_mapping else first_offsets_mapping

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        batch_outputs = self.prepare_for_model(
            first_ids=first_ids,
            second_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            offsets_mapping=offsets_mapping,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping = return_offsets_mapping,
            return_tensors=return_tensors,
            prepend_batch_axis=False,
        )
        return batch_outputs


    def prepare_for_model(
        self,
        first_ids: List[int],
        second_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        offsets_mapping=None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        pair = bool(second_ids is not None)  # 输入的是句子还是句子对
        len_ids = len(first_ids)
        len_second_ids = len(second_ids) if pair else 0

        encoded_inputs = {}
        total_len = len_ids + len_second_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            first_ids, second_ids, overflowing_tokens = self.truncate_sequences(
                first_ids,
                second_ids=second_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length if total_len - max_length>=0 else 0

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(first_ids, second_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(first_ids, second_ids)
            attention_mask = [1] * len(sequence)
        else:
            sequence = first_ids + second_ids if pair else first_ids
            token_type_ids = [0] * len(first_ids) + ([0] * len(second_ids) if pair else [])
            attention_mask = [1] * len(sequence)

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids
        encoded_inputs["attention_mask"] = attention_mask
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(first_ids, second_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)


        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            encoded_inputs = self.pad(encoded_inputs, max_length=max_length, padding_strategy=padding_strategy)

        if return_offsets_mapping:
            encoded_inputs['offsets_mapping'] = offsets_mapping


        batch_outputs = BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)
        return batch_outputs

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ) -> BatchEncoding:

        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs)

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text)))
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

        offsets_mapping = [] if return_offsets_mapping else None
        input_ids = []
        for first_ids_or_second_ids in batch_text_or_text_pairs:
            if not isinstance(first_ids_or_second_ids, (list, tuple)):
                first_ids, second_ids = first_ids_or_second_ids, None
            elif is_split_into_words and not isinstance(first_ids_or_second_ids[0], (list, tuple)):
                first_ids, second_ids = first_ids_or_second_ids, None
            else:
                first_ids, second_ids = first_ids_or_second_ids

            if return_offsets_mapping:    # first_ids, second_ids此时为token
                first_ids_mapping = self.rematch(first_ids, self._tokenize(first_ids))
                second_ids_mapping = self.rematch(second_ids, self._tokenize(second_ids)) if second_ids is not None else None
                if second_ids is not None:
                    offsets_mapping.append((first_ids_mapping, second_ids_mapping))
                else:
                    offsets_mapping.append(first_ids_mapping)

            first_ids = get_input_ids(first_ids)
            second_ids = get_input_ids(second_ids) if second_ids is not None else None
            input_ids.append((first_ids, second_ids))   # token_id

        if padding_strategy == PaddingStrategy.BATCH_LONGEST:
            if add_special_tokens:
                batch_max_length = max([len(first_ids + second_ids)+3 if second_ids is not None else len(first_ids)+2
                                      for first_ids, second_ids in input_ids])
                max_length = max_length if batch_max_length > max_length else batch_max_length
            else:
                batch_max_length = max([len(first_ids + second_ids) if second_ids is not None else len(first_ids)
                                      for first_ids, second_ids in input_ids])
                max_length = max_length if batch_max_length > max_length else batch_max_length
        else:
            max_length = max_length

        batch_outputs = {}
        for first_ids, second_ids in input_ids:
            outputs = self.prepare_for_model(
                first_ids=first_ids,
                second_ids=second_ids,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                offsets_mapping=offsets_mapping,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,
                return_offsets_mapping=return_offsets_mapping,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False)

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        return batch_outputs

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding_strategy: Union[PaddingStrategy] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:

        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):  # 如果是batch
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        required_input = encoded_inputs["input_ids"]
        if required_input and not isinstance(required_input[0], (list, tuple)): # required_input[0]不是list,则代表输入的不是一个batch

            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy)
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        # batch_size = len(required_input)
        # assert all(
        #     len(v) == batch_size for v in encoded_inputs.values()
        # ), "Some items in the output dictionary have a different batch size than others."
        #
        # if padding_strategy == PaddingStrategy.LONGEST:
        #     max_length = max(len(inputs) for inputs in required_input)
        #     padding_strategy = PaddingStrategy.MAX_LENGTH
        #
        # batch_outputs = {}
        # for i in range(batch_size):
        #     inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        #     outputs = self._pad(inputs, max_length=max_length, padding_strategy=padding_strategy)
        #
        #     for key, value in outputs.items():
        #         if key not in batch_outputs:
        #             batch_outputs[key] = []
        #         batch_outputs[key].append(value)
        #
        # return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def truncate_sequences(
        self,
        ids: List[int],
        second_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:

        if num_tokens_to_remove <= 0:
            return ids, second_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if second_ids is None or len(ids) > len(second_ids):
                    if not overflowing_tokens:
                        window_len = min(len(ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(ids[-window_len:])
                    ids = ids[:-1]
                else:
                    if not overflowing_tokens:
                        window_len = min(len(second_ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(second_ids[-window_len:])
                    second_ids = second_ids[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]

        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and second_ids is not None:
            if len(second_ids) > num_tokens_to_remove:
                window_len = min(len(second_ids), stride + num_tokens_to_remove)
                overflowing_tokens = second_ids[-window_len:]
                second_ids = second_ids[:-num_tokens_to_remove]

        return (ids, second_ids, overflowing_tokens)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    ) -> dict:
        required_input = encoded_inputs["input_ids"]
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) < max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            encoded_inputs["token_type_ids"] = (encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference)
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs["input_ids"] = required_input + [self.pad_token_id] * difference

        return encoded_inputs

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True) -> List[str]:
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                spaces_between_special_tokens=spaces_between_special_tokens,)
            for seq in sequences]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True) -> str:

        token_ids = to_py_obj(token_ids)
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        """
        针对特殊token的mask，对应位置为1，否则为0
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument."
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )
        all_special_ids = self.all_special_ids  # cache the property
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        清理掉一些空格
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    @staticmethod  # 该方法不强制要求传递参数
    def stem(token):  # 获取token的“词干”（如果是##开头，则自动去掉##）
            if token[:2] == '##':
                return token[2:]
            else:
                return token

    @staticmethod
    def _is_special(ch):  # 判断是不是有特殊含义的符号
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self.do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self.do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping



