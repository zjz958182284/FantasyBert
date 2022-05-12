#! -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset
from itertools import chain
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SequentialSampler
import random, abc
import json
import dataclasses
from typing import Any, Optional, List, Tuple, Union, Dict
from fantasybert.tokenization.tokenizers import BertTokenizer

from dataclasses import dataclass, field, asdict
def set_seed(args):
    SEED = args.SEED
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainerState:
    # must passed in parameters
    learning_rate: float = field(default_factory=float)
    n_epochs: int = field(default_factory=int)
    max_len: int = field(default_factory=int)
    batch_size: int = field(default_factory=int)
    model_path: str = field(default_factory=str)

    save_path: Optional[str] = None
    data_dir: Optional[str] = None
    tokenizer: BertTokenizer = None


    train_batch_size: int = None
    eval_batch_size: int = None
    SEED: int = 42
    metric_key: str = "accuracy"
    patience: int = 5

    global_step: int = 0
    max_steps: int = 0
    test_use_tqdm = False

    save_only_param: bool = True

    # block_shuffle_args: Dict[bool, int, Any] = field(default_factory=set_block_shuffle_args)
    use_block_shuffle: bool = False
    eval_use_block_shuffle: bool = False
    batch_in_shuffle: bool = True
    sort_bs_num: Optional[int] = None   # Optional[X] 等价于 Union[X, None]
    sort_key = None

    num_labels: int = None
    add_polynominal: bool = False

    save_step_model: bool = False

    best_metric: Optional[float] = None

    fp16: bool = False
    use_adv: Union[bool, str] = False
    use_ema: bool = False
    use_lookahead: bool = False

    gradient_clip: bool = True
    clip_value = 1.0
    clip_type: str = 'norm'

    gradient_accumulation_steps: int = 1

    warmup_ratios: float = 0.1
    weight_decay: float = 1e-3

    load_checkpoint: bool = False

    # dataloader
    num_workers = 0
    sampler = None
    drop_last: bool = False
    train_shuffle: bool = True
    eval_shuffle: bool = False
    pin_memory = None


    device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getstate__(self):
        return self.__dict__

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.sort_key is None:
            self.sort_key = lambda x: len(self.tokenizer.tokenize(x[0]))  # x[0] is text in Dataset

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        if self.batch_size is not None and self.train_batch_size is None and self.eval_batch_size is None:
            self.train_batch_size = self.eval_batch_size = self.batch_size

    def save_to_json(self, json_path: str):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    @property
    def to_dict(self):
        return asdict(self)


class DataSetGetter(Dataset):
    def __init__(self, file_path=None, datas=None):
        if isinstance(file_path, (str, list)):
            self.datas, self.total_labels = self.load_data(file_path)
        elif isinstance(datas, list):
            self.datas = datas
        else:
            raise ValueError('The input args shall be str format file_path / list format datas')
        print(f'Num samples of {file_path} is {len(self.datas)}')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    @staticmethod
    def load_data(file_path):
        return file_path

    @property
    def data(self):
        return self.datas

    @property
    def num_labels(self):
        return len(list(set(self.total_labels))) if self.total_labels is not None and self.total_labels != [] else None

    @property
    def label2id(self):
        return {l:i for i,l in enumerate(sorted(list(set(self.total_labels))))}\
            if self.total_labels is not None and self.total_labels != [] else None

    @property
    def id2label(self):
        return {i:l for i,l in enumerate(sorted(list(set(self.total_labels))))}\
            if self.total_labels is not None and self.total_labels != [] else None

    @property
    def labels_distribution(self):
        if self.total_labels is not None and self.total_labels != []:
            labels_dic = {}
            for label in self.total_labels:
                labels_dic[label] = labels_dic.get(label, 0) + 1
            total_num = sum(list(labels_dic.values()))
            label_distribution = dict((x, round((y/total_num)*100, 3)) for x, y in labels_dic.items())
            sorted_label_distribution = dict(sorted(label_distribution.items(), key=lambda x: -float(x[1])))
            final_label_distribution = {k: str(v) + '%' for k, v in sorted_label_distribution.items()}
            return final_label_distribution
        else:
            return None


class BatchIter(DataLoader):
    def __init__(self,
                 dataset: Union[DataSetGetter, Dataset],
                 sort_bs_num=None,
                 sort_key=None,
                 use_block_shuffle: bool = False,
                 batch_in_shuffle: bool = False,
                 batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None, shuffle=False,
                 **kwargs,
                 ):
        batch_sampler = batch_sampler
        if batch_sampler is not None:
            kwargs['batch_size'] = 1
            kwargs['sampler'] = None
            kwargs['drop_last'] = False

        super().__init__(dataset=dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn,
            batch_sampler=batch_sampler, shuffle=shuffle)

        assert len(dataset) > 0, 'dataset cannot be None'
        assert isinstance(dataset.datas, list), "the data attribute of DatasetGetter object must be a list"

        self.use_block_shuffle = use_block_shuffle
        self.sort_bs_num = sort_bs_num
        self.sort_key = sort_key
        self.batch_in_is_shuffle = batch_in_shuffle

    def __iter__(self):
        if self.use_block_shuffle is False:
            if self.num_workers == 0:
                return _SingleProcessDataLoaderIter(self)
            else:
                return _MultiProcessingDataLoaderIter(self)

        if self.use_block_shuffle is True:
            # self.dataset is the attribute in torch DataLoader
            self.dataset.datas = self.block_shuffle(self.dataset.datas, self.batch_size, self.sort_bs_num,
                                                       self.sort_key, self.batch_in_is_shuffle)
            if self.num_workers == 0:
                return _SingleProcessDataLoaderIter(self)
            else:
                return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, batch_in_shuffle):
        random.shuffle(data)
        # 将数据按照batch_size大小进行切分
        tail_data = [] if len(data) % batch_size == 0 else data[-(len(data) % batch_size):]
        data = data[:len(data) - len(tail_data)]
        assert len(data) % batch_size == 0
        # 获取真实排序范围
        sort_bs_num = len(data) // batch_size if sort_bs_num is None else sort_bs_num
        # 按照排序范围进行数据划分
        data = [data[i:i + sort_bs_num * batch_size] for i in range(0, len(data), sort_bs_num * batch_size)]
        # 在排序范围，根据排序函数进行降序排列
        data = [sorted(i, key=sort_key, reverse=True) for i in data]
        # 将数据根据batch_size获取batch_data
        data = list(chain(*data))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # 判断是否需要对batch_data序列进行打乱
        if batch_in_shuffle:
            random.shuffle(data)
        # 将tail_data填补回去
        data = list(chain(*data)) + tail_data
        return data





