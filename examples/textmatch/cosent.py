#! -*- coding: utf-8 -*-
import torch
import json
import torch.nn as nn
from fantasybert.models.model_building import build_bert_model
from fantasybert.models.modeling_utils import Bertpooling
from fantasybert.core.trainer import Trainer
from fantasybert.core.loss_main import LossBase
from fantasybert.core.data_utils import set_seed, TrainerState, DataSetGetter
from fantasybert.tokenization.tokenizers import BertTokenizer

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=10,
    max_len=120,
    batch_size=64,
    patience=10,
    metric_key='spearman_correlation',
    save_path="S:/Myproject/experiments/outputs/consent_afqmc",
    model_path="S:/Myproject/resources/chinese-roberta-wwm-ext",
    data_dir="S:/Myproject/datasets/afqmc/"
)

set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")

class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], None
        with open(args.data_dir + filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                text_a = line['sentence1']
                text_b = line['sentence2']
                label = line['label']
                D.append((text_a, int(label)))
                D.append((text_b, int(label)))
        return D, total_labels

class TestDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], None
        with open(args.data_dir + filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                text_a = line['sentence1']
                text_b = line['sentence2']
                label = line['label']
                D.append((text_a, text_b, float(label)))
        return D, total_labels

train_dataset, dev_dataset = MyDataset('train.json'), TestDataset('dev.json')


def train_collate_fn(batch):
    texts, labels = [t[0] for t in batch], [t[1] for t in batch]
    encodings = tokenizer(texts, max_length=args.max_len, padding='batch_longest', truncation=True, return_tensors='pt')
    return {'input_ids': encodings['input_ids'], 'label': torch.tensor(labels)}

def eval_collate_fn(batch):
    text_a, text_b, labels = [_[0] for _ in batch], [_[1] for _ in batch], [_[2] for _ in batch]
    encodings_a = tokenizer(text_a, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    encodings_b = tokenizer(text_b, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    return {'text_a_input_ids': encodings_a["input_ids"],'text_b_input_ids': encodings_b["input_ids"],'label':torch.tensor(labels)}


class CosentModel(nn.Module):
    def __init__(self):
        super(CosentModel, self).__init__()
        self.bert = build_bert_model(args.model_path)

    def forward(self, input_ids=None, **inputs):
        output = self.bert(input_ids=input_ids, output_hidden_states=True)
        pooled_output = Bertpooling(output, pool_type='first-last-avg')
        return {'pred': pooled_output}

    def predict(self, text_a_input_ids, text_b_input_ids):
        out_a = self.bert(input_ids=text_a_input_ids, output_hidden_states=True)
        vec_a = Bertpooling(out_a, pool_type='first-last-avg')
        out_b = self.bert(input_ids=text_b_input_ids, output_hidden_states=True)
        vec_b = Bertpooling(out_b, pool_type='first-last-avg')
        return {'pred': (vec_a, vec_b)}


class CosentLoss(LossBase):
    def __init__(self, pred=None, target=None, **kwargs):
        super(CosentLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)

    def compute_loss(self, y_pred, y_true):
        # 1. 取出真实的标签
        y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签

        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
        # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
        y_pred = y_pred / norms

        # 3. 奇偶向量相乘
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20  # 奇/偶数位分别是是句子对的两个句子

        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值

        # print(y_true[:, None])   [[1.],[0.],[0.],[1.],...]   (1,bs)
        # print(y_true[None, :])   [[1., 0., 0., 1.,...]]      (bs,1)
        # None是增加维度的作用，类似unsqueeze
        y_true = y_true[:, None] < y_true[None, :]  # 例如对于上面的[1.]就和[1., 0., 0., 1.,...]进行比较得到一行
        # print(y_true.size())    bs*bs  里面的值为True/False
        y_true = y_true.float()  # True/False转为1/0

        y_pred = y_pred - (1 - y_true) * 1e12
        # 标签为1的sample cosine值应大于标签为0的sample
        # 具体就是标签为0与标签为1的值为cosine差值，标签相同值为-e12，不用更新loss

        y_pred = y_pred.view(-1)
        if torch.cuda.is_available():
            y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred),
                               dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1，对应blog公式(6)
        else:
            y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)

        return torch.logsumexp(y_pred, dim=0)

    def get_loss(self, pred, target, batch_dict):
        loss = self.compute_loss(pred, target)
        return loss


model = CosentModel()
trainer = Trainer(model, args, train_dataset, loss=CosentLoss(), eval_loss='none',
                  dev_data=dev_dataset, validate_every=100,
                  train_collate_fn=train_collate_fn, eval_collate_fn=eval_collate_fn, metrics='spearman')


if __name__ == "__main__":
    trainer.train()
    # lcqmc  loss7.39  0.7906852422718601
    # bert4torch lqcmc 0.79386  loss 7.898897
