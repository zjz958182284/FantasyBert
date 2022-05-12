#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import json
import numpy as np
from fantasybert.core.trainer import Trainer
from fantasybert.core.metrics import MetricBase
from fantasybert.core.utils import batch_sequence_padding
from fantasybert.core.data_utils import set_seed, TrainerState, DataSetGetter
from fantasybert.tokenization.tokenizers import BertTokenizer
from fantasybert.models.model_building import build_bert_model

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=10,
    max_len=100,
    batch_size=80,
    metric_key='f1',
    save_path="S:/Myproject/experiments/outputs/globalpointer_CMeEE",
    model_path="S:/Myproject/resources/NEZHA",
    data_dir="S:/Myproject/datasets/CMeEE/"
)

set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")
args.efficient_global_pointer = True

class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], []
        for d in json.load(open(args.data_dir + filename, encoding='utf-8')):
            D.append([d['text']])
            for e in d['entities']:
                start, end, label = e['start_idx'], e['end_idx'], e['type']
                total_labels.append(label)
                if start <= end:
                    D[-1].append((start, end, label))
        return D, total_labels

train_dataset, dev_dataset, test_dataset = MyDataset('train.json'), MyDataset('dev.json'), MyDataset('dev.json')
num_labels, label2id = train_dataset.num_labels, train_dataset.label2id
print(train_dataset.label2id)

def my_collate_fn(batch):
    def encoder(item):
        text = item[0]
        encoder_txt = tokenizer.encode_plus(text, max_length=args.max_len, padding=True, return_offsets_mapping=True, truncation=True)
        input_ids = encoder_txt["input_ids"]
        attention_mask = encoder_txt["attention_mask"]
        mapping = encoder_txt["offsets_mapping"]
        start_mapping = {j[0]: i-1 for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i-1 for i, j in enumerate(mapping) if j}
        return start_mapping, end_mapping, input_ids, attention_mask

    batch_input_ids, batch_attention_mask, batch_labels = [], [], []
    for item in batch:
        start_mapping, end_mapping, input_ids, attention_mask = encoder(item)
        labels = np.zeros((num_labels, args.max_len, args.max_len))
        # 生后15～18个月，肺血管即可发生不可逆性改变。 ['生', '后', '15～18', '个', '月',   分词与标注是否一致？
        for start, end, label in item[1:]:
            label_id = label2id[label]
            if start in start_mapping and end in end_mapping and start < args.max_len and end < args.max_len:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[label_id, start, end] = 1  # [start, end]的span为一个entity, 标签为label

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])

    input_ids = torch.tensor(batch_input_ids).long()
    att_mask = torch.tensor(batch_attention_mask).float()
    labels = torch.tensor(batch_sequence_padding(batch_labels, seq_dims=3)).long()

    return {'input_ids': input_ids, 'attention_mask': att_mask, 'label': labels}


class GlobalPointerModel(nn.Module):
    def __init__(self):
        super(GlobalPointerModel, self).__init__()
        self.bert, self.config = build_bert_model(args.model_path, model_name='nezha', return_config=True)
        self.inner_dim = 64
        self.dense = torch.nn.Linear(self.config.hidden_size, num_labels * self.inner_dim * 2)
        self.effcient_dense1 = torch.nn.Linear(self.config.hidden_size, self.inner_dim * 2, bias=True)
        self.effcient_dense2 = torch.nn.Linear(self.inner_dim * 2, num_labels * 2, bias=True)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices   # [seq_len, output_dim // 2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq_len, output_dim // 2, 2]
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))   # [bs, seq_len, output_dim // 2, 2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))   # [bs, seq_len, output_dim]
        embeddings = embeddings.to(args.device)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, **inputs):
        encoded_layers = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        batch_size, seq_len = encoded_layers.size(0), encoded_layers.size(1)

        if args.efficient_global_pointer:
            outputs = self.effcient_dense1(encoded_layers)
            qw, kw = outputs[..., ::2], outputs[..., 1::2]  # (batch_size, seq_len, inner_dim)

            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)  # pos_emb:(batch_size, seq_len, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1).squeeze(2)  #  (batch_size, seq_len, inner_dim)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1).squeeze(2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)  # (batch_size, seq_len, inner_dim/2, 2)
            qw2 = qw2.reshape(qw.shape)  # (batch_size, seq_len, inner_dim)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos  # (batch_size, seq_len, inner_dim)

            logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.inner_dim ** 0.5
            bias = self.effcient_dense2(outputs).transpose(1, 2) / 2  # 'bnh->bhn'
            logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits:(batch_size, num_labels, seq_len, seq_len)

        else:
            outputs = self.dense(encoded_layers)  # (batch_size, seq_len, num_labels*inner_dim*2)
            outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # tuple: (batch_size, seq_len, inner_dim*2,  ) * num_labels,
            outputs = torch.stack(outputs, dim=-2)  # (batch_size, seq_len, num_labels, inner_dim*2)
            qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # qw,kw:(batch_size, seq_len, num_labels, inner_dim)

            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)  # pos_emb:(batch_size, seq_len, inner_dim)
            # print(pos_emb[..., None, 1::2].size())  # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim/2)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)  # (batch_size, seq_len, num_labels, inner_dim/2, 2)
            qw2 = qw2.reshape(qw.shape)  # (batch_size, seq_len, num_labels, inner_dim)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos  # (batch_size, seq_len, num_labels, inner_dim)
            # 对应原文 https://spaces.ac.cn/archives/8265  公式13

            logit = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # logits:(batch_size, num_labels, seq_len, seq_len)
            logits = logit / self.inner_dim ** 0.5

            # TODO:add mask!!!

        return {'pred': logits}


class GlobalPointerMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target)
        self.p, self.r, self.f = 0, 0, 0
        self.batch_num = 0

    def evaluate(self, pred, target):
        y_pred = pred.data.cpu().numpy()
        y_true = target.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * (X + 1e-10) / (Y + Z + 1e-10), (X + 1e-10) / (Y + 1e-10), (X + 1e-10) / (Z + 1e-10)
        self.p += precision
        self.r += recall
        self.f += f1
        self.batch_num += 1

    def get_metric(self, reset=True):
        evaluate_result = {'precision': round(float(self.p) / self.batch_num, 6),
                           'recall': round(float(self.r) / self.batch_num, 6),
                           'f1': round(float(self.f) / self.batch_num, 6)}
        if reset:
            self.p, self.r, self.f = 0, 0, 0
            self.batch_num = 0
        return evaluate_result


model = GlobalPointerModel()
trainer = Trainer(model, args, train_dataset, loss='multi_label_loss',
                  dev_data=dev_dataset, test_data=test_dataset, validate_every=200,
                  collate_fn=my_collate_fn, metrics=GlobalPointerMetric())
trainer.train()
#trainer.test()
# best score on dev is 0.647427