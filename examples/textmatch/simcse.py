#! -*- coding: utf-8 -*-
import torch
from fantasybert.models.model_building import build_bert_model
from fantasybert.models.modeling_utils import Bertpooling
from fantasybert.core.trainer import Trainer
from fantasybert.core.data_utils import set_seed, TrainerState, DataSetGetter
from fantasybert.tokenization.tokenizers import BertTokenizer

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=10,
    max_len=120,
    batch_size=64,
    patience=10,
    metric_key='spearman_correlation',
    save_path="S:/bert4torch/experiments/outputs/simcse_stsb",
    model_path="S:/bert4torch/resources/chinese-roberta-wwm-ext",
    data_dir="S:/bert4torch/datasets/STS-B/"
)


set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")


class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], None
        with open(args.data_dir + filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                D.append(line.strip('\n'))
        return D, total_labels

class TestDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], None
        with open(args.data_dir + filename, "r", encoding="utf-8-sig") as f:
            for line in f.readlines():
                cache = line.split('||')
                text1, text2, label = cache[1], cache[2], cache[-1]
                D.append((text1, text2, float(label)))
        return D, total_labels

train_dataset, dev_dataset, test_dataset = MyDataset('train-unsup.txt'), TestDataset('dev.txt'), TestDataset('test.txt')


def train_collate_fn(batch):
    texts = [t for t in batch]
    encodings = tokenizer(texts, max_length=args.max_len, padding='batch_longest', truncation=True, return_tensors='pt')
    return {'input_ids': encodings['input_ids']}

def eval_collate_fn(batch):
    text_a, text_b, labels = [_[0] for _ in batch], [_[1] for _ in batch], [_[2] for _ in batch]
    encodings_a = tokenizer(text_a, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    encodings_b = tokenizer(text_b, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    return {'text_a_input_ids': encodings_a["input_ids"],'text_b_input_ids': encodings_b["input_ids"],'label':torch.tensor(labels)}


class SimCSEModel(torch.nn.Module):
    def __init__(self):
        super(SimCSEModel, self).__init__()
        self.bert = build_bert_model(args.model_path)

    def forward(self, input_ids=None, **inputs):
        output = self.bert(input_ids=input_ids, output_hidden_states=True)
        output2 = self.bert(input_ids=input_ids, output_hidden_states=True)
        pooled_output = Bertpooling(output, pool_type='first-last-avg')
        pooled_output2 = Bertpooling(output2, pool_type='first-last-avg')
        return {'pred': (pooled_output, pooled_output2)}

    def predict(self, text_a_input_ids, text_b_input_ids):
        out_a = self.bert(input_ids=text_a_input_ids, output_hidden_states=True)
        vec_a = Bertpooling(out_a, pool_type='first-last-avg')
        out_b = self.bert(input_ids=text_b_input_ids, output_hidden_states=True)
        vec_b = Bertpooling(out_b, pool_type='first-last-avg')
        return {'pred': (vec_a, vec_b)}


model = SimCSEModel()
trainer = Trainer(model, args, train_dataset, loss='infoNCE', eval_loss='none',
                  dev_data=dev_dataset, test_data=test_dataset, validate_every=50,
                  train_collate_fn=train_collate_fn, eval_collate_fn=eval_collate_fn, metrics='spearman')


if __name__ == "__main__":
    trainer.train()
    # trainer.test()
    # best score is 0.700035
    # cls best score is 0.650955
    # fl avg spearman correlation: 0.656479

