#! -*- coding: utf-8 -*-
import torch
import json
import torch.nn as nn
from fantasybert.models.model_building import build_bert_model
from fantasybert.core.trainer import Trainer
from fantasybert.core.data_utils import set_seed, TrainerState, DataSetGetter
from fantasybert.tokenization.tokenizers import BertTokenizer

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=20,
    add_polynominal=True,
    max_len=68,
    metric_key='f1',
    batch_size=256,
    save_path="S:/Myproject/experiments/outputs/text_match_afqmc",
    model_path="S:/Myproject/resources/NEZHA",
    data_dir="S:/Myproject/datasets/afqmc/"
)

set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")


class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], []
        with open(args.data_dir + filename, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                text_a = line['sentence1']
                text_b = line['sentence2']
                label = line['label']
                D.append((text_a, text_b, label))
                total_labels.append(label)
        return D, total_labels

train_dataset, dev_dataset, test_dataset = MyDataset('train.json'), MyDataset('dev.json'), MyDataset('dev.json')
num_labels, label2id = train_dataset.num_labels, test_dataset.label2id
print(train_dataset.labels_distribution)
args.num_labels = num_labels

def my_collate_fn(batch):
    text_a, text_b, labels = [_[0] for _ in batch], [_[1] for _ in batch], [label2id[_[2]] for _ in batch]
    encodings = tokenizer(text_a,text_pair=text_b, max_length=args.max_len, padding='batch_longest', return_tensors='pt', truncation=True)
    return {'input_ids': encodings["input_ids"], 'attention_mask': encodings['attention_mask'], 'label': torch.tensor(labels)}


class ClsModel(nn.Module):
    def __init__(self):
        super(ClsModel, self).__init__()
        self.bert, self.config = build_bert_model(args.model_path,model_name='nezha', return_config=True)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, **inputs):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.fc(output.pooler_output)
        return {'pred': logits}



model = ClsModel()
trainer = Trainer(model, args, train_dataset, dev_data=dev_dataset, test_data=test_dataset, validate_every=100,
                  train_collate_fn=my_collate_fn, eval_collate_fn=my_collate_fn, metrics=['acc', 'fpr'])

trainer.train()
# trainer.test('best_ClsModel_accuracy_2022-05-05-14-42-35-774503')
