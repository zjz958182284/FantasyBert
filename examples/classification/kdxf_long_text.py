#! -*- coding: utf-8 -*-
import torch
import pandas as pd
import torch.nn as nn
from fantasybert.models.model_building import build_bert_model
from fantasybert.core.trainer import Trainer
from fantasybert.core.utils import set_seed, TrainerState
from fantasybert.core.data_utils import DataSetGetter
from fantasybert.tokenization.tokenizers import BertTokenizer

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=10,
    max_len=510,
    batch_size=16,
    metric_key='f1',
    use_block_shuffle=True,
    save_path="S:/bert4torch/experiments/outputs/text_classify_kdxf",
    model_path="S:/bert4torch/resources/NEZHA",
    data_dir="S:/bert4torch/datasets/longtext_kdxf/"
)

set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")
args.tokenizer = tokenizer


class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], []
        data = pd.read_csv(args.data_dir + filename)
        for i in range(len(data)):
            sent = data.loc[i, "sentence"]
            label = data.loc[i, "label"]
            D.append((sent, label))
            total_labels.append(label)
        return D, total_labels

train_dataset, dev_dataset, test_dataset = MyDataset('train.csv'), MyDataset('dev.csv'), MyDataset('test.csv')
num_labels, label2id = train_dataset.num_labels, train_dataset.label2id
print(train_dataset.labels_distribution)


def my_collate_fn(batch):
    texts, labels = [_[0] for _ in batch], [label2id[_[1]] for _ in batch]
    encodings = tokenizer(texts, max_length=args.max_len, padding='batch_longest', return_tensors='pt', truncation=True)
    return {'input_ids': encodings["input_ids"], 'attention_mask': encodings['attention_mask'], 'label': torch.tensor(labels),
            'text':texts}


class ClsModel(nn.Module):
    def __init__(self):
        super(ClsModel, self).__init__()
        self.bert, self.config = build_bert_model(args.model_path, model_name='nezha', return_config=True)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **inputs):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        logits = self.fc(pooled_output)
        return {'pred': logits}


model = ClsModel()
trainer = Trainer(model, args, train_dataset, dev_data=dev_dataset, test_data=test_dataset, validate_every=100,
                  train_collate_fn=my_collate_fn, eval_collate_fn=my_collate_fn, metrics=['acc', 'fpr'])

print(trainer.callback_manager.callback_list)
trainer.train()

# trainer.test()
# prediction_output_file = os.path.join(args.save_path, "test_predictions.csv")
# trainer.predict(prediction_output_file)
# dev  f1 0.943747  acc 0.943335
# test f1  acc # test f1