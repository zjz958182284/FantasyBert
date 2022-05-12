 #! -*- coding: utf-8 -*-
import sys
import torch
import json
import torch.nn as nn
import torch.utils.checkpoint
sys.path.insert(0, './')
from fantasybert.models.crf import CRF
from fantasybert.models.model_building import build_bert_model
from fantasybert.core.trainer import Trainer
from fantasybert.core.metrics import SpanFPreRecMetric
from fantasybert.core.data_utils import DataSetGetter, set_seed, TrainerState
from fantasybert.tokenization.tokenizers import BertTokenizer

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=50,
    max_len=64,
    batch_size=256,
    metric_key='loss',
    use_block_shuffle=True,
    save_path="S:/bert4torch/experiments/outputs/ner_ccks_2017",
    model_path="S:/bert4torch/resources/NEZHA",
    data_dir="S:/bert4torch/datasets/ccks_2017/"
)

set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")


class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], []
        lines = json.load(open(args.data_dir +filename, "r", encoding="utf-8"))
        for (i, line) in enumerate(lines):
            words_sequence = line["sentence"]
            # list of spans: [[12, 13, "Symptom"], ....]
            #   结束位置index是内含的
            labels_by_span = line["labeled entities"]
            labels_in_sequence = ["O"] * len(words_sequence)

            for span in labels_by_span:
                labels_in_sequence[int(span[0])] = f"B-{span[-1]}"
                labels_in_sequence[int(span[0]) + 1: int(span[1]) + 1] = [f"I-{span[-1]}", ] * (
                        int(span[1]) - int(span[0]))

            assert len(labels_in_sequence) == len(words_sequence)
            D.append((words_sequence, labels_in_sequence))

            for label in labels_in_sequence:
                total_labels.append(label)
        return D, total_labels


train_dataset, dev_dataset, test_dataset = MyDataset('train.json'), MyDataset('dev.json'), MyDataset('test.json')
num_labels, label2id = train_dataset.num_labels, train_dataset.label2id
id2label = {v: k for k, v in label2id.items()}
print(train_dataset.labels_distribution)   #  add_label="PAD" ???


def my_collate_fn(batch):
    batch_input, batch_att_mask, batch_labels = [], [], []
    for words, labels in batch:
        # 对于NER数据，如果一个词切分为多个subtokens，非起始subtokens采用pad token的ner label
        #   e.g., A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, PAD, PAD;
        label_ids = []
        subtokens = []
        for word, lab in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)  # 分词  将词输入到bert的tokenizer中去将它转化为bert词表中的tokens  ['i']
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            subtokens.extend(word_tokens)
            label_ids.extend([label2id[lab]] + [0] * (len(word_tokens) - 1))

        # 处理超过长度的样本
        special_tokens_count = 2
        if len(subtokens) > args.max_len - special_tokens_count:
            subtokens = subtokens[:(args.max_len - special_tokens_count)]
            label_ids = label_ids[:(args.max_len - special_tokens_count)]

        subtokens += [tokenizer.sep_token]
        label_ids += [0]  # [SEP] label: pad_token_label_id
        subtokens = [tokenizer.cls_token] + subtokens
        label_ids = [0] + label_ids  # [CLS] label: pad_token_label_id

        # subtokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(subtokens)
        attention_mask = [1] * len(input_ids)
        batch_input.append(input_ids + (args.max_len - len(input_ids)) * [0])
        batch_att_mask.append(attention_mask + (args.max_len - len(attention_mask)) * [0])
        batch_labels.append(label_ids + (args.max_len - len(label_ids)) * [0])

    return {'input_ids': torch.tensor(batch_input), 'attention_mask':torch.tensor(batch_att_mask), 'label': torch.tensor(batch_labels)}



class NerModel(nn.Module):
    def __init__(self):
        super(NerModel, self).__init__()
        self.bert, self.config = build_bert_model(args.model_path, model_name='nezha', return_config=True)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        self.crf_layer = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, **inputs):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = output.last_hidden_state

        logits = self.fc(seq_output)  # 取最后一个输出层的所有向量经过CRF
        label = inputs['label']
        loss = self.crf_layer(emissions=logits, tags=label,
                              mask=attention_mask.byte(), reduction='mean')  # reduction='mean' 为对loss进行归一化，否则值很大
        loss = -1 * loss  # negative log-likelihood
        logits = torch.tensor(self.crf_layer.decode(logits)).to('cuda')
        return {'pred': logits, 'loss': loss}


model = NerModel()
trainer = Trainer(model, args, train_dataset, dev_data=dev_dataset, test_data=test_dataset, validate_every=50,
                  train_collate_fn=my_collate_fn, eval_collate_fn=my_collate_fn,
                  metrics=SpanFPreRecMetric(tag_vocab=id2label), loss='loss_in_forward',)


trainer.train()
# trainer.test()

# 2022/05/05 22:48:12 -
# In Epoch:30/Step:840, got best dev performance:
# 2022/05/05 22:48:12 - SpanFPreRecMetric: f1=0.890114, precision=0.987636, recall=0.810121
#   File "S:\bert4torch\examples\NER\bert_crf.py", line 124, in <module>
#     trainer.train()
#   File "S:\bert4torch\easytransformers\core\trainer.py", line 214, in train
#     load_succeed = self._load_model(self.model, model_name)
#   File "S:\bert4torch\easytransformers\core\trainer.py", line 445, in _load_model
#     states = torch.load(model_path).state_dict()
# AttributeError: 'collections.OrderedDict' object has no attribute 'state_dict'
#
# Process finished with exit code 1


