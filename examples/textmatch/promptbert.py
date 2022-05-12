#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from fantasybert.models.model_building import build_bert_model
from fantasybert.core.trainer import Trainer
from fantasybert.core.data_utils import set_seed, TrainerState, DataSetGetter
from fantasybert.tokenization.tokenizers import BertTokenizer

args = TrainerState(
    learning_rate=2.5e-5,
    n_epochs=10,
    max_len=100,
    batch_size=32,
    metric_key='spearman_correlation',
    #use_block_shuffle=True,
    model_path="S:/bert4torch/resources/chinese-roberta-wwm-ext",
    data_dir="S:/bert4torch/datasets/STS-B/",
    save_path="S:/bert4torch/experiments/outputs/promptbert_stsb",
)

args.replace_token = '[X]'
args.mask_token = "[MASK]"
args.prompt_templates = ['"{}"，它的意思为[MASK]'.format(args.replace_token), '"{}"。这句话的意思是[MASK]'.format(args.replace_token)]
args.tao = 0.05

set_seed(args)
tokenizer = BertTokenizer(args.model_path + "/vocab.txt")

special_token_dict = {'additional_special_tokens': ['[X]']}
tokenizer.add_special_tokens(special_token_dict)
mask_id = tokenizer.convert_tokens_to_ids(args.mask_token)

class MyDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], None
        with open(args.data_dir + filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sent = line.strip()
                words_num = len(tokenizer.tokenize(sent))
                sentence_pair = []
                for template in args.prompt_templates:
                    if words_num > args.max_len - 15:
                        sent = sent[:args.max_len - 15]
                    sent_num = len(tokenizer.tokenize(sent))
                    prompt_sent = template.replace(args.replace_token, sent)
                    template_sent = template.replace(args.replace_token, (args.replace_token + ' ') * sent_num)
                    sentence_pair.append([prompt_sent, template_sent])
                D.append(sentence_pair)
        return D, total_labels

class TestDataset(DataSetGetter):
    @staticmethod
    def load_data(filename):
        D, total_labels = [], None
        with open(args.data_dir + filename, "r", encoding="utf-8-sig") as f:
            for line in f.readlines():
                cache = line.split('||')
                text1, text2, label = cache[1], cache[2], cache[-1]
                prompt_text_a = args.prompt_templates[0].replace("[X]", text1)
                prompt_text_b = args.prompt_templates[0].replace("[X]", text2)
                D.append((prompt_text_a, prompt_text_b, float(label)))
        return D, total_labels

train_dataset, dev_dataset, test_dataset = MyDataset('train-unsup.txt'), TestDataset('dev.txt'), TestDataset('test.txt')


def train_collate_fn(batch):
    prompt_lines0 = [_[0][0] for _ in batch]
    template_lines0 = [_[0][1] for _ in batch]
    prompt_lines1 = [_[1][0] for _ in batch]
    template_lines1 = [_[1][1] for _ in batch]

    prompt0 = tokenizer(prompt_lines0, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    prompt1 = tokenizer(prompt_lines1, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    template0 = tokenizer(template_lines0, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    template1 = tokenizer(template_lines1, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)

    return {'prompt0_input': prompt0['input_ids'], 'prompt1_input': prompt1['input_ids'],
            'template0_input': template0['input_ids'], 'template1_input': template1['input_ids']}


def eval_collate_fn(batch):
    prompt_text_a, prompt_text_b, labels = [_[0] for _ in batch], [_[1] for _ in batch], [_[2] for _ in batch]
    encodings_a = tokenizer(prompt_text_a, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    encodings_b = tokenizer(prompt_text_b, max_length=args.max_len, padding=True, return_tensors='pt', truncation=True)
    return {'text_a_input_ids': encodings_a["input_ids"], 'text_b_input_ids': encodings_b["input_ids"],
            'label': torch.tensor(labels)}



class PromptBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = build_bert_model(args.model_path)
        self.mask_id = mask_id

    def forward(self, prompt0_input, prompt1_input, template0_input, template1_input):
        key = self.get_sentence_embedding(prompt0_input, template0_input)
        query = self.get_sentence_embedding(prompt1_input, template1_input)
        return {'pred': (key, query)}

    def get_sentence_embedding(self, prompt_input_ids, template_input_ids):
        prompt_mask_embedding = self.calculate_mask_embedding(prompt_input_ids)
        template_mask_embedding = self.calculate_mask_embedding(template_input_ids)
        sentence_embedding = prompt_mask_embedding - template_mask_embedding
        return sentence_embedding

    def calculate_mask_embedding(self, input_ids):
        output = self.bert(input_ids=input_ids)
        token_embeddings = output.last_hidden_state
        mask_index = (input_ids == self.mask_id).long()
        mask_embedding = self.get_mask_embedding(token_embeddings, mask_index)
        return mask_embedding

    def get_mask_embedding(self, token_embeddings, mask_index):
        input_mask_expanded = mask_index.unsqueeze(-1).expand(token_embeddings.size()).float()
        mask_embedding = torch.sum(token_embeddings * input_mask_expanded, 1)
        return mask_embedding

    def predict(self, text_a_input_ids, text_b_input_ids):
        vec_a = self.calculate_mask_embedding(input_ids=text_a_input_ids)
        vec_b = self.calculate_mask_embedding(input_ids=text_b_input_ids)
        return {'pred': (vec_a, vec_b)}


model = PromptBERT()
model.bert.resize_token_embeddings(len(tokenizer))
trainer = Trainer(model, args, train_dataset, loss='infoNCE', dev_data=dev_dataset, test_data=test_dataset, validate_every=30,
                  train_collate_fn=train_collate_fn, eval_collate_fn=eval_collate_fn, metrics='spearman')

if __name__ == "__main__":
    trainer.train()
    # trainer.test()
    # best score is 0.745121

