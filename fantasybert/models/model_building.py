import os
from transformers import BertModel, BertConfig
from fantasybert.models.modeling_nezha import NeZhaModel

def build_bert_model(
        model_path,
        model_name=None,
        return_config=False,
):
    if model_name == 'nezha':
        model = NeZhaModel.from_pretrained(model_path)
    else:
        model = BertModel.from_pretrained(model_path)

    config = None
    if return_config:
        config = BertConfig.from_pretrained(model_path)

    if not return_config:
        return model
    else:
        return model, config








