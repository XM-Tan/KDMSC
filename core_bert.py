import torch
from torch import nn

from transformers import BertTokenizer, BertModel, BertConfig, BertLayer

class CoreBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)


        self.config.num_hidden_layers = 6
        self.config.num_attention_heads = 6

        self.encoder.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

config = BertConfig.from_pretrained('bert-base-uncased')
model = CoreBertModel(config)

