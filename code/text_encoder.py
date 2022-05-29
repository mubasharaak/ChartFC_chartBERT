from abc import abstractmethod

import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertTokenizer, BertModel


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, txt, txt_encode, txt_len):
        pass


class SimpleTextEncoder(TextEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.text_dim = 768
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.word_embeddings = nn.Embedding(len(self.tokenizer), config.text_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.simple_encoder_max_position_embeddings, config.text_dim)
        self.token_type_embeddings = nn.Embedding(len(self.tokenizer), config.text_dim)

        self.LayerNorm = FusedLayerNorm(config.text_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.3)

    def forward(self, txt, txt_encode, txt_len):
        embeddings = self.tokenizer.batch_encode_plus(list(txt), padding='longest', return_tensors='pt',
                                                      return_attention_mask=True)
        position_ids = torch.arange(0, embeddings["input_ids"].size(1), dtype=torch.long).unsqueeze(0).repeat(
            embeddings["input_ids"].size(0), 1).to("cuda")
        input_ids = embeddings["input_ids"].to("cuda")

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # attention_mask = embeddings["attention_mask"]
        token_type_embeddings = self.token_type_embeddings(embeddings["token_type_ids"].to("cuda"))

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LstmEncoder(TextEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.text_dim = 768
        self.config = config
        self.embedding = nn.Embedding(config.txt_token_count, config.lstm_embedding_dim)
        self.lstm = nn.LSTM(input_size=config.lstm_embedding_dim, hidden_size=config.text_dim, num_layers=2)
        self.drop = nn.Dropout(0.3)

    def forward(self, txt, txt_encode, txt_len):
        txt_encode = txt_encode.cuda()
        embedding = self.embedding(txt_encode)
        embedding = torch.tanh(embedding)

        packed = pack_padded_sequence(embedding, txt_len, batch_first=True, enforce_sorted=False)
        o, (h, c) = self.lstm(packed)

        txt_feat = c.permute(1, 0, 2)
        txt_feat = self.drop(txt_feat)
        return txt_feat


class BertEncoder(TextEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.text_dim = 768
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.bert_encoder = BertModel.from_pretrained(config.pretrained_model, output_hidden_states=True)

    def forward(self, txt, txt_encode, txt_len):
        embeddings = self.tokenizer.batch_encode_plus(list(txt), padding='longest', return_tensors='pt',
                                                      return_attention_mask=True)
        embeddings = embeddings.to('cuda')
        out = self.bert_encoder(**embeddings)
        txt_feat = out[0]
        return txt_feat
