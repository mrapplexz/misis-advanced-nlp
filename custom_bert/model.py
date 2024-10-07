import math

import torch
from torch import nn
import torch.nn.functional as F

from custom_bert.config import BertConfig
from custom_bert.preprocessing import build_square_attention_mask


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.padding_token_idx
        )
        self.position_embeddings = nn.Embedding(
            config.max_positions,
            config.hidden_size
        )
        self.segment_embeddings = nn.Embedding(
            config.segment_size,
            config.hidden_size
        )
        self.dropout = nn.Dropout(
            p=config.dropout_prob_embeddings
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            segment_ids: torch.Tensor
    ):
        position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0).repeat(input_ids.shape[0], 1)
        input_token_emb = self.token_embeddings(input_ids)
        input_position_emb = self.position_embeddings(position_ids)
        input_segment_emb = self.segment_embeddings(segment_ids)
        ret_value = input_token_emb + input_position_emb + input_segment_emb
        ret_value = self.dropout(ret_value)
        return ret_value


class BertFeedForward(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.ReLU()
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob_model)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.up(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.down(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class BertFeedForwardWithSkip(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.feed_forward = BertFeedForward(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)

    def forward(self, hidden_state: torch.Tensor):
        new_hidden_state = self.feed_forward(hidden_state)
        return self.layer_norm(hidden_state + new_hidden_state)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        assert config.hidden_size % config.num_attention_heads == 0
        self.head_size = config.hidden_size // config.num_attention_heads
        self.denominator = math.sqrt(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob_model)


    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # batch x seq length x head x per-head hidden state
        x = x.reshape(batch_size, seq_len, -1, self.head_size)
        # batch x head x seq length x per-head hidden state
        x = x.transpose(2, 1)
        # batch*head x seq length x per-head hidden state
        x = x.reshape(-1, seq_len, self.head_size)
        return x

    def unsplit_heads(
            self,
            x: torch.Tensor,
            batch_size: int,
            seq_length: int
    ) -> torch.Tensor:
        # input: batch*head x seq length x per-head hidden state
        # after: batch x head x seq length x per-head hidden state
        x = x.reshape(batch_size, -1, seq_length, x.shape[2])
        # after: batch x seq length x head x per-head hidden state
        x = x.transpose(1, 2)
        # after batch x seq length x hidden state
        x = x.reshape(batch_size, seq_length, -1)
        return x


    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = hidden_state.shape[0]
        seq_len = hidden_state.shape[1]

        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)

        # batch x sequence length x num heads x per-head hidden dim
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        qk = torch.bmm(q, k.transpose(-1, -2))
        qk = qk / self.denominator
        qk = qk + attention_mask
        attention_scores = F.softmax(qk, -1)
        vs = torch.bmm(attention_scores, v)

        vs_unsplit = self.unsplit_heads(vs, batch_size, seq_len)
        outputs = self.o_proj(vs_unsplit)
        outputs = self.dropout(outputs)
        return outputs


class BertAttentionWithSkip(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        new_hidden_state = self.attention(hidden_state, attention_mask)
        return self.layer_norm(hidden_state + new_hidden_state)


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.feed_forward = BertFeedForwardWithSkip(config)
        self.attention = BertAttentionWithSkip(config)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        hidden_state = self.attention(hidden_state, attention_mask)
        hidden_state = self.feed_forward(hidden_state)
        return hidden_state


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embedding = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.n_layers)])
        self.num_heads = config.num_attention_heads

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, attention_mask: torch.Tensor):
        input_embeds = self.embedding(input_ids, segment_ids)
        attention_mask = build_square_attention_mask(attention_mask, input_embeds.dtype, input_embeds.device,
                                                     num_heads=self.num_heads)
        hidden_state = input_embeds
        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)
        return hidden_state
