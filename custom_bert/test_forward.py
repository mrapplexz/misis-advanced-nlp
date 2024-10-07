import torch
from torchinfo import summary

from custom_bert.config import BertConfig
from custom_bert.model import BertModel

if __name__ == '__main__':
    config = BertConfig(
        vocab_size=1000,
        segment_size=2,
        max_positions=512,
        hidden_size=128,
        padding_token_idx=0,
        dropout_prob_embeddings=0.1,
        intermediate_size=128 * 4,
        num_attention_heads=4,
        layernorm_eps=1e-5,
        n_layers=4,
        dropout_prob_model=0.1
    )
    input_ids = torch.tensor(
        [[1, 5, 10, 30, 2], [1, 200, 215, 2, 0]]
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    )
    segment_ids = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    )
    model = BertModel(config)
    summary(model, input_data={
        'input_ids': input_ids,
        'segment_ids': segment_ids,
        'attention_mask': attention_mask
    })
