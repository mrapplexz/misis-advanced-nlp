import torch

from custom_bert.config import BertConfig
from custom_bert.model import BertEmbeddings, BertAttention

if __name__ == '__main__':
    config = BertConfig(
        vocab_size=1000,
        segment_size=2,
        max_positions=512,
        hidden_size=128,
        padding_token_idx=0,
        dropout_prob_embeddings=0.1,
        intermediate_size=128 * 4,
        num_attention_heads=4
    )
    # input_ids = torch.tensor(
    #     [[1, 5, 10, 30, 2], [1, 200, 215, 2, 0]]
    # )
    # attention_mask = torch.tensor(
    #     [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    # )
    # segment_ids = torch.tensor(
    #     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    # )

    # batch x sequence length x hidden size
    input_embeds = torch.randn((8, 16, 128), dtype=torch.float32)
    attention_mask = torch.zeros((8, 4, 16, 16), dtype=torch.float32)
    l = BertAttention(config)
    print(l(input_embeds, attention_mask))
    print(l)
