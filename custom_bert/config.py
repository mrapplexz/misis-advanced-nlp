from transformers import PretrainedConfig


class BertConfig(PretrainedConfig):
    model_type = 'custom_bert'

    def __init__(
            self,
            vocab_size: int,
            segment_size: int,
            max_positions: int,
            hidden_size: int,
            intermediate_size: int,
            padding_token_idx: int,
            dropout_prob_embeddings: float,
            num_attention_heads: int
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.segment_size = segment_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.padding_token_idx = padding_token_idx
        self.dropout_prob_embeddings = dropout_prob_embeddings
        self.num_attention_heads = num_attention_heads
