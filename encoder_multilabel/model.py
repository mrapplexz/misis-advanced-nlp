import torch
from torch import nn, Tensor
from transformers import AutoModel

from encoder_multilabel.const import BASE_MODEL_NAME, LABELS


class MultilabelPoolerWithHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, last_hidden_state: Tensor):
        # last hidden state shape: [BATCH x SEQ_LEN x HIDDEN_SIZE]
        # Seq: [CLS], Tok1, Tok2, Tok3, [SEP]
        x = last_hidden_state[:, 0]  # [BATCH x HIDDEN_SIZE]
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)  # [BATCH x NUM_CLASSES]
        return x



class MultiLabelModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(BASE_MODEL_NAME)
        self.pooler_head = MultilabelPoolerWithHead(self.encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        encoder_outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = encoder_outs.last_hidden_state
        logits = self.pooler_head(last_hidden_state)
        return logits


class MultiLabelWrap(nn.Module):
    def __init__(self, model: MultiLabelModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        probas = torch.sigmoid(logits)
        return probas
