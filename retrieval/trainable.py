import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss
from torch import nn

from retrieval.dataset import load_data, RetrievalDataset, RetrievalCollator
from retrieval.model import RetrievalModel


class RetrievalTrainable:
    def __init__(self, model: RetrievalModel):
        self._model = model
        self._loss = ContrastiveLoss(pos_margin=0.1, neg_margin=0.9, distance=CosineSimilarity())

    def compute_loss(self, data: dict):
        query_embeds = self._model(data['query']['input_ids'], data['query']['attention_mask'], is_query=True)
        document_embeds = self._model(data['document']['input_ids'], data['document']['attention_mask'], is_query=False)
        all_embeds = torch.cat((query_embeds, document_embeds), dim=0)
        partial_labels = torch.arange(0, len(query_embeds), device=query_embeds.device, dtype=torch.long)
        all_labels = torch.cat((partial_labels, partial_labels), dim=0)
        loss_value = self._loss(all_embeds, all_labels)
        return loss_value


if __name__ == '__main__':
    data = load_data()
    dataset = RetrievalDataset(data)
    collator = RetrievalCollator()
    model = RetrievalModel()
    batch = collator([dataset[0], dataset[1]])
    trainable = RetrievalTrainable(model)
    loss = trainable.compute_loss(batch)
    print(loss)