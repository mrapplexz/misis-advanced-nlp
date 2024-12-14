import datasets
import torch.nn.functional as F
import torch.utils.data
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from retrieval.const import BASE_MODEL_NAME


def pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(x) for x in tensors)
    return torch.stack([F.pad(x, pad=(0, max_len - len(x)), mode='constant', value=0) for x in tensors])


def load_data() -> datasets.Dataset:
    dataset = load_dataset("sentence-transformers/trivia-qa-triplet", "triplet")
    dataset = dataset['train']
    return dataset


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset):
        self._dataset = dataset
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def __getitem__(self, index) -> dict[str, dict[str, torch.Tensor]]:
        item = self._dataset[index]
        anchor_enc = self._tokenizer.encode_plus(item['anchor'], max_length=512).encodings[0]
        document_enc = self._tokenizer.encode_plus(item['positive'], max_length=512).encodings[0]
        return {
            'query': {
                'input_ids': torch.tensor(anchor_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(anchor_enc.attention_mask, dtype=torch.long),
            },
            'document': {
                'input_ids': torch.tensor(document_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(document_enc.attention_mask, dtype=torch.long),
            }
        }


    def __len__(self):
        return len(self._dataset)



class RetrievalCollator:
    def __call__(self, items: list[dict[str, dict[str, torch.Tensor]]]):
        return {
            'query': {
                'input_ids': pad_tensors([x['query']['input_ids'] for x in items]),
                'attention_mask': pad_tensors([x['query']['attention_mask'] for x in items]),
            },
            'document': {
                'input_ids': pad_tensors([x['document']['input_ids'] for x in items]),
                'attention_mask': pad_tensors([x['document']['attention_mask'] for x in items]),
            }
        }


def create_dataloader(dataset: RetrievalDataset, collator: RetrievalCollator, device: str):
    return DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4,
        collate_fn=collator, pin_memory=True, persistent_workers=False,
        pin_memory_device=device
    )
