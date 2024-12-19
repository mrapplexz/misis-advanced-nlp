from pathlib import Path

import torch.nn.functional as F
import torch.utils.data
from datasets import load_dataset, Features, Value, Dataset
from torch import Tensor
from transformers import AutoTokenizer

from encoder_multilabel.const import BASE_MODEL_NAME, LABELS


def load_multilabel(data_path: Path):
    ds = load_dataset(
        "csv",
        data_files=str(data_path),
        features=Features(
            {
                'comment_text': Value('string'),
                **{
                    name: Value('int8') for name in LABELS
                }
            }
        )
    )['train']
    return ds


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def __getitem__(self, index):
        item = self._dataset[index]
        comment_enc = self._tokenizer.encode_plus(
            item['comment_text'],
            max_length=512
        ).encodings[0]
        return {
            'input_ids': torch.tensor(comment_enc.ids, dtype=torch.int64),
            'attention_mask': torch.tensor(comment_enc.attention_mask, dtype=torch.int64),
            'token_type_ids': torch.tensor(comment_enc.type_ids, dtype=torch.int64),
            'labels': torch.stack([torch.scalar_tensor(item[name]) for name in LABELS])
            # **{name: torch.scalar_tensor(item[name]) for name in LABELS}
        }

    def __len__(self):
        return len(self._dataset)


def pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(x) for x in tensors)
    return torch.stack([F.pad(x, pad=(0, max_len - len(x)), mode='constant', value=0) for x in tensors])


class MultiLabelCollator:
    def __call__(self, batch: list[dict[str, Tensor]]):
        return {
            'input_ids': pad_tensors([x['input_ids'] for x in batch]),
            'attention_mask': pad_tensors([x['attention_mask'] for x in batch]),
            'token_type_ids': pad_tensors([x['token_type_ids'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
            # **{name: torch.stack([x[name] for x in batch]) for name in LABELS}
        }


if __name__ == '__main__':
    dataset = load_multilabel(Path('data/train.csv'))
    torch_dataset = MultiLabelDataset(dataset)
    item_1, item_2 = torch_dataset[0], torch_dataset[1]
    collator = MultiLabelCollator()
    batch = collator([item_1, item_2])
    print()
