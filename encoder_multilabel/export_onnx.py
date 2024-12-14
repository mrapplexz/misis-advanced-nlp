import json
from pathlib import Path

import click
import safetensors.torch
import torch
from tokenizers.tokenizers import Tokenizer
from transformers import AutoTokenizer

from encoder_multilabel.const import BASE_MODEL_NAME, LABELS
from encoder_multilabel.model import MultiLabelModel, MultiLabelWrap


_TEST_SEQUENCE = 'Hello, world! This a regular comment'


@click.command()
@click.option('--weights-path', type=Path, required=True)
@click.option('--save-dir', type=Path, required=True)
def main(weights_path: Path, save_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(save_dir / 'tokenizer')

    with (save_dir / 'class_map.json').open('w') as f:
        json.dump({class_name: class_idx for class_idx, class_name in enumerate(LABELS)}, f)

    test_data = tokenizer.encode_plus(_TEST_SEQUENCE)

    model = MultiLabelModel()
    model_weights = safetensors.torch.load_file(str(weights_path))
    model.load_state_dict(model_weights)
    model = MultiLabelWrap(model).eval()

    input_ids = torch.tensor(test_data.encodings[0].ids)[None, :]
    attention_mask = torch.tensor(test_data.encodings[0].attention_mask)[None, :]
    token_type_ids = torch.tensor(test_data.encodings[0].type_ids)[None, :]

    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(save_dir / 'model.onnx'),
        verbose=False,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['probabilities'],
        opset_version=17,
        export_modules_as_functions=False,
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence_len'},
            'attention_mask': {0: 'batch', 1: 'sequence_len'},
            'token_type_ids': {0: 'batch', 1: 'sequence_len'},
            'probabilities': {0: 'batch'}
        },
        dynamo=False
    )


if __name__ == '__main__':
    main()
