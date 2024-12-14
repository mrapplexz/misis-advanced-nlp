import json
from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from tokenizers import Tokenizer


class BertInfer:
    def __init__(self, save_dir: Path):
        options = SessionOptions()
        options.inter_op_num_threads = 4
        options.intra_op_num_threads = 4
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = InferenceSession(
            save_dir / 'model.onnx',
            options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._tokenizer = Tokenizer.from_file(str(save_dir / 'tokenizer' / 'tokenizer.json'))
        class_map = json.loads((save_dir / 'class_map.json').read_text(encoding='utf-8'))
        class_map = {i: x for x, i in class_map.items()}
        self._class_map = class_map


    def predict(self, text: str) -> dict[str, float]:
        encoding = self._tokenizer.encode(text)
        result = self._session.run(['probabilities'], {
            'input_ids': np.asarray(encoding.ids)[None, :],
            'attention_mask': np.asarray(encoding.attention_mask)[None, :],
            'token_type_ids': np.asarray(encoding.type_ids)[None, :]
        })
        result = result[0][0]
        return {k: result[i].item() for i, k in self._class_map.items()}


if __name__ == '__main__':
    infer = BertInfer(Path('/home/me/projects/misis/bert-impl/saves/multilabel'))
    result = infer.predict('Hello! Some non-toxic text')
    print(123)