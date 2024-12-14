from pathlib import Path

from pydantic import BaseModel


class TrainConfig(BaseModel):
    data_path: Path
    random_state: int
    eval_split_fraction: float
    gradient_accumulation_steps: int
    lr: float
    weight_decay: float
    warmup_proportion: float
    batch_size: int
    num_workers: int
    epochs: int
    experiment_name: str
    save_steps: int
    save_path: Path
    eval_steps: int
