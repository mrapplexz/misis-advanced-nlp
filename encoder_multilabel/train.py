import json
from pathlib import Path

import click
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchmetrics import F1Score, AUROC
from transformers import AdamW

from encoder_multilabel.config import TrainConfig
from encoder_multilabel.const import LABELS
from encoder_multilabel.dataset import load_multilabel, MultiLabelDataset, MultiLabelCollator
from encoder_multilabel.model import MultiLabelModel


@click.command()
@click.option('--config-path', type=Path, required=True)
def train(config_path: Path):
    config = TrainConfig.model_validate_json(config_path.read_text(encoding='utf-8'))

    set_seed(config.random_state)

    data = load_multilabel(config.data_path)
    data = data.train_test_split(
        test_size=config.eval_split_fraction, shuffle=True,
        seed=config.random_state
    )
    dataset_train = MultiLabelDataset(data['train'])
    dataset_test = MultiLabelDataset(data['test'])
    collator = MultiLabelCollator()

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir='.',
        log_with="aim"
    )
    accelerator.init_trackers(config.experiment_name, config=json.loads(config.model_dump_json()))

    model = MultiLabelModel()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_dataloader = DataLoader(
        dataset_train, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        dataset_test, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              eval_dataloader)
    total_steps = config.epochs * len(train_dataloader)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_proportion),
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)
    loss = BCEWithLogitsLoss()

    with tqdm(desc='Training', total=total_steps) as pbar:
        for epoch in range(config.epochs):
            for batch_i, batch in enumerate(train_dataloader):
                current_step = batch_i + epoch * len(train_dataloader)
                with accelerator.accumulate(model):
                    logits = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids']
                    )
                    loss_value = loss(logits, batch['labels'])
                    accelerator.log({"train_loss": loss_value.item()}, step=current_step + 1)
                    accelerator.backward(loss_value)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if (current_step + 1) % config.eval_steps == 0 or current_step + 1 == total_steps:
                    model.eval()
                    f1 = F1Score(task='multilabel', num_labels=len(LABELS), average='macro').to(accelerator.device)
                    auroc = AUROC(task='multilabel', num_labels=len(LABELS), average='macro').to(accelerator.device)
                    with tqdm(desc='Eval', total=len(eval_dataloader)) as eval_pbar:
                        with torch.no_grad():
                            for eval_batch in eval_dataloader:
                                logits_eval = model(
                                    input_ids=eval_batch['input_ids'],
                                    attention_mask=eval_batch['attention_mask'],
                                    token_type_ids=eval_batch['token_type_ids']
                                )
                                preds_proba_eval = torch.sigmoid(logits_eval)
                                eval_predictions_total, eval_target_total = accelerator.gather_for_metrics(
                                    (preds_proba_eval, eval_batch['labels'])
                                )
                                f1.update(eval_predictions_total, eval_target_total.to(torch.long))
                                auroc.update(eval_predictions_total, eval_target_total.to(torch.long))
                                eval_pbar.update(1)
                    accelerator.log({'eval_f1_macro': f1.compute().item(), 'eval_auroc_macro': auroc.compute().item()}, step=current_step + 1)
                    f1.reset()
                    auroc.reset()
                    model.train()

                if (current_step + 1) % config.save_steps == 0 or current_step + 1 == total_steps:
                    accelerator.wait_for_everyone()
                    accelerator.save_model(model, config.save_path / f'step_{current_step + 1}')

    accelerator.end_training()


if __name__ == '__main__':
    train()
