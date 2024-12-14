import torch
import transformers
from aim import Run
from datasets import Dataset
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.dataset import load_data, RetrievalDataset, RetrievalCollator, create_dataloader
from retrieval.model import RetrievalModel
from retrieval.trainable import RetrievalTrainable


def map_device(item, device):
    match item:
        case dict():
            return {k: map_device(v, device) for k, v in item.items()}
        case torch.Tensor():
            return item.to(device)
        case _:
            raise NotImplementedError()


def train(train_dataset: Dataset, eval_dataset: Dataset, device: str):
    target_device = torch.device(device)

    run = Run(experiment='test-retrieval')

    collator = RetrievalCollator()

    train_dataset, eval_dataset = RetrievalDataset(train_dataset), RetrievalDataset(eval_dataset)
    train_dataloader = create_dataloader(train_dataset, collator, device)
    eval_dataloader = create_dataloader(eval_dataset, collator, device)

    model = RetrievalModel().to(target_device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_dataloader) * 0.1),
        num_training_steps=len(train_dataloader)
    )

    trainable = RetrievalTrainable(model)

    model.freeze_base_model()

    for i, batch in enumerate(tqdm(train_dataloader, desc='Train Loop')):
        batch = map_device(batch, target_device)
        loss = trainable.compute_loss(batch)
        run.track(loss.item(), 'loss', i)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i == 100:
            model.unfreeze_base_model()

        if (i % 100 == 0) and i != 0:
            with torch.no_grad():
                model.eval()
                losses = []
                for eval_batch in tqdm(eval_dataloader, desc='Evaluation'):
                    eval_batch = map_device(eval_batch, target_device)
                    loss_eval = trainable.compute_loss(eval_batch)
                    losses.append(loss_eval.item())
                run.track(torch.tensor(losses).mean().item(), 'eval loss', i)

                save_file(model.state_dict(), f"models/retrieval-{i}.safetensors")
                model.train()

    run.close()



if __name__ == '__main__':
    data = load_data()
    data = data.train_test_split(shuffle=True, test_size=0.02)
    result_model = train(data['train'], data['test'], 'cuda:0')
