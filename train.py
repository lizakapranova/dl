import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from criterion import LabelSmoothingLoss
from dataset import TranslationDataset
from masks import create_mask
from model import Seq2SeqTransformer


def plot_losses(train_losses: list[float], val_losses: list[float]) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 15})

    clear_output()
    plt.plot(list(range(1, len(train_losses) + 1)), train_losses, '-o', label='train')
    plt.plot(list(range(1, len(val_losses) + 1)), val_losses, '-o', label='val')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def train_epoch(model: Seq2SeqTransformer,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Adam,
                tqdm_desc: str,
                device: torch.device) -> float:
    model.train()
    losses = 0

    for src, tgt in tqdm(train_loader, desc=tqdm_desc):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask = create_mask(src, tgt_input, TranslationDataset.PAD_ID, device)
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask)

        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item() * src.shape[0]

    return losses / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model: Seq2SeqTransformer,
             val_loader: DataLoader,
             criterion: nn.Module,
             tqdm_desc: str,
             device: torch.device) -> float:
    model.eval()
    losses = 0

    for src, tgt in tqdm(val_loader, desc=tqdm_desc):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask = create_mask(src, tgt_input, TranslationDataset.PAD_ID,
                                                           device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask)

        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item() * src.shape[0]

    return losses / len(val_loader.dataset)


def train(model: Seq2SeqTransformer,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: optim.Adam,
          num_epochs: int, device: torch.device,
          build_graph: bool = False) -> None:
    train_losses, val_losses = [], []
    criterion = LabelSmoothingLoss(ignore_index=TranslationDataset.PAD_ID)
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, f"Training {epoch}/{num_epochs}", device)
        val_loss = evaluate(model, val_loader, criterion, f"Validating {epoch}/{num_epochs}", device)

        train_losses += [train_loss]
        val_losses += [val_loss]
        if build_graph:
            plot_losses(train_losses, val_losses)
