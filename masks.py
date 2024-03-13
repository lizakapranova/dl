import torch
from torch import Tensor


def generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: Tensor, tgt: Tensor, pad_id: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_id)

    return src_mask, tgt_mask, src_padding_mask
