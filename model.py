import math

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import Vocab

from dataset import TranslationDataset
from masks import generate_square_subsequent_mask


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(maxlen, emb_size)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, token_embedding: Tensor) -> Tensor:
        pe = self.pe[:token_embedding.size(1), :].transpose(0, 1)
        pe = pe.repeat(token_embedding.size(0), 1, 1)
        token_embedding = token_embedding + pe
        return self.dropout(token_embedding)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class EncoderLayer(nn.Module):
    def __init__(self, emb_size, nhead, dim_feedforward, dropout) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(emb_size, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_size),
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None) -> Tensor:
        src2 = self.norm1(src)
        src = src + self.dropout(self.self_attn(src2, src2, src2, attn_mask=src_mask)[0])
        src2 = self.norm2(src)
        src = src + self.dropout(self.feed_forward(src2))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, emb_size, nhead, dim_feedforward, dropout) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(emb_size, nhead, dropout=dropout, batch_first=True)
        self.enc_attn = nn.MultiheadAttention(emb_size, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_size),
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None) -> Tensor:
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0])
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout(self.enc_attn(tgt2, memory, memory, key_padding_mask=memory_mask)[0])
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(tgt2))
        return tgt


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, num_layers, emb_size, nhead, dim_feedforward, dropout, max_length) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(src_vocab_size, emb_size)
        self.pos_encoding = PositionalEncoding(emb_size, dropout, max_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(emb_size, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src, src_mask=None) -> Tensor:
        src_emb = self.embedding(src)
        src_emb = self.pos_encoding(src_emb)
        for layer in self.layers:
            src_emb = layer(src_emb, src_mask)
        return src_emb


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, num_layers, emb_size, nhead, dim_feedforward, dropout, max_length) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(tgt_vocab_size, emb_size)
        self.pos_encoding = PositionalEncoding(emb_size, dropout, max_length)
        self.layers = nn.ModuleList(
            [DecoderLayer(emb_size, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None) -> Tensor:
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        for layer in self.layers:
            tgt_emb = layer(tgt_emb, memory, tgt_mask, memory_mask)
        return tgt_emb


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_length: int) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab_size, num_encoder_layers, emb_size, nhead, dim_feedforward, dropout,
                               max_length)
        self.decoder = Decoder(tgt_vocab_size, num_decoder_layers, emb_size, nhead, dim_feedforward, dropout,
                               max_length)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor) -> Tensor:
        memory = self.encoder(src, src_mask)
        output = self.decoder(trg, memory, tgt_mask, src_padding_mask)
        logits = self.generator(output)
        return logits

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.encoder(src, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.decoder(tgt, memory, tgt_mask)

    def greedy_decode(self,
                      src: Tensor,
                      src_mask: Tensor,
                      max_len: int,
                      start_symbol: int,
                      device: torch.device) -> Tensor:
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = self.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol).type(torch.long).to(device)
        for _ in range(max_len - 1):
            memory = memory.to(device)
            tgt_mask = generate_square_subsequent_mask(ys.size(1), device).to(device)
            out = self.decode(ys, memory, tgt_mask)
            prob = self.generator(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == TranslationDataset.EOS_ID:
                break
        return ys

    def translate(self, src: torch, trg_vocab: Vocab, device: torch.device) -> str:
        self.eval()
        num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, num_tokens + 5, TranslationDataset.BOS_ID, device).flatten()
        return " ".join(
            trg_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
