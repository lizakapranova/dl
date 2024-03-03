import os
import typing as tp

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm
from torch import Tensor

VOCAB_PATH = 'vocab.pth'


class TranslationDataset(Dataset):
    UNK_ID = 0
    PAD_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self,
                 src_file: str,
                 trg_file: str | None = None,
                 src_language: str = 'de',
                 trg_language: str = 'en',
                 is_test: bool = False,
                 truncate_to: int | None = None) -> None:
        super().__init__()

        self.tokenizers = {}
        self.vocabs = {}
        self.text_transformations = {}
        self.texts = {}

        self.src_lang = src_language
        self.trg_lang = trg_language
        self.is_test = is_test

        for lang in self.languages:
            self.tokenizers[lang] = get_tokenizer(None, language=lang)

        files = (src_file, trg_file) if not is_test else (src_file,)
        for file, lang in zip(files, self.languages):
            self.texts[lang] = TranslationDataset.fetch_texts_from_files(file, truncate_to)

        if not os.path.isfile(VOCAB_PATH):
            for lang in self.languages:
                self.vocabs[lang] = self.build_vocab(lang)

            torch.save(self.vocabs, VOCAB_PATH)
        else:
            self.vocabs = torch.load(VOCAB_PATH)

        for lang in self.languages:
            self.text_transformations[lang] = TranslationDataset.sequential_transforms(self.tokenizers[lang],
                                                                                       self.vocabs[lang],
                                                                                       self.tensor_transform)

    @property
    def languages(self) -> tuple[str, ...]:
        if self.is_test:
            return self.src_lang,
        return self.src_lang, self.trg_lang

    @staticmethod
    def tensor_transform(token_ids: list[int]) -> Tensor:
        return torch.cat((torch.tensor([TranslationDataset.BOS_ID]),
                          torch.tensor(token_ids),
                          torch.tensor([TranslationDataset.EOS_ID])))

    def build_vocab(self, language: str) -> Vocab:
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab = build_vocab_from_iterator([self.yield_token(language)], specials=special_symbols, min_freq=1,
                                          special_first=True)
        vocab.set_default_index(TranslationDataset.UNK_ID)
        return vocab

    @staticmethod
    def fetch_texts_from_files(filename: str, trunc_to: int | None) -> list[str]:
        with open(filename, 'r') as file:
            lines = file.readlines()

        if trunc_to is not None:
            lines = lines[:trunc_to]

        texts = [line.strip() for line in lines]
        return texts

    def yield_token(self, language: str) -> tp.Generator[str, tp.Any, None]:
        for line in tqdm(self.texts[language], desc="Building vocab"):
            sentence = self.tokenizers[language](line)
            for token in sentence:
                yield token

    @staticmethod
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    @property
    def vocab_sizes(self):
        return tuple(len(self.vocabs[lang]) for lang in self.languages)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(self.text_transformations[lang](self.texts[lang][index]) for lang in self.languages)

    def __len__(self) -> int:
        return len(self.texts[self.src_lang])

    def build_collate_fn(self) -> tp.Callable[[Tensor], tuple[Tensor, tp.Any]]:
        def collate_fn(batch: Tensor) -> tuple[Tensor, Tensor | None]:
            src_batch = pad_sequence([bat[0] for bat in batch], padding_value=TranslationDataset.PAD_ID,
                                     batch_first=True)
            if self.is_test:
                return src_batch, None
            trg_batch = pad_sequence([bat[1] for bat in batch], padding_value=TranslationDataset.PAD_ID,
                                     batch_first=True)
            return src_batch, trg_batch

        return collate_fn
