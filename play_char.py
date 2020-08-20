import logging
from mingpt.utils import set_seed

import numpy as np
import torch

from mingpt.model import GPT, GPTConfig

import math
from torch.utils.data import Dataset

from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample

set_seed(42)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size = len(data)
        vocab_size = len(chars)
        print(f"data has {data_size} characters, {vocab_size} unique")

        self.char2idx = {x: i for i, x in enumerate(chars)}
        self.idx2char = {i: x for i, x in enumerate(chars)}

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __getitem__(self, i):
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        chunk_idx = [self.char2idx[s] for s in chunk]
        x = torch.tensor(chunk_idx[:-1], dtype=torch.long)
        y = torch.tensor(chunk_idx[1:], dtype=torch.long)
        return x, y

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))


if __name__ == "__main__":
    sample_block_size = 128
    text = open("input.txt", "r", encoding="utf-8").read()
    train_dataset = CharDataset(text, block_size=sample_block_size)

    model_config = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=8,
        n_head=8,
        n_emb=512,
    )
    model = GPT(model_config)

    trainer_config = TrainerConfig(
        max_epochs=200,
        batch_size=512,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=200 * len(train_dataset) * sample_block_size,
        num_workers=4,
    )
    trainer = Trainer(model, train_dataset, None, trainer_config)
    trainer.train()

    context = "O God, O God!"
    sample_x = torch.tensor(
        [train_dataset.char2idx[s] for s in context],
        dtype=torch.long,
    )[None, ...].to(trainer.device)
    sample_y = sample(model, sample_x, 2000, temperature=0.9, sampling=True, top_k=5)[0]
    completion = "".join([train_dataset.idx2char[int(i)] for i in sample_y])
    print(completion)
