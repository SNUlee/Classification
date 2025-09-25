"""Dataset utilities for the noise classification project."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def _generate_noise_signal(length: int) -> torch.Tensor:
    return torch.randn(length)


def _generate_clean_signal(length: int) -> torch.Tensor:
    freq = random.uniform(1.0, 6.0)
    phase = random.uniform(0.0, math.pi)
    t = torch.linspace(0, 1, steps=length)
    sine = torch.sin(2 * math.pi * freq * t + phase)
    amplitude = random.uniform(0.5, 1.0)
    signal = amplitude * sine
    noise_level = random.uniform(0.0, 0.2)
    return signal + noise_level * torch.randn(length)


class SyntheticNoiseDataset(Dataset):
    """Dataset that labels pure Gaussian noise against structured signals."""

    def __init__(
        self,
        num_samples: int,
        length: int = 256,
        noise_probability: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if not 0 <= noise_probability <= 1:
            raise ValueError("noise_probability must be between 0 and 1")
        self.num_samples = num_samples
        self.length = length
        self.noise_probability = noise_probability
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        self.seed = seed

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def _generate(self, index: int) -> Tuple[torch.Tensor, int]:
        if self.seed is not None:
            torch.manual_seed(self.seed + index)
            random.seed(self.seed + index)
        label = 1 if random.random() < self.noise_probability else 0
        if label == 1:
            signal = _generate_noise_signal(self.length)
        else:
            signal = _generate_clean_signal(self.length)
        return signal.unsqueeze(0), label

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal, label = self._generate(index)
        return signal.float(), torch.tensor(label, dtype=torch.long)


@dataclass
class DataConfig:
    num_samples: int = 2048
    signal_length: int = 256
    batch_size: int = 64
    train_split: float = 0.8
    noise_probability: float = 0.5


def create_dataloaders(
    config: DataConfig,
    *,
    seed: int | None = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    dataset = SyntheticNoiseDataset(
        num_samples=config.num_samples,
        length=config.signal_length,
        noise_probability=config.noise_probability,
        seed=seed,
    )
    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed or 0)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
