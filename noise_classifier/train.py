"""Command line training utility for the noise classifier."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import DataConfig, create_dataloaders
from .models import NoiseClassifier
from .registry import ModelRegistry


def _train_one_epoch(
    model: NoiseClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * signals.size(0)
        total += signals.size(0)
    return total_loss / max(total, 1)


def _evaluate(
    model: NoiseClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * signals.size(0)
            total += signals.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / max(total, 1)
    return {"loss": total_loss / max(total, 1), "accuracy": accuracy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a noise classification model.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--signal-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--noise-probability", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes stored in the registry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_config = DataConfig(
        num_samples=args.num_samples,
        signal_length=args.signal_length,
        batch_size=args.batch_size,
        train_split=args.train_split,
        noise_probability=args.noise_probability,
    )

    train_loader, val_loader = create_dataloaders(data_config, seed=args.seed)
    model = NoiseClassifier(input_length=args.signal_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_metrics = {"accuracy": 0.0, "loss": float("inf")}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = _evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, val_loss: {metrics['loss']:.4f}, val_acc: {metrics['accuracy']:.4f}"
        )
        if metrics["accuracy"] >= best_metrics["accuracy"]:
            best_metrics = metrics
            best_state = model.state_dict()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = f"noise_classifier_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{model_name}.pt"
    torch.save(best_state or model.state_dict(), model_path)

    metadata = {
        "name": model_name,
        "model_path": str(model_path),
        "metrics": best_metrics,
        "config": data_config.__dict__,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    }
    metadata_path = args.output_dir / f"{model_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    registry = ModelRegistry(args.output_dir)
    registry.register(
        model_name,
        model_path,
        metrics=best_metrics,
        config={**data_config.__dict__, "epochs": args.epochs, "learning_rate": args.learning_rate},
        notes=args.notes,
    )
    print(f"Saved best model to {model_path}")
    print(f"Metadata stored at {metadata_path}")
    print(f"Registry updated at {registry.registry_path}")


if __name__ == "__main__":
    main()
