"""Utilities for tracking trained models on disk."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelRecord:
    name: str
    path: str
    created_at: str
    metrics: Dict[str, float]
    config: Dict[str, float | int | str]
    notes: str | None = None


class ModelRegistry:
    """Persist metadata for trained models in a simple JSON file."""

    def __init__(self, root: Path, registry_filename: str = "registry.json") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / registry_filename
        self._records: Dict[str, ModelRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text())
            self._records = {
                name: ModelRecord(**record)
                for name, record in data.items()
            }

    def save(self) -> None:
        serializable = {name: asdict(record) for name, record in self._records.items()}
        self.registry_path.write_text(json.dumps(serializable, indent=2))

    def register(
        self,
        name: str,
        model_path: Path,
        *,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, float | int | str]] = None,
        notes: str | None = None,
        overwrite: bool = False,
    ) -> ModelRecord:
        if not overwrite and name in self._records:
            raise ValueError(f"Model '{name}' already exists. Use overwrite=True to replace it.")
        record = ModelRecord(
            name=name,
            path=str(model_path),
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            config=config or {},
            notes=notes,
        )
        self._records[name] = record
        self.save()
        return record

    def get(self, name: str) -> ModelRecord:
        return self._records[name]

    def list(self) -> List[ModelRecord]:
        return list(self._records.values())

    def remove(self, name: str) -> None:
        if name in self._records:
            del self._records[name]
            self.save()


def load_registry(root: str | Path) -> ModelRegistry:
    return ModelRegistry(Path(root))


if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(description="Manage trained noise classification models.")
    parser.add_argument("root", type=Path, help="Directory where models and registry.json live")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List models in the registry")
    group.add_argument("--remove", metavar="NAME", help="Remove a model by name")
    args = parser.parse_args()

    registry = ModelRegistry(args.root)
    if args.list:
        records = registry.list()
        if not records:
            print("No models registered yet.")
        else:
            for record in records:
                print(textwrap.dedent(
                    f"""
                    Name: {record.name}
                    Path: {record.path}
                    Created: {record.created_at}
                    Metrics: {record.metrics}
                    Config: {record.config}
                    Notes: {record.notes or '-'}
                    """
                ).strip())
    else:
        registry.remove(args.remove)
        print(f"Removed model '{args.remove}'.")
