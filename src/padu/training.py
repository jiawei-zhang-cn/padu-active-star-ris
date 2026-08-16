"""Training helpers with explicit trajectory isolation and device placement."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import SharedGRUWindowDataset
from .models import SharedGRUPredictor


@dataclass(frozen=True)
class GRUTrainingSettings:
    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: float
    gradient_norm_limit: float
    data_loader_workers: int

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_norm_limit <= 0.0:
            raise ValueError("gradient_norm_limit must be positive")
        if self.data_loader_workers < 0:
            raise ValueError("data_loader_workers must be non-negative")


@dataclass(frozen=True)
class GRUTrainingResult:
    training_mse: tuple[float, ...]
    validation_mse: tuple[float, ...]
    best_epoch: int
    best_validation_mse: float


def set_reproducible_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_shared_gru(
    model: SharedGRUPredictor,
    training_dataset: SharedGRUWindowDataset,
    validation_dataset: SharedGRUWindowDataset,
    settings: GRUTrainingSettings,
    device: torch.device,
    checkpoint_path: str | Path,
    seed: int,
) -> GRUTrainingResult:
    settings.validate()
    if len(training_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("training and validation datasets must be non-empty")
    set_reproducible_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.data_loader_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.data_loader_workers,
        pin_memory=device.type == "cuda",
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_function = nn.MSELoss()
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    training_history: list[float] = []
    validation_history: list[float] = []
    best_validation = float("inf")
    best_epoch = -1

    for epoch in range(settings.epochs):
        model.train()
        training_loss = _run_gru_epoch(
            model,
            training_loader,
            loss_function,
            device,
            optimizer,
            settings.gradient_norm_limit,
        )
        model.eval()
        with torch.no_grad():
            validation_loss = _run_gru_epoch(
                model,
                validation_loader,
                loss_function,
                device,
                optimizer=None,
                gradient_norm_limit=None,
            )
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        print(
            "[gru] "
            f"epoch={epoch + 1}/{settings.epochs} "
            f"training_mse={training_loss:.10g} "
            f"validation_mse={validation_loss:.10g}",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_mse": validation_loss,
                    "seed": seed,
                },
                checkpoint,
            )

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    return GRUTrainingResult(
        training_mse=tuple(training_history),
        validation_mse=tuple(validation_history),
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
    )


def _run_gru_epoch(
    model: SharedGRUPredictor,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_norm_limit: float | None,
) -> float:
    total_loss = 0.0
    total_samples = 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(history)
        loss = loss_function(prediction, target)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_norm_limit),
            )
            optimizer.step()
        batch_size = history.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def predict_shared_gru(
    model: SharedGRUPredictor,
    normalized_history: Tensor,
    device: torch.device,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        return model(normalized_history.to(device)).cpu()
