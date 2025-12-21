import itertools
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
import tqdm
from candidate_dataset import CandidateDataFrame


class Projector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super(Projector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor):
        logits = self.projector(features)
        logits = torch.clamp(logits, min=-2.0, max=2.0)
        return logits


class CheatingDetector:
    def __init__(self, num_features: int, hidden_dim: int, dropout: float):
        self.num_features = num_features
        self.classifier: torch.nn.Module = Projector(input_dim=num_features, output_dim=1, hidden_dim=hidden_dim,
                                                     dropout=dropout)

    def train(
            self,
            dataloader: DataLoader,
            epochs: int = 5,
            lr: float = 1e-3,
            device: Optional[torch.device] = None,
            pos_weight: Optional[float] = None,
    ):
        """
        Simple training loop using binary cross entropy on the is_cheating label.
        pos_weight boosts the loss for positives (cheaters) to counter imbalance.
        Provide pos_weight as the negative/positive ratio computed externally.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.classifier.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # Increase penalty on false negatives to address class imbalance (cheaters are rare).
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight is not None else None
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for features, labels in tqdm.tqdm(dataloader):
                features = features.to(device)
                targets = labels[:, 0].float().to(device)  # is_cheating
                high_conf_clean = labels[:,1].float().to(device) # high_conf_clean

                optimizer.zero_grad()
                logits = model(features)[:, 0]
                loss = bce_loss(logits, targets) + .1 * (torch.exp(logits) * high_conf_clean).mean()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    running_loss += loss.item() * features.size(0)
                    running_correct += (preds == targets).sum().item()
                    total += features.size(0)

            epoch_loss = running_loss / max(1, total)
            epoch_acc = running_correct / max(1, total)
            print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

        return model


class CandidateSampleDataset(Dataset):
    """
    Map-style dataset that mirrors the full training dataframe (excluding ID).
    Labels are 2D: [is_cheating, high_conf_clean].
    Optional class balancing is handled by the sampler, not by subsampling here.
    """

    def __init__(self, candidate_dataset: CandidateDataFrame):
        if candidate_dataset.df_train.is_empty():
            raise ValueError("CandidateDataset.df_train is empty. Run build_pipeline first.")

        columns_wo_id = [c for c in candidate_dataset.df_train.columns if c != "ID"]
        label_cols = ["is_cheating", "high_conf_clean"]
        for lc in label_cols:
            if lc not in columns_wo_id:
                raise ValueError(f"Column '{lc}' not found in training dataframe.")

        feature_cols = [c for c in columns_wo_id if c not in label_cols]
        features_np = candidate_dataset.df_train.select(feature_cols).to_numpy()
        labels_np = candidate_dataset.df_train.select(label_cols).to_numpy()

        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.labels = torch.tensor(labels_np, dtype=torch.int64)

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class BalancedBatchSampler(Sampler[list[int]]):
    """
    Yields batches with a target positive fraction.
    Falls back to random class-balanced cycling when one class runs short.
    """

    def __init__(self, labels: torch.Tensor, batch_size: int, pos_fraction: float):
        if not (0 < pos_fraction < 1):
            raise ValueError("pos_fraction must be in (0, 1).")
        self.batch_size = batch_size
        self.pos_per_batch = max(1,
                                 min(batch_size - 1, int(round(batch_size * pos_fraction))))  # clamp between 1 and B-1
        self.neg_per_batch = batch_size - self.pos_per_batch

        # labels[:, 0] corresponds to is_cheating; high_conf_clean (labels[:, 1]) is not balanced
        pos_indices = torch.nonzero(labels[:, 0] == 1, as_tuple=False).flatten().tolist()
        neg_indices = torch.nonzero(labels[:, 0] == 0, as_tuple=False).flatten().tolist()
        if not pos_indices or not neg_indices:
            raise ValueError("Both classes are required for balanced batching.")

        self.pos_cycle = itertools.cycle(pos_indices)
        self.neg_cycle = itertools.cycle(neg_indices)
        # Number of full batches we can emit; approximate by minority class
        self.length = (min(len(pos_indices), len(neg_indices)) * 2) // self.batch_size

    def __iter__(self):
        for _ in range(self.length):
            batch = [next(self.pos_cycle) for _ in range(self.pos_per_batch)]
            batch += [next(self.neg_cycle) for _ in range(self.neg_per_batch)]
            yield batch

    def __len__(self) -> int:
        return self.length


def build_balanced_dataloader(
        candidate_dataset: CandidateDataFrame,
        batch_size: int,
        pos_fraction: Optional[float] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader over the full training data.
    If pos_fraction is provided, batches will target that positive-class proportion.
    Otherwise, standard shuffling is used.
    """
    dataset = CandidateSampleDataset(candidate_dataset)

    if pos_fraction is not None:
        sampler = BalancedBatchSampler(dataset.labels, batch_size, pos_fraction)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
