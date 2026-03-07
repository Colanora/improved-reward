import json
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CounterfactualPairDataset(Dataset):
    def __init__(self, jsonl_path: str):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.rows[index]

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return batch


def load_terminal_latent(path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    latent = torch.load(path, map_location=device or "cpu")
    if isinstance(latent, dict) and "x0" in latent:
        latent = latent["x0"]
    if isinstance(latent, dict) and "latent" in latent:
        latent = latent["latent"]
    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"Expected a tensor latent at {path}, got {type(latent)!r}.")
    return latent.float()


def preference_loss(
    chosen_score: torch.Tensor,
    rejected_score: torch.Tensor,
    beta: float,
    margin_threshold: float = 0.0,
    safeguarded: bool = False,
) -> torch.Tensor:
    margin = chosen_score - rejected_score - margin_threshold
    loss = -F.logsigmoid(beta * margin)
    if safeguarded:
        loss = loss + 0.5 * torch.relu(-margin) ** 2
    return loss


def preference_accuracy(
    chosen_score: torch.Tensor,
    rejected_score: torch.Tensor,
    margin_threshold: float = 0.0,
) -> torch.Tensor:
    return (chosen_score - rejected_score > margin_threshold).float()
