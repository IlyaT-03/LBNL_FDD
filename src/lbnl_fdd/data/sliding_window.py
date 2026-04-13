from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        window_size: int,
        stride: int = 1,
        get_step_next: bool = False,
    ):
        self.features = list(df.columns)
        self.data = torch.as_tensor(df.values, dtype=torch.float32)
        self.target = torch.as_tensor(target.values, dtype=torch.float32) if target is not None else None

        self.index = df.index
        self.window_size = window_size
        self.get_step_next = get_step_next

        # NOTE: оставляем как у тебя (если не используешь get_step_next — не важно)
        if self.get_step_next:
            self.window_size += 1

        self.valid_windows = self._precompute_valid_windows(stride)

    def _precompute_valid_windows(self, stride: int):
        valid_windows = []
        run_ids = self.index.get_level_values(0).unique()

        for run_id in tqdm(run_ids, desc="Building safe windows"):
            run_mask = self.index.get_level_values(0) == run_id
            run_indices = np.where(run_mask)[0]

            if len(run_indices) < self.window_size:
                continue

            # end_pos — позиция последнего элемента окна (inclusive) внутри run
            for end_pos in range(self.window_size - 1, len(run_indices), stride):
                start_pos = end_pos - (self.window_size - 1)

                start_idx = run_indices[start_pos]
                end_idx = run_indices[end_pos]  # inclusive

                # корректная проверка границы run_id
                assert self.index[start_idx][0] == self.index[end_idx][0], "Window crosses run_id boundary!"

                valid_windows.append((start_idx, end_idx))

        return valid_windows

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        start_idx, end_idx = self.valid_windows[idx]

        # включаем end_idx => +1
        sample = self.data[start_idx:end_idx + 1]

        target = self.target[start_idx:end_idx + 1].max() if self.target is not None else sample[-1]
        return sample, target