from torch.utils.data import Dataset
import pandas as pd
import torch


class SelectedWindowsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, windows_df: pd.DataFrame):
        self.windows_df = windows_df.reset_index(drop=True)

        # Cache per-group tensors
        self.group_data = {}
        for group_idx, group_df in df.groupby(level=0, sort=False):
            # drop the first level of MultiIndex, keep row order inside group
            group_values = group_df.to_numpy(copy=True)
            self.group_data[group_idx] = torch.tensor(group_values, dtype=torch.float32)

    def __len__(self):
        return len(self.windows_df)

    def __getitem__(self, idx):
        row = self.windows_df.iloc[idx]

        group_idx = row["group_idx"]
        start_pos = int(row["start_pos"])
        end_pos = int(row["endpoint_pos"])

        group_tensor = self.group_data[group_idx]

        if start_pos < 0 or end_pos >= len(group_tensor):
            raise IndexError(
                f"Window [{start_pos}, {end_pos}] is out of bounds for group {group_idx} "
                f"with length {len(group_tensor)}"
            )

        x = group_tensor[start_pos:end_pos + 1]
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y