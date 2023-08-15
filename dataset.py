from pathlib import Path

import librosa
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from transformers import Wav2Vec2FeatureExtractor


class CommonPhoneDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        assert split in ("train", "dev", "test")
        self.df = df[df.split == split]

        self.index_to_label = sorted(df.text.unique())
        self.label_to_index = {l: i for i, l in enumerate(self.index_to_label)}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(row.audio, sr=16000, mono=True)
        return {
            "audio": audio,
            "label": self.label_to_index[row.text],
            "min": row["min"],
            "max": row["max"],
        }


class CommonPhoneDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_path: Path,
            model_name: str,
            max_length: float,
            batch_size: int,
            num_workers: int,
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()

        df = pd.read_csv(dataset_path)
        self.train_ds = CommonPhoneDataset(df, "train")
        self.val_ds = CommonPhoneDataset(df, "dev")
        self.test_ds = CommonPhoneDataset(df, "test")
        self.num_classes = len(self.train_ds.index_to_label)

        self.extractor = Wav2Vec2FeatureExtractor(return_attention_mask=True)

    def _common_kwargs(self):
        return {
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "collate_fn": self._collator,
        }

    def _collator(self, batch):
        # Normalizing & Padding
        inputs = self.extractor(
            [b["audio"] for b in batch],
            sampling_rate=16000,
            max_length=int(16000 * self.hparams.max_length),
            padding="max_length",
            return_tensors="pt",
        )

        # Specific text location
        for i, b in enumerate(batch):
            mask = torch.zeros_like(inputs["attention_mask"][i])
            mask[int(b["min"] * 16000) : int(b["max"] * 16000)] = 1
            inputs["attention_mask"][i] &= mask

        # Final label
        labels = torch.LongTensor([b["label"] for b in batch])

        return inputs, labels

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, shuffle=True, **self._common_kwargs())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, **self._common_kwargs())
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, shuffle=False, **self._common_kwargs())
