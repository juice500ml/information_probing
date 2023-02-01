import torch
import torchvision.transforms as tvtransforms
import torchvision.models as tvmodels
import librosa
import librosa.display
import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt
import itertools


def get_signal(freq=None, mag=1.0, dur=1.0, sr=16_000):
    if not isinstance(freq, Iterable):
        freq = [freq]
    if not isinstance(mag, Iterable):
        mag = [mag] * len(freq)

    assert len(freq) == len(mag)

    y = 0.0
    for f, m in zip(freq, mag):
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        y += m * np.sin(2 * np.pi * f * t)
    return y


def get_melspec(signal, sr=16_000):
    s = librosa.feature.melspectrogram(y=signal, sr=sr)
    s_db = librosa.power_to_db(s, ref=np.max)
    return s_db


class AudioClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, cases: np.array):
        self.cases = cases
        self._mean, self._std = self._get_stat(cases)
        self._imgs = self._transform(cases)
        self._lbls = torch.LongTensor(list(range(len(cases))))

    def _transform(self, cases):
        cases = np.tile(cases, (3, 1, 1, 1)).swapaxes(0, 1)  # gray -> rgb
        cases = torch.FloatTensor(cases)
        cases = tvtransforms.functional.resize(cases, (224, 224))
        cases = tvtransforms.functional.normalize(cases, mean=self._mean, std=self._std)
        return cases

    def _get_stat(self, cases):
        mean = cases.flatten().mean()
        std = cases.flatten().std()
        return mean, std

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, i):
        return self._imgs[i], self._lbls[i]


def get_dataloader(dataset, batch_size=8, infinite=True):
    if infinite:
        while True:
            dl = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            for d in dl:
                yield d
    else:
        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        for d in dl:
            yield d


def _mask(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).bool()


def _mine(logits, labels):
    batch_size, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    t = torch.mean(joints)
    et = torch.logsumexp(logits, dim=(0, 1)) - np.log(batch_size * classes)
    return t, et


def remine(logits, labels, alpha=1.0):
    t, et = _mine(logits, labels)
    return t - et - alpha * torch.square(et)


def estimate_mi(model, dl):
    device = next(model.parameters()).device

    logits = []
    labels = []
    for x, y in dl:
        x = x.to(device)
        logits.append(model(x).detach().cpu().numpy())
        labels.append(y.numpy())
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)

    t, et = _mine(torch.FloatTensor(logits), torch.LongTensor(labels))
    return (t - et).numpy()



if __name__ == "__main__":
    cases = np.array([
        get_melspec(get_signal(f * 128))
        for f in range(1, 11)
    ])
    ds = AudioClassificationDataset(cases)
    train_dl = get_dataloader(ds, infinite=True)

    model = tvmodels.resnet18(num_classes=len(cases))
    optim = torch.optim.Adam(model.parameters())
    device = next(model.parameters()).device
    
    total_it = 10000
    infer_it = 100

    for it, (x, y) in enumerate(train_dl):
        if it % infer_it == 0:
            test_dl = get_dataloader(ds, infinite=False)
            print(it, estimate_mi(model, test_dl))

        logits = model(x.to(device))
        labels = y.to(device)

        # loss = -remine(logits, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()
        optim.step()

        if it >= total_it:
            break
