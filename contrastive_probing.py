import torch
import torchvision as tv
import tqdm
import numpy as np


def _get_datasets(batch_size):
    tfs = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dl = torch.utils.data.DataLoader(
        tv.datasets.CIFAR10(root=".", download=True, train=True, transform=tfs),
        batch_size=batch_size, shuffle=True, num_workers=10,
    )
    test_dl = torch.utils.data.DataLoader(
        tv.datasets.CIFAR10(root=".", download=True, train=False, transform=tfs),
        batch_size=batch_size, shuffle=False, num_workers=10,
    )

    return train_dl, test_dl


def _masked_dot_products(m1_feats, m2_feats, labels):
    marginal_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    joint_mask = (labels[:, None] == labels) & marginal_mask
    dot_mat = torch.matmul(m1_feats, m2_feats.T)
    return torch.masked_select(dot_mat, joint_mask), torch.masked_select(dot_mat, marginal_mask)


def _remine(m1_feats, m2_feats, labels, alpha=0.1):
    batch_size = labels.shape[0]
    joint, marginal = _masked_dot_products(m1_feats, m2_feats, labels)

    t = joint.mean()
    et = torch.logsumexp(marginal, dim=0) - np.log(batch_size * (batch_size - 1))
    return t - et - alpha * torch.square(et)


if __name__ == "__main__":
    device = 0
    epochs = 10
    batch_size = 100

    train_dl, test_dl = _get_datasets(batch_size=batch_size)

    m1 = tv.models.resnet18(pretrained=True).to(device)
    m2 = tv.models.densenet121(pretrained=True).to(device)
    optimizer = torch.optim.Adam(list(m1.parameters()) + list(m2.parameters()))

    for epoch in range(epochs):
        train_loop = tqdm.tqdm(enumerate(train_dl, 0))
        for i, (inputs, labels) in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            m1_feats = m1(inputs)
            m2_feats = m2(inputs)
            mi = _remine(m1_feats, m2_feats, labels)

            (-mi).backward()
            optimizer.step()
            train_loop.set_description(f'Epoch [{epoch}/{epochs}] {mi.item():.4f}')
