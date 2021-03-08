#!/usr/bin/env python

import torch

from train import Dataset
from train import get_model


data = Dataset(split="train")
loader = torch.utils.data.DataLoader(data, batch_size=1)

model = get_model()
model.eval()
model.load_state_dict(
    torch.load("./data/train/20210225_180602/model_best.pth")
)

for x, y_true in loader:
    x = x.to("cuda:0")
    y_true = y_true.to("cuda:0")

    with torch.no_grad():
        y_pred = model(x)

        print(y_pred[0, 5], y_true[0, 5])
