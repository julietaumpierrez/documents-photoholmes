import os

import numpy as np
import torch
from torch.utils.data import Dataset

from photoholmes.methods.method_factory import MethodFactory
from photoholmes.utils.image import plot_multiple, read_image

SAMPLE_PATH = "/home/dsense/extra/tesis/photoholmes/data/debug/zero"
FILENAMES = ["pelican.png", "roma.png", "tampered1.png", "tampered2.png"]


class SampleDataset(Dataset):
    def __init__(self, sample_dir, transform, filenames=None) -> None:
        super().__init__()
        self.sample_dir = sample_dir
        self.transform = transform
        self.image_paths = os.listdir(sample_dir) if filenames is None else filenames

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        im_name = self.image_paths[index]
        print(os.path.join(self.sample_dir, im_name))
        image = read_image(os.path.join(self.sample_dir, im_name))
        x = {"image": image}
        if self.transform:
            x = self.transform(**x)
        return x


method_name = "zero"
method, pre = MethodFactory.load(method_name)
dataset = SampleDataset(SAMPLE_PATH, transform=pre, filenames=FILENAMES)

true_mks = [
    np.loadtxt(os.path.join(SAMPLE_PATH, f"output_{f}.csv"), delimiter=",")
    for f in FILENAMES
]

ims = []
mks = []
for x in dataset:
    im = x["image"]
    ims.append(im)
    out = method.predict(**x)
    mask = out["mask"].numpy()
    mks.append(mask)

plot_multiple(ims + true_mks + mks, titles=FILENAMES * 3, ncols=4)

for i in range(len(mks)):
    print("Coincide:", (mks[i] == true_mks[i]).all())
