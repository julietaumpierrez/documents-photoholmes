from tqdm import tqdm

from photoholmes.datasets.realistic_tampering import RealisticTamperingDataset
from photoholmes.utils.image import read_image, save_image

PATH = "/home/dsense/extra/tesis/datos/realistic_tampering"
dataset = RealisticTamperingDataset(
    PATH,
)
SAVE_PATH = "/home/dsense/extra/tesis/datos/realistic_tampering_webp"

for x, mask, name in tqdm(dataset):
    im = x["image"]
    filename = name.split("_")[-1]
    case = "tampered" if (mask != 0).any() else "pristine"
    save_image(f"{SAVE_PATH}/{case}/{filename}.webp", im)
