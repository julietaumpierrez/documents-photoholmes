import glob
import os

import cv2 as cv
from tqdm import tqdm

from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.utils.image import read_image, save_image

PATH = "/home/dsense/extra/tesis/datos/columbia"
dataset = ColumbiaDataset(
    PATH,
)
SAVE_PATH = "/home/dsense/extra/tesis/datos/columbia_webp"


# def save_image_webp(path, img: torch.Tensor | np.ndarray, *args):
#     img_bgr = cv.cvtColor(tensor2numpy(img), cv.COLOR_RGB2BGR)
#     cv.imwrite(path + ".webp", img_bgr, *args)

TAMP_DIR = "4cam_splc"
AUTH_DIR = "4cam_auth"
MASKS_DIR = "ground-truth"
IMAGE_EXTENSION = ".tif"

for case in [TAMP_DIR, AUTH_DIR]:
    case_path = os.path.join(PATH, TAMP_DIR, f"*{IMAGE_EXTENSION}")
    print(case_path)
    for filepath in tqdm(glob.glob(case_path)):
        im = read_image(filepath)
        filename = filepath.split("/")[-1].split(".")[0]
        save_image(
            f"{SAVE_PATH}/{case}/{filename}.webp",
            im,
            [cv.IMWRITE_WEBP_QUALITY, 80],
        )

# for x, mask, name in tqdm(dataset):
#     im = x["image"]
#     filename = name.split("_")[-1]
#     print(name)
#     case = "tampered" if (mask != 0).any() else "pristine"
#     save_image(
#         f"{SAVE_PATH}/{case}/{filename}.webp",
#         im,
#         [cv.IMWRITE_WEBP_QUALITY, 80],
#     )
