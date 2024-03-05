import glob
import os

import cv2 as cv
from tqdm import tqdm

from photoholmes.datasets.realistic_tampering import RealisticTamperingDataset
from photoholmes.utils.image import read_image, save_image

PATH = "/home/dsense/extra/tesis/datos/realistic_tampering"
dataset = RealisticTamperingDataset(
    PATH,
)
SAVE_PATH = "/home/dsense/extra/tesis/datos/realistic_tampering_webp"


# def save_image_webp(path, img: torch.Tensor | np.ndarray, *args):
#     img_bgr = cv.cvtColor(tensor2numpy(img), cv.COLOR_RGB2BGR)
#     cv.imwrite(path + ".webp", img_bgr, *args)

CAMERA_FOLDERS = ["Canon_60D", "Nikon_D90", "Nikon_D7000", "Sony_A57"]
TAMP_DIR = "tampered-realistic"
AUTH_DIR = "pristine"
MASKS_DIR = "ground-truth"
IMAGE_EXTENSION = ".TIF"

for camera in CAMERA_FOLDERS:
    print("Camera case:", camera)
    for case in [TAMP_DIR, AUTH_DIR]:
        case_path = os.path.join(PATH, camera, TAMP_DIR, f"*{IMAGE_EXTENSION}")
        print(case_path)
        for filepath in tqdm(glob.glob(case_path)):
            im = read_image(filepath)
            filename = filepath.split("/")[-1].split(".")[0]
            save_image(
                f"{SAVE_PATH}/{camera}/{case}/{filename}.webp",
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
