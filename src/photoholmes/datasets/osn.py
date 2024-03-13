import os

from .casia1 import CasiaBaseDataset
from .columbia import ColumbiaDataset


class Casia1SplicingOSNDataset(CasiaBaseDataset):
    """
    Class for the CASIA 1.0 Splicing subset with Online Social Networks (OSN)
    modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    │   ├── Sp
    │   │   └── [Spliced images in jpg]
    │   ├── CASIA_GT
    │       └── [Copy Move and Spliced masks in png]
    └── Possibly more files
    """

    IMAGES_TAMPERED_DIR = "Casia_Facebook/Sp"
    MASK_TAMPERED_DIR = "Casia_Facebook/CASIA_GT"


class Casia1CopyMoveOSNDataset(CasiaBaseDataset):
    """
    Class for the CASIA 1.0 Copy Move subset with Online Social Networks (OSN)
    modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    │   ├── Sp
    │   │   └── [Spliced images in jpg]
    │   └── CASIA_GT
    │       └── [Copy Move and Spliced masks in png]
    ├── Columbia_Facebook
    └── DSO_Facebook
    """

    IMAGES_TAMPERED_DIR = "Casia_Facebook/CM"
    MASK_TAMPERED_DIR = "Casia_Facebook/CASIA_GT"


class ColumbiaOSNDataset(ColumbiaDataset):
    """
    Class for the Columbia Uncompressed Image Splicing Detection dataset with
    Online Social Networks (OSN) modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    ├── Columbia_Facebook
    │   ├── Columbia_GT
    │   │   └── [masks in PNG]
    │   └── [images in JPG]
    └── DSO_Facebook
    """

    TAMP_DIR = "Columbia_Facebook"
    IMAGE_EXTENSION = ".jpg"
    MASKS_DIR = "Columbia_Facebook/Columbia_GT"
    MASK_EXTENSION = ".png"

    def _get_mask_path(self, image_path: str) -> str:
        """
        Get the path of the mask for the given image path.

        Args:
            image_path (str): Path to the image.

        Returns:
            str: Path to the mask.
        """
        image_filename = image_path.split("/")[-1]
        image_name_list = ".".join(image_filename.split(".")[:-1]).split("_")
        mask_name = "_".join(image_name_list + ["gt"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.MASKS_DIR, mask_filename)
