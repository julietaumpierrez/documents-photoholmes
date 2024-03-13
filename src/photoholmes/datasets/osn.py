from .casia1 import CasiaBaseDataset


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
    │   ├── CASIA_GT
    │       └── [Copy Move and Spliced masks in png]
    └── Possibly more files
    """

    IMAGES_TAMPERED_DIR = "Casia_Facebook/CM"
    MASK_TAMPERED_DIR = "Casia_Facebook/CASIA_GT"
