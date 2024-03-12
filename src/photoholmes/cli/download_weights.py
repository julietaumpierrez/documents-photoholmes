import logging
import os
from functools import partial
from pathlib import Path

import wget

logger = logging.getLogger(__name__)


def callback(current, total, width=80, message: str = "Downloading..."):
    def progress_bar(current, total, width=80):
        progress = int(width * current / total)
        bar = "[" + "=" * progress + " " * (width - progress) + "]"
        return bar

    percent = current / total * 100
    progress = progress_bar(current, total, width)
    print(f"\r{message}: {percent:.2f}% {progress}", end="", flush=True)


def download_focal_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/HRNet_checkpoint/HRNet.pth",  # noqa: E501
        out=str(weights_folder / "FENet.pth"),
        bar=partial(callback, message="Downloading FENet"),
    )
    print()
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth",  # noqa: E501
        out=str(weights_folder / "SegNet.pth"),
        bar=partial(callback, message="Downloading SegNet"),
    )
    print()
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth",  # noqa: E501
        out=str(weights_folder / "ClsNet.pth"),
        bar=partial(callback, message="Downloading ClsNet"),
    )
    print()
    logger.info(f"Downloaded PSCC-Net weights to {weights_folder}")


def download_exif_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://drive.usercontent.google.com/download?id=1qHG-m0cLsT_wEUrOX1coX8q1jUY8ObCK&export=download&authuser=0&confirm=t&uuid=e9b0e308-1a90-4c6c-afce-970f5654f253&at=APZUnTXYD5ThqZsRJXHeY0PIxhHF%3A1710200587419",  # noqa
        out=str(weights_folder / "weights.pth"),
        bar=partial(callback, message="Downloading weights"),
    )
    print()
    logger.info(f"Downloaded Exif as Language weights to {weights_folder}")


def download_adaptive_cfa_net_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://raw.githubusercontent.com/qbammey/adaptive_cfa_forensics/master/src/models/pretrained.pt",  # noqa
        out=str(weights_folder / "weights.pth"),
        bar=partial(callback, message="Downloading weights"),
    )
    print()
    logger.info(f"Downloaded Adaptive CFA Net weights to {weights_folder}")
