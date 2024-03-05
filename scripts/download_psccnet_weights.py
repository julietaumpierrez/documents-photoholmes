import argparse
import os
from pathlib import Path

import wget  # pip install wget

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PSCCNet weights")
    parser.add_argument(
        "weights_folder", type=str, help="weights folder to download the weights to."
    )
    args = parser.parse_args()

    weights_folder = Path(args.weights_folder)
    os.makedirs(weights_folder, exist_ok=True)

    print("Downloading FENet")
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/HRNet_checkpoint/HRNet.pth",  # noqa: E501
        out=str(weights_folder / "psccnet/FENet.pth"),
    )
    print("\nDownloading NLCDetection")
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth",  # noqa: E501
        out=str(weights_folder / "psccnet/SegNet.pth"),
    )
    print("\nDownloading ClsNet")
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth",  # noqa: E501
        out=str(weights_folder / "psccnet/ClsNet.pth"),
    )
    print()
