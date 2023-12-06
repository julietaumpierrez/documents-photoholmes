import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from photoholmes.methods.psccnet.config import pretrained_arch
from photoholmes.methods.psccnet.network.detection_head import DetectionHead
from photoholmes.methods.psccnet.network.nlc_detection import NLCDetection
from photoholmes.methods.psccnet.network.seg_hrnet import HighResolutionNet
from photoholmes.methods.psccnet.network.seg_hrnet_config import get_hrnet_cfg

# from photoholmes.methods.psccnet.utils import load_network_weight
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.image import plot, plot_multiple, read_image

IM_PATH = "data/copymove2.png"
im = read_image(IM_PATH)[None, :, :, :]


class TestData(Dataset):
    def __init__(self):
        super(TestData, self).__init__()

        ddir = "data/"
        names = os.listdir(ddir)
        authentic_names = []
        authentic_class = [0] * len(authentic_names)

        fake_names = ["data/copymove2.png"]
        fake_class = [1] * len(fake_names)

        self.image_names = authentic_names + fake_names
        self.image_class = authentic_class + fake_class

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype="float32")
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype="float32") / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype="uint8")

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = imageio.imread(image_name)

        if image.shape[-1] == 4:
            image = self.rgba2rgb(image)

        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        return image, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)


def load_network_weight(net, weight_path, name):
    net_state_dict = torch.load(weight_path, map_location=DEVICE)
    net.load_state_dict(net_state_dict)
    print("{} weight-loading succeeds".format(name))


def test():
    # define backbone
    FENet_name = "HRNet"
    FENet_cfg = get_hrnet_cfg()
    FENet = HighResolutionNet(FENet_cfg)
    FENet.init_weights(FENet_cfg.PRETRAINED, DEVICE)

    # define localization head
    SegNet_name = "NLCDetection"
    SegNet = NLCDetection(pretrained_arch, CROP_SIZE)

    # define detection head
    ClsNet_name = "DetectionHead"
    ClsNet = DetectionHead(pretrained_arch, CROP_SIZE)

    FENet_checkpoint_dir = "weights/pscc/HRNet.pth"
    SegNet_checkpoint_dir = "weights/pscc/NLCDetection.pth"
    ClsNet_checkpoint_dir = "weights/pscc/DetectionHead.pth"

    # load FENet weight
    FENet = FENet.to(DEVICE)
    FENet = torch.nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)
    FENet = FENet.module.to(DEVICE)

    # load SegNet weight
    SegNet = SegNet.to(DEVICE)
    SegNet = torch.nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)
    SegNet = SegNet.module.to(DEVICE)

    # load ClsNet weight
    ClsNet = ClsNet.to(DEVICE)
    ClsNet = torch.nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)
    ClsNet = ClsNet.module.to(DEVICE)
    image = im.float().to(DEVICE)

    test_data_loader = DataLoader(
        TestData(), batch_size=1, shuffle=False, num_workers=8
    )
    feat, pred_mask, pred_logit = None, None, None
    for batch_id, test_data in enumerate(test_data_loader):
        image, cls, name = test_data
        image = image.to(DEVICE)
        with torch.no_grad():
            # backbone network
            FENet.eval()
            feat = FENet(image)

            # localization head
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            pred_mask = F.interpolate(
                pred_mask,
                size=(image.size(2), image.size(3)),
                mode="bilinear",
                align_corners=True,
            )

            # classification head
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # ce
        sm = torch.nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        # _, binary_cls = torch.max(pred_logit, 1)

        # pred_tag = "forged" if binary_cls.item() == 1 else "authentic"

        # # if args.save_tag:
        # save_image(pred_mask, name, "mask")
        # torch.save(feat, f"feat_results/{name[0].split('/')[-1]}.pth")

    return feat, pred_mask, pred_logit


test()
true_mk_pred = torch.load("data/debug/copymove2.png.pth")
assert true_mk_pred[0].size() == feat[0].size()
print(torch.Tensor.norm(true_mk_pred[0] - feat[0]))

plot_multiple([im[0], mk_pred[0]], title="Imagen y máscara predecida")
