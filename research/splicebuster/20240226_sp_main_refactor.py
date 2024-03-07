import array
import glob
import os
from fileinput import filename

from sympy import plot
from torch import load, save

from photoholmes.methods.method_factory import MethodFactory
from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.image import plot_multiple, read_image

IM_NAMES = [f"img{i:02d}" for i in range(5)]
RUN_REFERENCE = True

METHOD_NAME = "splicebuster"
CONFIG_PATH = "src/photoholmes/methods/splicebuster/config.yaml"
REFS = "data/debug/splicebuster/refactor_references/"

config = load_yaml(CONFIG_PATH)
method, preprocess = MethodFactory.load(METHOD_NAME, config)
for im_name in IM_NAMES:
    IMAGE_PATH = "data/debug/" + im_name + ".png"
    REF_IMAGE_PATH = os.path.join(REFS, im_name + "-heatmap.pkl")
    im = read_image(IMAGE_PATH)
    im_preprocessed = preprocess(image=im)
    out = method.predict(**im_preprocessed)
    heatmap = out["heatmap"]
    if RUN_REFERENCE:
        plot_multiple([im, heatmap])
        save(heatmap, REF_IMAGE_PATH)
    else:
        true_heatmap = load(REF_IMAGE_PATH)
        assert true_heatmap.shape == heatmap.shape
        assert (true_heatmap == heatmap).all()
