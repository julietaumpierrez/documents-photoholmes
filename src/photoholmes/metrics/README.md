# Metrics Module

This module provides a collection of metrics for evaluating the performance of the 
implemented methods in the PhotoHolmes library with the provided datasets. 

## Available Metrics

Imported from [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/):
- AUROC: Area Under the Receiver Operating Characteristic curve. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html) 
- IoU: Intersection over Union, also known as Jaccard Index. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html)
- MCC: Matthews Correlation Coefficient. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/matthews_corr_coef.html)
- Precision: [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html)
- ROC: Receiver Operating Characteristic curve. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/roc.html)
- F1 score: [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)

Our implementations:
- TPR: True Positive Rate defined as 
$$TPR = \frac{TP}{TP + FN}$$
- meanAUROC: Calculates de auroc for each image and then averages those values over the full dataset.
- Weighted metrics:
    These metrics allow the comparison of performance between methods whose outputs are masks with methods whose outputs are heatmaps by regarding said heatmaps as maps of probability in which the value of each pixel corresponds to the probability of the pixel being forged. The same works with the detection problem in which the output is a single number that indicates the probability of the image as whole being forged. To accomplish that [Gardella, 2023](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000341) and [Bammey, 2021](https://openaccess.thecvf.com/content/WACV2022/papers/Bammey_Non-Semantic_Evaluation_of_Image_Forensics_Tools_Methodology_and_Database_WACV_2022_paper.pdf) define 
    weighted true positives, weighted false positives, weighted true negatives, and weighted false negatives as follows:

    $$
    TP_w = \sum_xH(x)M(x)
    $$

    $$
    FP_w = \sum_x(1-H(x))M(x)
    $$

    $$
    TN_w = \sum_x(1-H(x))(1-M(x))
    $$

    $$
    FN_w = \sum_xH(x)(1-M(x))
    $$
    in which $H(x)$ corresponds to the predicted output and $M(x)$ corresponds to the mask.

    The implemented weighted metrics are:
    - Weighted MCC: Mathews Correlation Coefficient
        $$
        MCC_{weighted} = \frac{TP_w \times TN_w - FP_w \times  FN_w}{\sqrt{(TP_w + FP_w)(TP_w+FN_w)(TN_w+FP_W)(TN_w+FN_w)}}
        $$
    - Weighted IoU: Intersection over Union
        $$
        IoU_{weighted} = \frac{TP_w}{TP_w + FN_w + FP_w}
        $$
    - Weighted F1: F1 score
        $$
        F1_{weighted} = \frac{2TP_w}{2TP_w + FN_w + FP_w}
        $$
    
    There are two versions of the weighted metrics:
    - v1: Corresponds to the mean version of each weighted metric. Those metrics accumulate the value of the metric for each image and then the output is the average of the metric over the full dataset. 
    This version of the metric is recommended to evaluate localization performance.
    - v2: Corresponds to the value of the metric over the full dataset as defined in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/). For each image the metric accumulates the FPw, TPW, TNw and FNw and then with those accumulations outputs the value of the metric for the full dataset. 
    This version of the metric is recommended to evaluate detection performance.

## Examples of Use

Here are some examples of how to use the metrics in this module:

### Using a single metric:

```python
from photoholmes.metrics.IoU_weighted_v1 import IoU_weighted_v1

iou_weighted_v1_metric = IoU_weighted_v1

for pred, mask in data:
    iou_weighted_v1_metric.update(pred, mask)
iou_weighted_v1 = iou_weighted_v1_metric.compute()
```

```python
from torchmetrics import AUROC

auroc_metric = AUROC()
for pred, mask in data:
    auroc_metric.update(pred, mask)
auroc_v1 = auroc_metric.compute()
```

### Using the metric factory

```python
from src.photoholmes.metrics.metric_factory import MetricFactory
from src.photoholmes.metrics.registry import MetricName

# How to import all the available metrics in the registry

metric_names = list(MetricName)
metrics = [metric.value for metric in metric_names]
metrics_objects = MetricFactory.load(metrics)

# Use one of them

metric = metrics_objects["iou_weighted_v2"]
for pred, mask in data:
    metroc.update(pred,mask)
metric_value = metric.compute()

# Load directly form the factory

metric = MetricFactory.load(["auroc"])
for pred, mask in data:
    metroc.update(pred,mask)
metric_value = metric.compute()

```
## How to add a new metric
If the metric already exists in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/) the steps to follow are:
1. Add metric to registry
    ```python
    class MetricName(Enum):
        NEW_TORCHMETRIC = "new_torch_metric"
    ```
2. Add the metric to the factory by following this template
    ``` python
    case MetricName.NEW_TORCHMETRIC:
        from torchmetrics import NewTorchMetric as NTM
        metrics.append(NTM(task="binary"))
    ```
If the metric does not exist in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/) you should follow the instructions provided by [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/) [here](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html), so the steps are as follows:
1. Create the .py file as explained in the tutorial of [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/).
2. Add metric to registry
    ```python
    class MetricName(Enum):
        FANCY_NEW_METRIC = "fancy_new_metric"
    ```
3. Add the metric to the factory by following this template
    ``` python
    case MetricName.FANCY_NEW_METRIC:
        from photoholmes.metrics.fancy_new_metric import Fancy_New_Metric as FNM
        metrics.append(FNM(task="binary"))
    ```

## References

```tex
@article{Noisesniffer,
  title={Image Forgery Detection Based on Noise Inspection: Analysis and Refinement of the Noisesniffer Method},
  author={Gardella, Marina and Mus{\'e}, Pablo and Colom, Miguel and Morel, Jean-Michel},
  journal={Preprint},
  year={2023},
  month={March},
  institution={Universitè Paris-Saclay, ENS Paris-Saclay, Centre Borelli, F-91190 Gif-sur-Yvette, France; IIE, Facultad de Ingenieria, Universidad de la Republica, Uruguay},
}
```

```tex
@misc{bammey2021nonsemantic,
      title={Non-Semantic Evaluation of Image Forensics Tools: Methodology and Database}, 
      author={Quentin Bammey and Tina Nikoukhah and Marina Gardella and Rafael Grompone and Miguel Colom and Jean-Michel Morel},
      year={2021},
      eprint={2105.02700},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```