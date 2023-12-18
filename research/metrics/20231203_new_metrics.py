# %%
import os

if "research" in os.path.abspath("."):
    os.chdir("../../")
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
# %%
from src.photoholmes.metrics.metric_factory import MetricFactory

# do tests for all the metrics in the registry
# %%
from src.photoholmes.metrics.registry import MetricName

# %%
metric_names = list(MetricName)
metric_names
# %%
import torch

pred1 = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
pred2 = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
)
pred3 = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
mask = torch.tensor(
    [
        [0, 0, 0],
        [0, 1, 1],
    ]
)
# %%
metrics = [metric.value for metric in metric_names]

metrics
# %%
metrics_objects = MetricFactory.load(metrics)
metrics_objects
# %%
for metric in metrics_objects:
    print(metric)
    metric.update(pred1, mask)
    metric.update(pred2, mask)
    metric.update(pred3, mask)
    print(metric.compute())
    metric.reset()
    print("-" * 80)
# %%
metric_names[0].value
# %%
metrics2 = []
MetricFactory.load(metrics2)
# %%
