from enum import Enum, unique


@unique
class MetricName(Enum):
    AUROC = "auroc"
    FPR = "fpr"
    IoU = "iou"
    MAP = "map"
    MCC = "mcc"
    Precision = "precision"
    ROC = "roc"
    TPR = "tpr"
