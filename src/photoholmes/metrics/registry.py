from enum import Enum, unique


@unique
class MetricName(Enum):
    AUROC = "auroc"
    FPR = "fpr"
    IoU = "iou"
    MCC = "mcc"
    Precision = "precision"
    ROC = "roc"
    TPR = "tpr"
    IoU_WEIGHTED = "iou_weighted"
    F1_WEIGHTED = "f1_weighted"
    MCC_WEIGHTED = "mcc_weighted"
