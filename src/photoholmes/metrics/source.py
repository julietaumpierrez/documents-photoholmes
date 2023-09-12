import numpy as np

def metric_mapping(metric_name:str):
    if metric_name == "iou":
        return iou

def iou(pred, true):
    '''Computes the Intersection over Union of predicted and true masks of an image, or over a list of such.
    '''
    if (type(pred)==list):
        assert(len(pred) == len(true))
        intersections = [np.logical_and(pred[i], true[i]) for i in range(len(pred))]
        unions = [np.logical_or(pred[i], true[i]) for i in range(len(pred))]
        ious = [np.sum(intersections[i]) / np.sum(unions[i]) for i in range(len(pred))]
        return np.array(ious)
    else:
        assert(pred.shape == true.shape)
        intersection = np.logical_and(pred, true)
        union = np.logical_or(pred, true)
        iou = np.sum(intersection) / np.sum(union)  
        return iou