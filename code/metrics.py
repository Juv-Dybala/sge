from typing import Sequence, Union
import numpy as np
import scipy.stats
from sklearn.metrics import average_precision_score,precision_score

from .registry import registry


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total
    

@registry.register_metric('auprc')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    
    labels = []
    preds = []
    for label, score in zip(target, prediction):
        label_array = np.asarray(label)
        pred_array = np.asarray(score).argmax(-1)
        mask = label_array != -1
        labels.extend(label_array[mask])
        preds.extend(pred_array[mask])

    auprc = average_precision_score(labels,preds)
    return auprc