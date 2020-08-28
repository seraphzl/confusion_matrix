import torch

def get_cm(y_pred, y_true, n_classes):
    y_pred = torch.from_numpy(y_pred.flatten())
    y_true = torch.from_numpy(y_true.flatten())
    indices = n_classes * y_true + y_pred
    cm = torch.bincount(indices, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return cm
