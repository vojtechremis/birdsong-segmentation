import torch
import torch.nn as nn
from torchmetrics import Metric
import torch.nn.functional as F
import warnings


class SegmentsLoss(nn.Module):
    def __init__(self, weight_segments=1.0, weight_motifs=1.0):
        super().__init__()
        self.weight_segments = weight_segments
        self.weight_motifs = weight_motifs
        self.mse_loss = nn.MSELoss()  # Mean Squared Error Loss

    def forward(self, y_hat, y_true):
        # Extracting segments and motifs
        pred_segments, pred_motifs = y_hat[:, 0], y_hat[:, 1]
        target_segments, target_motifs = y_true[0], y_true[1]

        # MSE calculation for each entry
        loss_segments = self.mse_loss(pred_segments, target_segments)
        loss_motifs = self.mse_loss(pred_motifs, target_motifs)

        # Weighting
        total_loss = (self.weight_segments * loss_segments) + (self.weight_motifs * loss_motifs)

        return total_loss


class CustomAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # State variables initialization
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def __call__(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        self.update(y_hat, y_true)

    def _input_format(self, y_hat, y_true):
        if y_hat.device == self.device and y_true.device == self.device:
            return y_hat, y_true
        else:
            warnings.warn(f'Tensors y_hat or y_true is not on correct device {self.device}.')
            return y_hat.to(self.device), y_true.to(self.device)

    def update(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Update
        """

        y_hat, y_true = self._input_format(y_hat, y_true)
        if y_hat.shape != y_true.shape:
            raise ValueError("Predicted and True values must have the same shape!")

        self.correct += torch.sum(y_hat == y_true)
        self.total += y_true.numel()

    def compute(self):
        """
        Computing
        """
        return self.correct.float() / self.total


class CustomMetric(Metric):
    """
    Computes precision, recall and F1 score when there is no fixed num_of_classes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define states
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, y_hat, y_true):
        if y_hat.device == self.device and y_true.device == self.device:
            return y_hat, y_true
        else:
            warnings.warn(f'Tensors y_hat or y_true is not on correct device {self.device}.')
            return y_hat.to(self.device), y_true.to(self.device)

    def update(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        """
        Update states based on predictions and targets.
        """

        y_hat, y_true = self._input_format(y_hat, y_true)
        if y_hat.shape != y_true.shape:
            raise ValueError("Predicted and True values must have the same shape!")

        # Ensure y_hat and y_true are flattened
        y_hat = y_hat.flatten()
        y_true = y_true.flatten()

        # Find the unique classes in y_hat and y_true
        unique_classes = torch.unique(torch.cat([y_hat, y_true]))

        # Loop over each class and calculate TP, FP, FN
        for cls in unique_classes:
            # True Positives: correct predictions of the class
            tp = torch.sum((y_hat == cls) & (y_true == cls))

            # False Positives: incorrect predictions of the class
            fp = torch.sum((y_hat == cls) & (y_true != cls))

            # False Negatives: missed predictions for the class
            fn = torch.sum((y_hat != cls) & (y_true == cls))

            # Update states
            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn

    def compute(self):
        """
        Compute precision.
        """

        precision = self.true_positives.float() / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives.float() / (self.true_positives + self.false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


def pad_to_largest(img_list):
    """
    Padování všech obrázků v batchi na velikost největšího obrázku.

    Args:
        batch: Seznam tensorů [C, H, W], kde každý tensor je obrázek.

    Returns:
        Padded batch: Tensor [B, C, max_H, max_W].
    """
    # Najít maximální výšku a šířku v batchi
    max_height = max(img.shape[1] for img in img_list)
    max_width = max(img.shape[2] for img in img_list)

    # Padovat každý obrázek na velikost [C, max_height, max_width]
    padded_batch = []
    for img in img_list:
        _, h, w = img.shape
        # Spočítat rozdíl ve výšce a šířce
        pad_h = max_height - h
        pad_w = max_width - w
        # Použít funkci pad (pad vlevo/vpravo/nahoře/dole)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        padded_batch.append(padded_img)

    return torch.stack(padded_batch)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: Pokud se mění počet kanálů nebo velikost, transformuj shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out