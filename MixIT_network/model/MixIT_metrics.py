import torch
import torch.nn as nn
from torchmetrics import Metric

def SiSNR(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8):
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    projection = torch.sum(estimate * reference, dim=-1, keepdim=True) * reference / (
        torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    )

    noise = estimate - projection

    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)

class SiSNR_Metric(Metric):

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("total_SiSNR", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_hat: torch.Tensor, y: torch.Tensor):
        sisnr_batch = SiSNR(y, y_hat, self.eps)

        self.total_SiSNR += sisnr_batch.sum()
        self.total_count += y.size(0)

    def compute(self):
        if self.total_count == 0:
            return torch.tensor(float('nan'))
        return self.total_SiSNR / self.total_count

    def reset(self):
        super().reset()

class SiSNRi_Metric(Metric):

    def __init__(self, num_sources: int, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.num_sources = num_sources

        self.add_state("total_sisnri_per_source", default=torch.zeros(num_sources), dist_reduce_fx="sum")
        self.add_state("total_samples_per_source", default=torch.zeros(num_sources, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, y_hat: torch.Tensor, y: torch.Tensor, input_mixture: torch.Tensor):
        B, S, T = y_hat.shape
        if S != self.num_sources:
            raise ValueError(f"Mismatch between num_sources in init ({self.num_sources}) and y_hat.shape[1] ({S})")
        if y.shape != y_hat.shape:
            raise ValueError(f"Shape mismatch between y_hat {y_hat.shape} and y {y.shape}")
        if input_mixture.ndim != 2 or input_mixture.shape[0] != B or input_mixture.shape[1] != T:
            raise ValueError(f"Shape mismatch for input_mixture. Expected [{B}, {T}], got {input_mixture.shape}")

        for i in range(S):
            target_i = y[:, i, :]
            estimate_i = y_hat[:, i, :]

            sisnr_est = SiSNR(target_i, estimate_i, self.eps)
            sisnr_mix = SiSNR(target_i, input_mixture, self.eps)
            sisnr_i = sisnr_est - sisnr_mix

            self.total_sisnri_per_source[i] += sisnr_i.sum()
            self.total_samples_per_source[i] += B

    def compute(self, per_source: bool = False):
        avg_sisnri = self.total_sisnri_per_source / (self.total_samples_per_source.float() + 1e-10)
        if per_source:
            return avg_sisnri
        else:
            valid_sources_mask = self.total_samples_per_source > 0
            if valid_sources_mask.any():
                return avg_sisnri[valid_sources_mask].mean()
            else:
                return torch.tensor(float('nan'))

    def reset(self):
        super().reset()