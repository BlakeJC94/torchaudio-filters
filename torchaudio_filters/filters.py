from typing import List, Tuple

import torch
from scipy import signal
from torch import nn
from torchaudio.functional import filtfilt

from .pad import Pad


class _BaseFilter(nn.Module):
    def __init__(self, b: torch.Tensor, a: torch.Tensor, dtype=torch.float32):
        super().__init__()
        self.register_buffer("b", b.type(dtype))
        self.register_buffer("a", a.type(dtype))

    @staticmethod
    def get_filter_coeffs(
        order: int,
        cutoffs: List[float],
        btype: str,
        sample_rate: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, a = signal.butter(order, cutoffs, btype=btype, output="ba", fs=sample_rate)
        return torch.from_numpy(b), torch.from_numpy(a)

    def forward(self, x):
        scale = torch.max(torch.abs(x), dim=-1).values.unsqueeze(-1)
        x = x / scale

        padlen = 3 * max(len(self.a), len(self.b))
        x = Pad.odd_ext(x, padlen)

        x = filtfilt(x, self.a, self.b)

        x = x[..., padlen:-padlen]
        return x * scale


class LowPass(_BaseFilter):
    def __init__(self, cutoff: float, sample_rate: float, order=2):
        """Constructor for the LowPass class.

        Args:
            cutoff: Cutoff frequency in hertz.
            sample_rate: Input sampling rate in hertz.
            order: Degree of polynomial in filter (default = 2)
        """
        b, a = self.get_filter_coeffs(
            order,
            cutoff,
            "lowpass",
            sample_rate,
        )
        super().__init__(b, a)


class HighPass(_BaseFilter):
    def __init__(self, cutoff, sample_rate, order=2):
        b, a = self.get_filter_coeffs(
            order,
            cutoff,
            "highpass",
            sample_rate,
        )
        super().__init__(b, a)


class BandPass(_BaseFilter):
    def __init__(self, cutoff_low, cutoff_high, sample_rate, order=2):
        b, a = self.get_filter_coeffs(
            order,
            (cutoff_low, cutoff_high),
            "band",
            sample_rate,
        )
        super().__init__(b, a)


class Notch(_BaseFilter):
    def __init__(self, cutoff_low, cutoff_high, sample_rate, order=2):
        b, a = self.get_filter_coeffs(
            order,
            (cutoff_low, cutoff_high),
            "bandstop",
            sample_rate,
        )
        super().__init__(b, a)
