from typing import List, Tuple, Literal

import torch
from scipy import signal
from torch import nn
from torchaudio.functional import filtfilt

from .pad import Pad


class _BaseFilter(nn.Module):
    def __init__(
        self,
        order: int,
        cutoff: int | List[int],
        btype: Literal["lowpass", "highpass", "band", "bandstop"],
        sample_rate: float,
        dtype=torch.float32,
    ):
        super().__init__()
        b, a = self.get_filter_coeffs(
            order,
            cutoff,
            btype,
            sample_rate,
        )
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
    def __init__(
        self,
        cutoff: float,
        sample_rate: float,
        order: int = 2,
        dtype=torch.float32,
    ):
        """Constructor for the LowPass class.

        Args:
            cutoff: Cutoff frequency in hertz.
            sample_rate: Input sampling rate in hertz.
            order: Degree of polynomial in filter (default = 2).
            dtype: Torch dtype to use, must match input data dtype.
        """
        super().__init__(
            order,
            cutoff,
            "lowpass",
            sample_rate,
            dtype=dtype,
        )


class HighPass(_BaseFilter):
    def __init__(
        self,
        cutoff: float,
        sample_rate: float,
        order: int = 2,
        dtype=torch.float32,
    ):
        """Constructor for the HighPass class.

        Args:
            cutoff: Cutoff frequency in hertz.
            sample_rate: Input sampling rate in hertz.
            order: Degree of polynomial in filter (default = 2).
            dtype: Torch dtype to use, must match input data dtype.
        """
        super().__init__(
            order,
            cutoff,
            "highpass",
            sample_rate,
            dtype=dtype,
        )


class BandPass(_BaseFilter):
    def __init__(
        self,
        cutoff_low: float,
        cutoff_high: float,
        sample_rate: float,
        order: int = 2,
        dtype=torch.float32,
    ):
        """Constructor for the BandPass class.

        Args:
            cutoff_low: Lower cutoff frequency in hertz.
            cutoff_high: Upper cutoff frequency in hertz.
            sample_rate: Input sampling rate in hertz.
            order: Degree of polynomial in filter (default = 2).
            dtype: Torch dtype to use, must match input data dtype.
        """
        super().__init__(
            order,
            (cutoff_low, cutoff_high),
            "band",
            sample_rate,
            dtype=dtype,
        )


class Notch(_BaseFilter):
    def __init__(
        self,
        cutoff_low: float,
        cutoff_high: float,
        sample_rate: float,
        order: int = 2,
        dtype=torch.float32,
    ):
        """Constructor for the Notch class.

        Args:
            cutoff_low: Lower cutoff frequency in hertz.
            cutoff_high: Upper cutoff frequency in hertz.
            sample_rate: Input sampling rate in hertz.
            order: Degree of polynomial in filter (default = 2).
            dtype: Torch dtype to use, must match input data dtype.
        """
        super().__init__(
            order,
            (cutoff_low, cutoff_high),
            "bandstop",
            sample_rate,
            dtype=dtype,
        )
