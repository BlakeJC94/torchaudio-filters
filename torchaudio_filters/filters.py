from typing import List, Tuple

import torch
from scipy import signal
from torch import nn
from torchaudio.functional import filtfilt, lfilter


# Probably not needed from scipy>=1.2.0
def _normalise_freq(frequency: float, sample_rate: float) -> float:
    """Normalise frequency to Nyquist Frequency (Fs/2)."""
    return 2.0 * frequency / sample_rate


def _filter_coeffs(
    btype: str,
    order: int,
    cutoffs: List[float],
    sample_rate: float,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cutoffs = [_normalise_freq(c, sample_rate) for c in cutoffs]
    b, a = signal.butter(order, cutoffs, btype=btype, output="ba", **kwargs)
    return torch.from_numpy(b), torch.from_numpy(a)


class _BaseFilter(nn.Module):
    def __init__(self, b, a, causal: bool = False):
        super().__init__()
        self.register_buffer("b", b)
        self.register_buffer("a", a)
        self.causal = causal

    @staticmethod
    def _pad_odd(x, n):
        left_end = x[..., 0:1]
        left_ext = x[..., 0:n].flip(dims=(-1,))

        right_end = x[..., -1:]
        right_ext = x[..., -(n + 2):-2].flip(dims=(-1,))

        return torch.cat(
            (
                2 * left_end - left_ext,
                x,
                2 * right_end - right_ext,
            ),
            dim=-1,
        )

    def forward(self, x):
        scale = torch.max(torch.abs(x), dim=-1).values.unsqueeze(-1)
        x = x / scale

        padlen = 3 * max(len(self.a), len(self.b))
        x = self._pad_odd(x, padlen)

        if self.causal:
            x = lfilter(x, self.a, self.b, clamp=False, batching=True)
        else:
            x = filtfilt(x, self.a, self.b)

        x = x[..., padlen:-padlen]
        return x * scale


class LowPass(_BaseFilter):
    def __init__(self, cutoff, sample_rate, order=2, causal=False):
        """Constructor for the LowPass class.

        Args:
            cutoff: Cutoff frequency in hertz.
            sample_rate: Input sampling rate in hertz.
            order: Degree of polynomial in filter (default = 2)
            causal: If True, apply filter in a causal manner.
        """
        b, a = _filter_coeffs(
            "lowpass",
            order,
            [cutoff],
            sample_rate,
        )
        super().__init__(b, a, causal)


class HighPass(_BaseFilter):
    def __init__(self, cutoff, sample_rate, order=2, causal=False):
        b, a = _filter_coeffs(
            "highpass",
            order,
            [cutoff],
            sample_rate,
        )
        super().__init__(b, a, causal)


class BandPass(_BaseFilter):
    def __init__(self, cutoff_low, cutoff_high, sample_rate, order=2, causal=False):
        b, a = _filter_coeffs(
            "band",
            order,
            [cutoff_low, cutoff_high],
            sample_rate,
        )
        super().__init__(b, a, causal)


class Notch(_BaseFilter):
    def __init__(self, cutoff_low, cutoff_high, sample_rate, order=2, causal=False):
        b, a = _filter_coeffs(
            "bandstop",
            order,
            [cutoff_low, cutoff_high],
            sample_rate,
        )
        super().__init__(b, a, causal)
