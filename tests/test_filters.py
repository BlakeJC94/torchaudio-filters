from typing import Optional, Iterable, Tuple, List

import torch
import numpy as np
import pytest
from scipy import signal

from torchaudio_filters import LowPass, HighPass, BandPass, Notch


import plotly.express as px


@pytest.fixture
def sample_rate() -> float:
    return 256.0


@pytest.fixture
def sample_length() -> float:  # s
    return 1.0


@pytest.fixture
def n_channels() -> int:
    return 21


@pytest.fixture
def batch_size() -> int:
    return 0


@pytest.fixture
def frequencies() -> List[float]:
    return [5.0, 20.0, 100.0]


@pytest.fixture
def data(
    sample_rate, sample_length, n_channels, batch_size, frequencies
) -> torch.Tensor:
    out = []

    start_time = 0
    for _ in range(max(batch_size, 1)):
        times = start_time + np.arange(0, sample_length, 1 / sample_rate)

        x = np.zeros((n_channels, len(times)))
        for frequency in frequencies:
            sig = np.sin(2 * np.pi * frequency * times)
            x = x + np.expand_dims(sig, axis=0)  # Add signal to each channel

        out.append(x)
        start_time += sample_length

    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def calculate_relative_power_diffs(sample_rate, x_in, x_out):
    x_in = x_in.numpy()
    x_out = x_out.numpy()

    # Collapse batch
    if x_in.ndim == 3:
        x_in = x_in.sum(axis=0)
        x_out = x_out.sum(axis=0)

    # Collapse channels
    if x_in.ndim == 2:
        x_in = x_in.sum(axis=0)
        x_out = x_out.sum(axis=0)

    # Return powers and freqs
    fft_in = np.abs(np.fft.rfft(x_in))
    fft_out = np.abs(np.fft.rfft(x_out))
    fft_diff = fft_out / fft_in
    freq = np.fft.rfftfreq(len(x_out), 1 / sample_rate)

    return fft_diff, freq


def calculate_expected_output(x_in, transform):
    return torch.from_numpy(
        signal.filtfilt(
            transform.b,
            transform.a,
            x_in,
            axis=-1,
            padtype='odd',
        ).copy()
    )

import pandas as pd
@pytest.mark.parametrize("cutoff", [10, 50, 120])
def test_low_pass(cutoff, sample_rate, data, frequencies):
    ## Assemble
    transform = LowPass(cutoff, sample_rate)

    ## Act
    out = transform(data)

    ## Assert
    assert out.shape == data.shape


    expected_out = calculate_expected_output(data, transform)
    foo = pd.DataFrame(
        dict(
            a=data[0, 0, :],
            b=out[0, 0, :],
            c=expected_out[0, 0, :],
        )
    )
    breakpoint()

    fft_diff, fft_freqs = calculate_relative_power_diffs(sample_rate, data, out)
    for freq in frequencies:
        freq_ind = np.argmin(np.abs(fft_freqs - freq))
        if cutoff < freq < 1e6:
            assert fft_diff[freq_ind] < 0.2
        else:
            assert fft_diff[freq_ind] > 0.8


# Test lowpass
# TODO Create input data across 3 frequencies with batch size and channels
# TODO Create module
# TODO Compare difference in power for each frequency band
# TODO Compare output to

# @pytest.fixture
# def gen_data(
#     sample_rate: float,  # Hertz
#     sample_length: float,  # seconds
#     channels: int = 21,
#     batch_size: int = 0,
#     frequencies: Optional[Iterable[float]] = None,
#     sigma: float = 1.,
#     mu: float = 0.,
# ) -> Tuple[torch.Tensor]:
#     """Generate random or sinusoidal data for testing purposes."""
#     frequencies = frequencies or []

#     out = []
#     start_time = 0
#     for _ in range(max(batch_size, 1)):
#         times = start_time + np.arange(0, sample_length, 1 / sample_rate)

#         x = mu + sigma * np.random.randn(channels, len(times))
#         for frequency in frequencies:
#             sig = np.sin(2 * np.pi * frequency * times)
#             x = x + np.expand_dims(sig, axis=0)  # Add signal to each channel

#         out.append(x)
#         start_time += sample_length

#     out = out[0] if batch_size == 0 else np.stack(out, axis=0)
#     return torch.from_numpy(out)


# class TestFrequencyFilters:
#     """Tests for the frequency domain filters LowPass, HighPass, and Notch."""

#     in_rate = 256
#     in_len = 256
#     frequencies = [5, 20, 100]  # Frequencies present in input data

#     @pytest.fixture
#     def data(self):
#         return gen_data(self.in_rate, self.in_len, frequencies=self.frequencies)

#     def run_test(self, transform, data, low, high, band=False):
#         x_in, times = data
#         x_out = transform(torch.from_numpy(x_in).permute(1, 0)).permute(1, 0).numpy()

#         # Get FFT of data before and after transformation
#         fft_in = np.abs(np.fft.rfft(x_in[:, 0]))
#         fft_out = np.abs(np.fft.rfft(x_out[:, 0]))
#         freq = np.fft.rfftfreq(len(x_out), 1 / self.in_rate)

#         # Check each input frequency for attenuation
#         for frequency in self.frequencies:
#             freq_ind = np.argmin(np.abs(freq - frequency))
#             if low < frequency < high:
#                 assert fft_out[freq_ind] / fft_in[freq_ind] < 0.2
#             else:
#                 assert fft_out[freq_ind] / fft_in[freq_ind] > 0.8

#         assert x_out.shape[0] == self.in_len * self.in_rate

#     @pytest.mark.parametrize("cutoff", [10, 50, 120])
#     def test_lowpass(self, cutoff, data):
#         transform = LowPass(cutoff, self.in_rate)
#         self.run_test(transform, data, cutoff, 1e6)

#     @pytest.mark.parametrize("cutoff", [10, 50, 120])
#     def test_highpass(self, cutoff, data):
#         transform = HighPass(cutoff, self.in_rate)
#         self.run_test(transform, data, 0, cutoff)

#     @pytest.mark.parametrize(
#         "cutoff_low, cutoff_high",
#         [(10, 15), (15, 25), (120, 125)],
#     )
#     def test_notch(self, cutoff_low, cutoff_high, data):
#         transform = Notch(cutoff_low, cutoff_high, self.in_rate)
#         self.run_test(transform, data, cutoff_low, cutoff_high)

#     # TODO RFC
#     # @pytest.mark.parametrize(
#     #     "cutoff_low, cutoff_high",
#     #     [(10, 15), (15, 25), (120, 125)],
#     # )
#     # def test_bandpass(self, cutoff_low, cutoff_high, data):
#     #     transform = BandPass(cutoff_low, cutoff_high, self.in_rate)
#     #     self.run_test(transform, data, cutoff_low, cutoff_high, band=True)

# # from scipy import signal

# # def test_low_pass_filter():
# #     in_rate = 256
# #     cutoff = 10

# #     # transform = LowPass(cutoff, in_rate)

# #     order = 2
# #     cutoffs = [2.0 * cutoff / in_rate, 2.0 * (cutoff+10) / in_rate]
# #     btype = "band"
# #     b, a = signal.butter(order, cutoffs, btype=btype, output="ba")
# #     assert True
