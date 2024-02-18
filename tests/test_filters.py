from typing import Optional, Iterable, Tuple

import torch
import numpy as np
import pytest

from torchaudio_filters import LowPass, HighPass, BandPass, Notch


def gen_data(
    sample_rate: float,
    sample_length: float,
    channels: int = 21,
    frequencies: Optional[Iterable[float]] = None,
    start_time: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random or sinusoidal data for testing purposes."""
    times = np.arange(0, sample_length * 1000, 1 / sample_rate * 1000)
    if frequencies:
        x_channel = 0
        for frequency in frequencies:
            x_channel += np.sin(2 * np.pi * frequency * times / 1000)
        x = np.tile(x_channel, (channels, 1)).T
    else:
        x = np.random.randn(len(times), channels)

    if start_time:
        times += start_time

    return x, times


class TestFrequencyFilters:
    """Tests for the frequency domain filters LowPass, HighPass, and Notch."""

    in_rate = 256
    in_len = 256
    frequencies = [5, 20, 100]  # Frequencies present in input data

    @pytest.fixture
    def data(self):
        return gen_data(self.in_rate, self.in_len, frequencies=self.frequencies)

    def run_test(self, transform, data, low, high, band=False):
        x_in, times = data
        x_out = transform(torch.from_numpy(x_in).permute(1, 0)).permute(1, 0).numpy()

        # Get FFT of data before and after transformation
        fft_in = np.abs(np.fft.rfft(x_in[:, 0]))
        fft_out = np.abs(np.fft.rfft(x_out[:, 0]))
        freq = np.fft.rfftfreq(len(x_out), 1 / self.in_rate)

        # Check each input frequency for attenuation
        for frequency in self.frequencies:
            freq_ind = np.argmin(np.abs(freq - frequency))
            if low < frequency < high:
                assert fft_out[freq_ind] / fft_in[freq_ind] < 0.2
            else:
                assert fft_out[freq_ind] / fft_in[freq_ind] > 0.8

        assert x_out.shape[0] == self.in_len * self.in_rate

    @pytest.mark.parametrize("cutoff", [10, 50, 120])
    def test_lowpass(self, cutoff, data):
        transform = LowPass(cutoff, self.in_rate)
        self.run_test(transform, data, cutoff, 1e6)

    @pytest.mark.parametrize("cutoff", [10, 50, 120])
    def test_highpass(self, cutoff, data):
        transform = HighPass(cutoff, self.in_rate)
        self.run_test(transform, data, 0, cutoff)

    @pytest.mark.parametrize(
        "cutoff_low, cutoff_high",
        [(10, 15), (15, 25), (120, 125)],
    )
    def test_notch(self, cutoff_low, cutoff_high, data):
        transform = Notch(cutoff_low, cutoff_high, self.in_rate)
        self.run_test(transform, data, cutoff_low, cutoff_high)

    # TODO RFC
    # @pytest.mark.parametrize(
    #     "cutoff_low, cutoff_high",
    #     [(10, 15), (15, 25), (120, 125)],
    # )
    # def test_bandpass(self, cutoff_low, cutoff_high, data):
    #     transform = BandPass(cutoff_low, cutoff_high, self.in_rate)
    #     self.run_test(transform, data, cutoff_low, cutoff_high, band=True)
