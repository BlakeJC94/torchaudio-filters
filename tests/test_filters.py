from typing import Optional, Iterable, Tuple, List

import torch
import numpy as np
import pytest
from scipy import signal

from torchaudio_filters import LowPass, HighPass, BandPass, Notch


@pytest.fixture
def sample_rate() -> float:  # Hz
    return 256.0


@pytest.fixture
def sample_length() -> float:  # s
    return 3.0


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
            x_in.numpy(),
            axis=-1,
        ).copy()
    )


@pytest.mark.parametrize("cutoff", [10, 50, 120])
def test_low_pass(cutoff, sample_rate, data, frequencies):
    """Test that LowPass filters out higher frequencies and matches scipy implementation after
    trimming edges.
    """
    ## Assemble
    transform = LowPass(cutoff, sample_rate)

    ## Act
    out = transform(data)

    ## Assert
    # Check that output shape matches the input shape
    assert out.shape == data.shape

    # Check that non-edge values match up with the scipy implementation
    expected_out = calculate_expected_output(data, transform)
    outer_padlen = int(sample_rate / 4)
    center_slice = slice(outer_padlen, -outer_padlen)
    diff = torch.abs(out[..., center_slice] - expected_out[..., center_slice])
    assert diff.median() < 1e-5
    assert diff.max() < 1e-2

    # Check that correct frequencies were attenuated
    fft_diff, fft_freqs = calculate_relative_power_diffs(sample_rate, data, out)
    for freq in frequencies:
        freq_ind = np.argmin(np.abs(fft_freqs - freq))
        if cutoff < freq < 1e6:
            assert fft_diff[freq_ind] < 0.2
        else:
            assert fft_diff[freq_ind] > 0.8


@pytest.mark.parametrize("cutoff", [10, 50, 120])
def test_high_pass(cutoff, sample_rate, data, frequencies):
    """Test that HighPass filters out lower frequencies and matches scipy implementation after
    trimming edges.
    """
    ## Assemble
    transform = HighPass(cutoff, sample_rate)

    ## Act
    out = transform(data)

    ## Assert
    # Check that output shape matches the input shape
    assert out.shape == data.shape

    # Check that non-edge values match up with the scipy implementation
    expected_out = calculate_expected_output(data, transform)
    outer_padlen = int(sample_rate / 4)
    center_slice = slice(outer_padlen, -outer_padlen)
    diff = torch.abs(out[..., center_slice] - expected_out[..., center_slice])
    assert diff.median() < 1e-5
    assert diff.max() < 1e-2

    # Check that correct frequencies were attenuated
    fft_diff, fft_freqs = calculate_relative_power_diffs(sample_rate, data, out)
    for freq in frequencies:
        freq_ind = np.argmin(np.abs(fft_freqs - freq))
        if 0 < freq < cutoff:
            assert fft_diff[freq_ind] < 0.2
        else:
            assert fft_diff[freq_ind] > 0.8


@pytest.mark.parametrize(
    "cutoff_low, cutoff_high",
    [(10, 15), (15, 25), (120, 125)],
)
def test_band_pass(cutoff_low, cutoff_high, sample_rate, data, frequencies):
    """Test that BandPass filters out lower and higher frequencies and matches scipy implementation
    after trimming edges.
    """
    ## Assemble
    transform = BandPass(cutoff_low, cutoff_high, sample_rate)

    ## Act
    out = transform(data)

    ## Assert
    # Check that output shape matches the input shape
    assert out.shape == data.shape

    # Check that non-edge values match up with the scipy implementation
    expected_out = calculate_expected_output(data, transform)
    outer_padlen = int(sample_rate / 4)
    center_slice = slice(outer_padlen, -outer_padlen)
    diff = torch.abs(out[..., center_slice] - expected_out[..., center_slice])
    assert diff.median() < 1e-5
    assert diff.max() < 1e-2

    # Check that correct frequencies were attenuated
    fft_diff, fft_freqs = calculate_relative_power_diffs(sample_rate, data, out)
    for freq in frequencies:
        freq_ind = np.argmin(np.abs(fft_freqs - freq))
        if 0 < freq < cutoff_low or cutoff_high < freq < 1e6:
            assert fft_diff[freq_ind] < 0.2
        else:
            assert fft_diff[freq_ind] > 0.8


@pytest.mark.parametrize(
    "cutoff_low, cutoff_high",
    [(10, 15), (15, 25), (120, 125)],
)
def test_notch(cutoff_low, cutoff_high, sample_rate, data, frequencies):
    """Test that Notch filters out band frequencies and matches scipy implementation after trimming
    edges.
    """
    ## Assemble
    transform = Notch(cutoff_low, cutoff_high, sample_rate)

    ## Act
    out = transform(data)

    ## Assert
    # Check that output shape matches the input shape
    assert out.shape == data.shape

    # Check that non-edge values match up with the scipy implementation
    expected_out = calculate_expected_output(data, transform)
    outer_padlen = int(sample_rate / 4)
    center_slice = slice(outer_padlen, -outer_padlen)
    diff = torch.abs(out[..., center_slice] - expected_out[..., center_slice])
    assert diff.median() < 1e-5
    assert diff.max() < 1e-2

    # Check that correct frequencies were attenuated
    fft_diff, fft_freqs = calculate_relative_power_diffs(sample_rate, data, out)
    for freq in frequencies:
        freq_ind = np.argmin(np.abs(fft_freqs - freq))
        if cutoff_low < freq < cutoff_high:
            assert fft_diff[freq_ind] < 0.2
        else:
            assert fft_diff[freq_ind] > 0.8
