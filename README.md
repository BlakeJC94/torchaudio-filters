# torchaudio-filters

High-pass and low-pass filters implemented as modules with torchaudio.

This small package offers a simple API to implement basic butterworth filters in PyTorch modules.
Aims to maintain consistency with the PyTorch API (e.g. behaves similarly to
`torchaudio.transforms.Spectrogram`) and uses `torchaudio.functional.filtfilt` under the hood.

```python
>>> from torch import nn
>>> from torchaudio_filters import LowPass, Pad
>>> sample_rate = 128  # Hz
>>> sample_secs = 10
>>> x = torch.rand(32, 21, sample_rate * sample_secs)  # batch_idx, channel_idx, timestep_idx
# Add pre-processors to your models using Sequential
# * Pad will pad input and unpad output
# * LowPass will filter to the frequencies below a given cutoff
>>> model = nn.Sequential(
...     Pad(
...         module=LowPass(sample_rate, 70)  # Low-pass filter
...         padlen=32,
...     )
...     encoder,
...     decoder,
...     ...,
... )
>>> y_hat = model(x)
```

## Installation

Requires a relatively recent `torchaudio` to be installed (anything after `2.0.0` should be more
than okay), and also needs `scipy`.

To install the latest stable version from `PyPi`:
```bash
pip install torchaudio-filters
```

## Usage

This package includes the `torch.nn.module` objects `LowPass`, `HighPass`, `BandPass`, `Notch`, and
`Pad`. These all assume the final dimension represents time, and apply Butterworth filters in their
`forward` methods.

Let's generate a simple example on a sinusoidal 256 Hz signal with frequencies 5, 10, and 20 Hz:
```python
>>> sample_rate = 256  # Hz
>>> sample_secs = 1.5  # s
>>> t = torch.arange(0, sample_secs, 1 / sample_rate)
>>> freqs = [5, 10, 20]
>>> x = sum(torch.sin(2 * torch.pi * f * t) for f in freqs)
```

We can design low-pass and high-pass filters and call them as any other module:
```python
>>> lp = LowPass(15, sample_rate)
>>> x_lp = lp(x)
>>> hp = HighPass(15, sample_rate)
>>> x_hp = hp(x)
>>> notch = Notch(7, 12, sample_rate)
>>> x_notch = notch(x)
>>> bp = BandPass(7, 12, sample_rate)
>>> x_bp = bp(x)
```

Plotting these results:
```python
>>> fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9.6, 6))
>>> ax1.plot(t, x)
>>> ax1.set_title('Input signal')
>>> ax2.plot(t, x_lp)
>>> ax2.plot(t, x_hp)
>>> ax2.set_title('LowPass / HighPass')
>>> ax3.plot(t, x_notch)
>>> ax3.plot(t, x_bp)
>>> ax3.set_title('Notch / BandPass')
```
![fig00](https://github.com/BlakeJC94/torchaudio-filters/assets/16640474/189b77c4-7e65-44c3-8928-519874d918ae)

To decrease the edge effect, a wrapper module `Pad` is also provided
```python
>>> filt = torch.nn.Sequential(hp, notch)
>>> filt_pad = Pad(
...     filt,
...     padlen=128,
... )
>>> x_f = filt(x)
>>> x_fpad = filt_pad(x)
>>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
>>> ax1.plot(t, x)
>>> ax1.set_title('Input signal')
>>> ax2.plot(t, x_f)
>>> ax2.plot(t, x_fpad)
>>> ax2.set_title('Without pad / With pad')
>>> plt.show()
```
![fig01](https://github.com/BlakeJC94/torchaudio-filters/assets/16640474/51e4e7ba-927a-4d83-98cc-5f1eacf7be9d)


## Contributing
Pull requests are most welcome! If there's any questions, open an issue so we can discuss further.

Dependencies and environments are managed with `[poetry](https://python-poetry.org)`. To get started developing for this package:
```
$ git clone https://github.com/BlakeJC94/torchaudio_filters
$ cd torchaudio_filters
$ poetry install
$ pytest
```

Branch `main` should be stable, all the latest changes will go onto the `dev` branch before being released on `main`.

* Code is tested with `[pytest](https://docs.pytest.org)`
* Code is styled using `[black](https://black.readthedocs.io)`
* Code is linted with `[ruff](https://beta.ruff.rs)`
