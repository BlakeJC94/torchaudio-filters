# torchaudio-filters

High-pass and low-pass filters implemented as modules with torchaudio.

This small package offers a simple API to implement basic butterworth filters in PyTorch modules.
Aims to maintain consistency with the PyTorch API (e.g. behaves similarly to
`torchaudio.transforms.Spectrogram`) and uses `torchaudio.functional.filtfilt` under the hood.

## TODO

- [x] Setup project
- [x] Write core code
- [x] Write basic tests
- [ ] Manually verify
- [ ] Write up example scripts based on scipy docs
- [ ] Add short examples to README
- [ ] Implement equivalence test with scipy
- [ ] Implement tests
- [ ] Upload to pypi
