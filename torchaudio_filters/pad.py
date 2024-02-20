from typing import Literal

import torch
from torch import nn


PadMode = Literal["odd", "even", "const"]


class Pad(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        padlen: int,
        mode: PadMode = "odd",
        val: float = 0.0,
    ):
        super().__init__()
        self.module = module
        self.padlen = int(padlen)
        self.mode = mode
        self.val = val

    @staticmethod
    def odd_ext(x, n):
        left_end = x[..., :1]
        left_ext = x[..., 1 : n + 1].flip(dims=(-1,))

        right_end = x[..., -1:]
        right_ext = x[..., -(n + 1) : -1].flip(dims=(-1,))

        return torch.cat(
            (
                2 * left_end - left_ext,
                x,
                2 * right_end - right_ext,
            ),
            dim=-1,
        )

    @staticmethod
    def even_ext(x, n):
        left_ext = x[..., 1 : n + 1].flip(dims=(-1,))
        right_ext = x[..., -(n + 1) : -1].flip(dims=(-1,))
        return torch.cat(
            (
                left_ext,
                x,
                right_ext,
            ),
            dim=-1,
        )

    @staticmethod
    def _pad_const(x, n, val=0):
        ext = val * torch.ones_like(x)[..., :n]
        return torch.cat(
            (
                ext,
                x,
                ext,
            ),
            dim=-1,
        )

    def forward(self, x):
        if self.mode == "odd":
            x = self.odd_ext(x, self.padlen)
        elif self.mode == "even":
            x = self.even_ext(x, self.padlen)
        else:
            x = self.const_ext(x, self.padlen, self.val)
        x = self.module(x)
        return x[..., self.padlen : -self.padlen]
