import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        i = mx.arange(dims / 2)
        w = mx.power(base, -2 * i / dims)
        pos = mx.arange(seq_len)
        theta = mx.outer(pos, w)

        assert theta.shape == (seq_len, dims // 2)
        assert theta[0, 0] == 0
        assert theta[0, 1] == 0

        self.cos_freqs = mx.cos(theta)
        self.sin_freqs = mx.sin(theta)

        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape

        if offset is not None:
            if isinstance(offset, slice):
                cos_freq = self.cos_freqs[offset, :]
                sin_freq = self.sin_freqs[offset, :]
            elif isinstance(offset, list):
                offset = mx.array([list(range(i.start, i.stop)) for i in offset])
                cos_freq = self.cos_freqs[offset, :]
                sin_freq = self.sin_freqs[offset, :]
            else:
                cos_freq = self.cos_freqs[:L, :]
                sin_freq = self.sin_freqs[:L, :]
        else:
            cos_freq = self.cos_freqs[:L, :]
            sin_freq = self.sin_freqs[:L, :]

        cos_freq = cos_freq.reshape(1, L, 1, D // 2)
        sin_freq = sin_freq.reshape(1, L, 1, D // 2)

        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)
            x0 = x[..., 0]
            x1 = x[..., 1]
        else:
            x0 = x[..., : D // 2]
            x1 = x[..., D // 2 :]

        y0 = mx.multiply(x0, cos_freq) - mx.multiply(x1, sin_freq)
        y1 = mx.multiply(x0, sin_freq) + mx.multiply(x1, cos_freq)

        if self.traditional:
            y = mx.stack([y0, y1], axis=-1).reshape(N, L, H, D)
        else:
            y = mx.concat([y0, y1], axis=-1).reshape(N, L, H, D)

        return y
