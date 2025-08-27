import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1.0 / (key.shape[-1] ** 0.5)

    result = mx.matmul(query, mx.swapaxes(key, -2, -1))
    result = result * scale
    if mask is not None:
        result = mx.add(result, mask)
    result = softmax(result, axis=-1)
    result = mx.matmul(result, value)
    return result


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.E = hidden_size
        self.H = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N = query.shape[0]
        L = query.shape[1]
        E = query.shape[2]
        D = self.wq.shape[1] // self.H

        scale = mx.rsqrt(self.E // self.H)

        mq = linear(query.reshape(-1, E), self.wq)
        mq = mq.reshape(N, L, self.H, D).swapaxes(1, 2)
        mk = linear(key.reshape(-1, E), self.wk)
        mk = mk.reshape(N, L, self.H, D).swapaxes(1, 2)
        mv = linear(value.reshape(-1, E), self.wv)
        mv = mv.reshape(N, L, self.H, D).swapaxes(1, 2)

        o = scaled_dot_product_attention_simple(mq, mk, mv, mask=mask, scale=scale)
        o = o.swapaxes(1, 2)

        result = linear(o.reshape(-1, self.H * D), self.wo)
        result = result.reshape(N, L, E)
        return result


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
