import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.E = hidden_size
        self.H = num_heads
        self.H_kv = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B = x.shape[0]
        L = x.shape[1]

        q = linear(x, self.wq, self.bq).reshape(B, L, self.H, -1)
        k = linear(x, self.wk, self.bk).reshape(B, L, self.H_kv, -1)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.H_kv, -1).swapaxes(1, 2)

        rope = RoPE(self.E // self.H, L)
        q = rope(q, slice(0, L)).swapaxes(1, 2)
        k = rope(k, slice(0, L)).swapaxes(1, 2)

        scale = mx.rsqrt(self.E // self.H)
        out = scaled_dot_product_attention_grouped(q, k, v, mask=mask, scale=scale).swapaxes(1, 2).reshape(B, L, -1)

        out = linear(out, self.wo)
        return out


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
