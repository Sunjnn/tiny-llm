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

        rope = RoPE(self.E // self.H, L, self.theta)
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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = linear(x, self.w_gate)
        silu_gate = silu(gate)

        up = linear(x, self.w_up)
        y = silu_gate * up
        y = linear(y, self.w_down)
        return y


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
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.mha = Qwen2MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads, wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_layernorm = self.input_layernorm(x)
        mha = self.mha(input_layernorm, mask)
        y = x + mha
        post_attention_layernorm = self.post_attention_layernorm(y)
        mlp = self.mlp(post_attention_layernorm)
        o = y + mlp
        return o


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        hidden_size = mlx_model.args.hidden_size
        rms_norm_eps = mlx_model.args.rms_norm_eps
        precision = mx.float16

        self.embedding = Embedding(mlx_model.args.vocab_size, hidden_size, dequantize_linear(mlx_model.model.embed_tokens).astype(precision))

        self.layers_inner = []
        for i in range(mlx_model.args.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(precision)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(precision)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(precision)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(precision)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(precision)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(precision)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(precision)
            bq = mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision)
            bk = mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision)
            bv = mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision)

            w_input_layernorm = mlx_model.model.layers[i].input_layernorm.weight.astype(precision)
            w_post_attention_layernorm = mlx_model.model.layers[i].post_attention_layernorm.weight.astype(precision)

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=bq,
                bk=bk,
                bv=bv,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=w_input_layernorm,
                w_post_attention_layernorm=w_post_attention_layernorm,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta
            )
            self.layers_inner.append(layer)

        self.norm = RMSNorm(
            hidden_size,
            mlx_model.model.norm.weight.astype(precision),
            rms_norm_eps
        )

        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in self.layers_inner:
            h = layer(h, mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
