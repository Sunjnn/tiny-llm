import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable
from tiny_llm.kv_cache import TinyKvFullCache


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y)
        logits = output_logits[:, -1, :]
        return sampler(logits)[0]

    prompt_tokens = tokenizer.encode(prompt)
    response_tokens = []
    while True:
        token = _step(model, mx.array(prompt_tokens).reshape(1, -1))
        prompt_tokens.append(token)
        response_tokens.append(token)

        if token == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(response_tokens)
    return response


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        output_logits = model(y, offset, kv_cache)
        logits = output_logits[:, -1, :]
        return mx.argmax(logits, axis=-1)[0]

    cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    offset = 0

    prompt_tokens = tokenizer.encode(prompt)
    response_tokens = []
    while True:
        token = _step(model, mx.array(prompt_tokens).reshape(1, -1), offset, cache)
        offset += len(prompt_tokens)
        prompt_tokens = [token]
        response_tokens.append(token)

        if token == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(response_tokens)
    return response
