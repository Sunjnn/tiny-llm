import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        if top_k is not None and top_k > 0:
            k_index = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)
            other_index = k_index[:, top_k : ]
            logprobs[:, other_index] = -mx.inf
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            mask = cumsum < top_p
            mask[..., 0] = True
            logprobs[:, sorted_idx] = mx.where(mask, sorted_logprobs, -mx.inf)
        logprobs /= temp
        return mx.random.categorical(logprobs, axis=-1)

    return sample
