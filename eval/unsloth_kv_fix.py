"""
Monkey-patch for Unsloth's KV cache bug in LlamaAttention_fast_forward_inference.

Bug: position_ids can have shape [seq_len] (e.g. [170]) during inference,
     causing cos/sin to be [1, 1, 170, 128] which can't broadcast with
     Qn shape [1, 28, 1, 128].

Fix: During single-token inference (Qn has seq_len=1), slice position_ids
     to only the last position.

Usage:
    import eval.unsloth_kv_fix  # just import it before inference
"""

import torch
import unsloth.models.llama as llama_module
import inspect

# Get the original function source to understand the context
_original_fn = llama_module.LlamaAttention_fast_forward_inference


def _patched_fast_forward_inference(
    self, hidden_states, past_key_value, position_ids, do_prefill=False,
    attention_mask=None, use_sliding_window=False, rotary_seq_len=None,
):
    """Patched version that fixes position_ids shape for KV cache inference."""
    # Fix: when generating one token at a time, position_ids should be a single value
    if position_ids is not None and position_ids.numel() > 1:
        # During KV cache inference, we only need the last position
        position_ids = position_ids[-1:]

    return _original_fn(
        self, hidden_states, past_key_value, position_ids,
        do_prefill=do_prefill, attention_mask=attention_mask,
        use_sliding_window=use_sliding_window, rotary_seq_len=rotary_seq_len,
    )


# Apply the patch
llama_module.LlamaAttention_fast_forward_inference = _patched_fast_forward_inference
print("[unsloth_kv_fix] Patched LlamaAttention_fast_forward_inference for KV cache support")
