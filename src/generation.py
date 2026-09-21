"""
src/generation.py
─────────────────
Autoregressive text generation with sampling techniques:
  - Greedy decoding (argmax)
  - Temperature scaling
  - Top-K filtering
  - Top-P (Nucleus) sampling

─────────────────────────────────────────────────────────────────────────────
HOW AUTOREGRESSIVE GENERATION WORKS
─────────────────────────────────────────────────────────────────────────────
1. Start with a prompt string: e.g. "Artificial intelligence is"
2. Tokenize prompt into IDs: [120, 482, 31]
3. Input IDs to GPT model: shape (1, T)
4. Forward pass produces logits: shape (1, T, vocab_size)
5. Pluck logits for the LAST position only: logits[:, -1, :] -> shape (1, vocab_size)
6. Apply Temperature: logits = logits / temperature
7. Apply Top-K filtering: keep only the K highest-probability tokens
8. Apply Top-P filtering: keep the smallest set of tokens whose cumulative prob >= p
9. Sample next token from probability distribution: multinomial(probs)
10. Append sampled token to sequence and repeat until max_new_tokens is reached.
"""

from typing import List, Optional
import torch
import torch.nn.functional as F

from src.model.gpt import GPT
from src.tokenizer import BPETokenizer


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate new tokens autoregressively given a conditioning context tensor.

    Parameters
    ----------
    model : GPT
        Trained GPT model in eval mode.
    idx : torch.Tensor
        Conditioning prompt tokens of shape (B, T), dtype torch.long.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    temperature : float
        Sampling temperature (1.0 = standard, <1.0 = more deterministic/confident, >1.0 = more creative).
    top_k : Optional[int]
        If set, only sample from the top K most likely tokens.
    top_p : Optional[float]
        If set (e.g. 0.9), sample from the smallest set of tokens whose cumulative probability >= p.
    eos_id : Optional[int]
        Token ID for end-of-text. If all batch sequences generate eos_id, generation stops early.

    Returns
    -------
    torch.Tensor
        Sequence of tokens including prompt and newly generated tokens, shape (B, T + num_generated).
    """
    model.eval()
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        # 1. Truncate context if it exceeds model's context window
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]

        # 2. Forward pass to get logits
        logits, _ = model(idx_cond)

        # 3. Focus only on the last time-step
        logits = logits[:, -1, :]  # shape: (B, vocab_size)

        # 4. Greedy decoding if temperature is 0
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Scale logits by temperature
            logits = logits / temperature

            # 5. Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out any logits below the K-th value
                logits[logits < v[:, [-1]]] = -float("Inf")

            # 6. Top-P (Nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices right to keep the first token that exceeds top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")

            # 7. Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # 8. Sample next token from probability distribution
            next_token = torch.multinomial(probs, num_samples=1)

        # 9. Append sampled token to sequence
        idx = torch.cat((idx, next_token), dim=1)

        # 10. Early stopping if EOS token generated (for single-batch generation)
        if eos_id is not None and (next_token == eos_id).all():
            break

    return idx


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    device: str = "cpu",
) -> str:
    """
    Convenience wrapper to generate text from a string prompt.

    Parameters
    ----------
    model : GPT
        Trained model.
    tokenizer : BPETokenizer
        Trained tokenizer.
    prompt : str
        Input prompt text.
    max_new_tokens : int
        Number of tokens to generate.
    temperature : float
        Sampling temperature.
    top_k : Optional[int]
        Top-K filter.
    top_p : Optional[float]
        Nucleus filter.
    device : str
        Torch device ("cpu" or "cuda").

    Returns
    -------
    str
        Decoded generated text.
    """
    model.to(device)
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.endoftext_id]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out_idx = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.endoftext_id,
    )
    generated_tokens = out_idx[0].tolist()
    return tokenizer.decode(generated_tokens)
