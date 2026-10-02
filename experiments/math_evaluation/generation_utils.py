"""Tokenizer-based stopping for math evaluation."""


def stop_token_ids(tokenizer):
    """Never infer token IDs from a model directory's name."""
    eos = tokenizer.eos_token_id
    ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    vocab = tokenizer.get_vocab()
    for token in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<｜end▁of▁sentence｜>"):
        if token in vocab:
            ids.append(vocab[token])
    return list(dict.fromkeys(value for value in ids if value is not None))
