# src/data/loader.py
"""
Hugging Face dataset loader for PhreshPhish.
Uses streaming to avoid downloading the full ~36 GB dataset.
"""

from datasets import load_dataset
from typing import Optional, Iterator, Dict, Any


def load_phreshphish(
    split: str = "train",
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Load PhreshPhish dataset from Hugging Face.

    Parameters
    ----------
    split : str
        "train" or "test"
    streaming : bool
        If True, returns a streaming iterable (recommended)
    max_samples : int or None
        Limit the number of samples (useful for debugging)

    Yields
    ------
    dict
        Single example from the dataset
    """
    ds = load_dataset(
        "phreshphish/phreshphish",
        split=split,
        streaming=streaming,
        trust_remote_code=False,
    )

    for i, example in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        yield example


def load_phreshphish_as_list(
    split: str = "train",
    max_samples: int = 5000,
) -> list:
    """
    Convenience helper that materialises a limited number of samples
    into a Python list (still memory-safe if max_samples is small).
    """
    return list(load_phreshphish(split=split, streaming=True, max_samples=max_samples))