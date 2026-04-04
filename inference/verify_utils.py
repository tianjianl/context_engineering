"""Shared math verification infrastructure using math-verify."""

import multiprocessing
import os
import sys
import warnings
from typing import List, Optional, Tuple

# Default timeout for verification operations (in seconds)
DEFAULT_VERIFY_TIMEOUT = 5

try:
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    parse = None
    verify = None


def _pool_init():
    """Initialize pool worker - suppress output."""
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _verify_single(args: Tuple[str, str]) -> Tuple[bool, str, Optional[str]]:
    """Worker function for single verification.

    Returns (is_correct, status, parsed_answer_str).
    """
    gold_answer, generated_text = args
    try:
        # Parse gold answer
        gold_text = f"${gold_answer}$" if not gold_answer.startswith('$') else gold_answer
        gold_parsed = parse(gold_text)
        if not gold_parsed:
            return (False, "gold_parse_failed", None)

        # Parse generated text
        answer_parsed = parse(generated_text)
        if not answer_parsed:
            return (False, "answer_parse_failed", None)

        parsed_answer_str = str(answer_parsed) if answer_parsed else None
        is_correct = verify(gold_parsed, answer_parsed)
        return (is_correct, "verified" if is_correct else "incorrect", parsed_answer_str)

    except Exception as e:
        return (False, f"error: {str(e)[:50]}", None)


def verify_batch(items: List[Tuple[str, str]],
                 timeout: float = DEFAULT_VERIFY_TIMEOUT,
                 num_workers: int = None) -> List[Tuple[bool, str, Optional[str]]]:
    """Verify a batch of (gold_answer, generated_text) pairs in parallel.

    Uses multiprocessing.Pool with maxtasksperchild to handle stuck workers.
    Returns list of (is_correct, status, parsed_answer) tuples.
    """
    if not items:
        return []
    if not MATH_VERIFY_AVAILABLE:
        raise ImportError("math-verify not installed. Install with: pip install 'math-verify[antlr4_13_2]'")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    results = [(False, "timeout", None)] * len(items)

    with multiprocessing.Pool(num_workers, initializer=_pool_init, maxtasksperchild=10) as pool:
        async_results = []
        for i, item in enumerate(items):
            ar = pool.apply_async(_verify_single, (item,))
            async_results.append((i, ar))

        for i, ar in async_results:
            try:
                results[i] = ar.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                results[i] = (False, "timeout", None)
            except Exception as e:
                results[i] = (False, f"error: {str(e)[:30]}", None)

    return results


def verify_single(gold_answer: str, generated_text: str,
                  timeout: float = DEFAULT_VERIFY_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Verify a single (gold_answer, generated_text) pair.

    Convenience wrapper around verify_batch.
    """
    results = verify_batch([(gold_answer, generated_text)], timeout=timeout, num_workers=1)
    return results[0] if results else (False, "error", None)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute unbiased pass@k estimate.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        Probability that at least one of k samples is correct.
        Uses the formula: 1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # pass@k = 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result
