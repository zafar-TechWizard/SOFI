"""
BRAIN/llm/retry_utils.py — Jittered exponential backoff

Prevents retry storms when multiple sessions hit the same rate-limited
provider. Fixed exponential backoff creates synchronized retry spikes;
jitter decorrelates them.

Pattern from Hermes Agent (agent/retry_utils.py).
"""

import random
import time


def jittered_backoff(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter_ratio: float = 0.5,
) -> float:
    """
    Compute a jittered exponential backoff delay.

    Args:
        attempt:      0-indexed retry attempt number
        base_delay:   base delay in seconds (doubled each attempt)
        max_delay:    hard ceiling on delay
        jitter_ratio: fraction of delay to randomize (0.5 = ±50%)

    Returns:
        Delay in seconds, suitable for asyncio.sleep().
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    seed = (time.time_ns() ^ (attempt * 0x9E3779B9)) & 0xFFFFFFFF
    jitter = random.Random(seed).uniform(0, jitter_ratio * delay)
    return delay + jitter
