import random
import numpy as np
import torch


def worker_init_fn(wid, seed=None):
    """
    Fully deterministic worker seed initialization for DataLoader workers.
    Seeds Python's random, NumPy, and PyTorch.
    """
    base_seed = seed if seed is not None else torch.initial_seed()
    seed_seq = np.random.SeedSequence([base_seed, wid])

    # Derive independent seeds for the three RNGs
    child_seeds = seed_seq.generate_state(3, dtype=np.uint32)

    torch.manual_seed(int(child_seeds[0]))
    np.random.seed(int(child_seeds[1]))
    random.seed(int(child_seeds[2]))
