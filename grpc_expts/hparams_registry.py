import numpy as np
import hashlib


def seed_hash(*args):
    """Creates a unique seed hash for reproducibility"""
    return int(hashlib.sha256(str(args).encode("utf-8")).hexdigest(), 16) % (10**8)


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert name not in hparams
        random_state = np.random.RandomState(seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam("learning_rate", 3e-4, lambda r: 10 ** r.uniform(-5, -3))
    _hparam("batch_size", 32, lambda r: int(2 ** r.uniform(3, 6)))
    _hparam("dropout_rate", 0.5, lambda r: r.choice([0.0, 0.5]))

    # Algorithm-specific hparam definitions.
    if algorithm == "AlgorithmA":
        _hparam("alpha", 1.0, lambda r: 10 ** r.uniform(-3, 3))
        _hparam("beta", 0.5, lambda r: r.choice([0.1, 0.5, 1.0]))

    elif algorithm == "AlgorithmB":
        _hparam("gamma", 1.0, lambda r: 10 ** r.uniform(-2, 2))
        _hparam("weight_decay", 0.01, lambda r: 10 ** r.uniform(-6, -2))

    # Dataset-specific hparam definitions.
    if dataset == "DatasetX":
        _hparam("learning_rate", 1e-4, lambda r: 10 ** r.uniform(-5, -4))

    if dataset == "DatasetY":
        _hparam("batch_size", 64, lambda r: int(2 ** r.uniform(5, 7)))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
