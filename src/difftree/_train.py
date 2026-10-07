"""Shared training and calibration loops for differentiable trees.

The two phases:

1. Training. Optimize splits and leaves on the training set with Adam, while
   ``log_gamma`` is externally annealed (1 -> gamma_max) on a fixed schedule.
   ``log_alpha`` (regressor only) is held at 0.
2. Calibration. Freeze splits and leaves; optimize ``log_gamma`` (per-depth)
   and ``log_alpha`` (regressor only) on the validation set by minimizing NLL.
   This is the LLPR/temperature-scaling analogue.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _shuffled_batches(rng: np.random.Generator, n: int, batch_size: int):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def _build_optimizer(params, learning_rate, trainable_keys):
    """Adam on ``trainable_keys`` with gradient clipping; everything else
    receives a zero update (truly frozen)."""
    trainable = set(trainable_keys)

    def labels(p):
        return {k: ("train" if k in trainable else "freeze") for k in p}

    return optax.multi_transform(
        {
            "train": optax.chain(
                optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
            ),
            "freeze": optax.set_to_zero(),
        },
        labels,
    )


def _run_phase(
    params,
    loss_fn,
    X_tr,
    y_tr,
    optimizer,
    rng,
    n_epochs,
    batch_size,
    gamma_schedule,
    label,
    verbose,
):
    """Run one training phase. ``gamma_schedule`` is a callable epoch -> gamma."""
    opt_state = optimizer.init(params)
    depth = params["log_gamma"].shape[0]

    @jax.jit
    def step(params, opt_state, batch_x, batch_y):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n = X_tr.shape[0]
    for epoch in range(n_epochs):
        gamma_now = gamma_schedule(epoch)
        params = {**params, "log_gamma": jnp.full((depth,), float(jnp.log(gamma_now)))}
        epoch_loss = 0.0
        n_batches = 0
        for idx in _shuffled_batches(rng, n, batch_size):
            params, opt_state, loss = step(params, opt_state, X_tr[idx], y_tr[idx])
            epoch_loss += float(loss)
            n_batches += 1
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(
                f"  [{label}] epoch {epoch:4d}  gamma={gamma_now:5.2f}  "
                f"loss={epoch_loss / max(1, n_batches):.4f}"
            )
    return params


def train(
    params: dict,
    loss_fn: Callable,
    X_tr: jnp.ndarray,
    y_tr: jnp.ndarray,
    *,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    gamma_init: float,
    gamma_max: float,
    delta_gamma: float,
    rng: np.random.Generator,
    trainable_keys: tuple[str, ...],
    warmup_keys: tuple[str, ...] | None = None,
    warmup_fraction: float = 0.25,
    verbose: bool = False,
) -> dict:
    """Train splits and leaves with annealed gamma. Returns updated params.

    If ``warmup_keys`` is provided, the first ``warmup_fraction`` of epochs
    optimize only those parameters (rest frozen) before switching to
    ``trainable_keys``. This is the standard remedy for variance-collapse in
    heteroscedastic NLL: learn good leaf means before unfreezing leaf variances.
    The gamma schedule continues unchanged across phases.
    """
    n_warmup = int(round(n_epochs * warmup_fraction)) if warmup_keys else 0
    n_main = n_epochs - n_warmup

    def schedule(epoch):
        return min(gamma_init + epoch * delta_gamma, gamma_max)

    if n_warmup > 0:
        if verbose:
            print(f"  [warmup] {n_warmup} epochs with frozen leaf variances")
        warmup_opt = _build_optimizer(params, learning_rate, warmup_keys)
        params = _run_phase(
            params, loss_fn, X_tr, y_tr, warmup_opt, rng, n_warmup,
            batch_size, schedule, label="warmup", verbose=verbose,
        )

    main_schedule = lambda epoch: schedule(epoch + n_warmup)
    main_opt = _build_optimizer(params, learning_rate, trainable_keys)
    params = _run_phase(
        params, loss_fn, X_tr, y_tr, main_opt, rng, n_main,
        batch_size, main_schedule, label="train", verbose=verbose,
    )
    return params


def calibrate(
    params: dict,
    loss_fn: Callable,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    *,
    calibration_lr: float,
    n_calibration_steps: int,
    calib_keys: tuple[str, ...],
    verbose: bool = False,
) -> dict:
    """Fit ``calib_keys`` (e.g. ('log_gamma', 'log_alpha')) on validation NLL,
    holding everything else fixed. Full-batch since the validation set is small.
    """
    optimizer = _build_optimizer(params, calibration_lr, calib_keys)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, X_val, y_val)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    last = None
    for s in range(n_calibration_steps):
        params, opt_state, loss = step(params, opt_state)
        last = float(loss)
        if verbose and (s % max(1, n_calibration_steps // 10) == 0 or s == n_calibration_steps - 1):
            print(f"  calib step {s:4d}  val_nll={last:.4f}")
    return params
