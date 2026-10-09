"""Generate figures and animated GIFs for the difftree intro deck.

Run from the repo root:  uv run python presentation/make_figures.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from difftree import DifferentiableTreeClassifier, DifferentiableTreeRegressor
from difftree import _model
from difftree.regressor import _regressor_loss

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

NAVY, BLUE, MID, SKY, ICE = "#0b2545", "#1f5f99", "#2a7fc1", "#5aa9dd", "#dbeaf7"
ORANGE = "#e07a1f"
plt.rcParams.update({
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY,
    "ytick.color": NAVY, "axes.titlecolor": NAVY, "axes.titleweight": "bold",
})


def save_gif(anim, name, fps):
    anim.save(OUT / name, writer=PillowWriter(fps=fps), dpi=90)
    print("wrote", name)


# ---------------------------------------------------------------- 1D data
rng = np.random.default_rng(0)
x1 = np.sort(rng.uniform(-3, 3, 400))
f1 = lambda x: np.sin(1.5 * x) + 0.3 * x
s1 = lambda x: 0.05 + 0.25 * (x > 0) * np.abs(x) / 3
y1 = f1(x1) + rng.normal(0, s1(x1))
xg = np.linspace(-3.3, 3.3, 400)

model1 = DifferentiableTreeRegressor(depth=5, n_epochs=300, random_state=3).fit(
    x1[:, None], y1
)


def moments(params, model, X):
    Xs = (X - model.x_mean_) / model.x_std_
    m, tot, ale, rou = _model.batched_regressor_predict_moments(
        jnp.asarray(Xs), params["W"], params["b"], params["log_gamma"],
        params["leaf_mu"], params["leaf_A"], params["leaf_chol_raw"], params["log_alpha"],
    )
    ys = model.y_std_[0]
    return (np.asarray(m[:, 0]) * ys + model.y_mean_[0],
            np.sqrt(np.asarray(tot[:, 0, 0])) * ys,
            np.sqrt(np.asarray(ale[:, 0, 0])) * ys,
            np.sqrt(np.asarray(rou[:, 0, 0])) * ys)


# ---------------------------------------------------------------- GIF 1: gamma sweep
P1 = model1._params_jax()
lg0 = np.asarray(P1["log_gamma"])
scales = np.concatenate([np.geomspace(0.15, 25, 40), np.geomspace(25, 0.15, 40)])

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), height_ratios=[3, 1.3], sharex=True)


def draw_gamma(i):
    ax.clear(); ax2.clear()
    p = dict(P1); p["log_gamma"] = jnp.asarray(lg0 + np.log(scales[i]))
    m, s, sa, sr = moments(p, model1, xg[:, None])
    ax.scatter(x1, y1, s=6, color="0.7")
    ax.fill_between(xg, m - 2 * s, m + 2 * s, color=SKY, alpha=0.35, lw=0)
    ax.plot(xg, m, color=NAVY, lw=2.5)
    ax.set_ylim(-2.4, 2.6); ax.set_ylabel("y")
    ax.set_title(f"split steepness  γ × {scales[i]:6.2f}")
    ax2.stackplot(xg, sa, sr, colors=[MID, ORANGE], alpha=0.8,
                  labels=["aleatoric σ", "routing σ"])
    ax2.set_ylim(0, 0.9); ax2.set_xlabel("x"); ax2.set_ylabel("σ")
    ax2.legend(loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()


save_gif(FuncAnimation(fig, draw_gamma, frames=len(scales)), "anim_gamma.gif", fps=10)
plt.close(fig)

# ---------------------------------------------------------------- GIF 2: training
# A plain full-batch NLL loop with a gamma ramp, recording snapshots.
Xs1 = jnp.asarray((x1[:, None] - model1.x_mean_) / model1.x_std_)
ys1 = jnp.asarray((y1[:, None] - model1.y_mean_) / model1.y_std_)
params = _model.init_regressor_params(jax.random.PRNGKey(3), 5, 1, 1, 1.0, 1.0)
opt = optax.adam(0.03)
state = opt.init(params)
mask = {k: k not in ("log_gamma", "log_alpha", "leaf_A") for k in params}


@jax.jit
def step(params, state):
    loss, g = jax.value_and_grad(_regressor_loss)(params, Xs1, ys1)
    g = {k: (v if mask[k] else jnp.zeros_like(v)) for k, v in g.items()}
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state, loss


snaps, losses = [], []
n_steps = 1500
for it in range(n_steps + 1):
    gam = 1.0 + 4.0 * it / n_steps
    params["log_gamma"] = jnp.full_like(params["log_gamma"], np.log(gam))
    if it % 25 == 0:
        snaps.append((it, dict(params)))
    params, state, loss = step(params, state)
    losses.append(float(loss))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[2.2, 1])


def draw_train(i):
    it, p = snaps[i]
    ax.clear(); ax2.clear()
    m, s, _, _ = moments(p, model1, xg[:, None])
    ax.scatter(x1, y1, s=6, color="0.7")
    ax.fill_between(xg, m - 2 * s, m + 2 * s, color=SKY, alpha=0.35, lw=0)
    ax.plot(xg, m, color=NAVY, lw=2.5)
    ax.set_ylim(-2.4, 2.6); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"step {it}")
    ax2.plot(losses[: it + 1], color=MID, lw=2)
    ax2.set_xlim(0, n_steps); ax2.set_ylim(min(losses) - 0.1, max(losses[:5]) + 0.1)
    ax2.set_xlabel("Adam step"); ax2.set_title("NLL")
    fig.tight_layout()


save_gif(FuncAnimation(fig, draw_train, frames=len(snaps)), "anim_training.gif", fps=8)
plt.close(fig)

# ---------------------------------------------------------------- GIF 3: oblique partition (classifier)
Xm, ym = make_moons(600, noise=0.2, random_state=0)
clf = DifferentiableTreeClassifier(depth=4, n_epochs=300, random_state=0).fit(Xm, ym)
Pc = clf._params_jax()
lgc = np.asarray(Pc["log_gamma"])
gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 160), np.linspace(-1.2, 1.7, 120))
G = np.c_[gx.ravel(), gy.ravel()]
Gs = jnp.asarray((G - clf.x_mean_) / clf.x_std_) if hasattr(clf, "x_mean_") else jnp.asarray(G)
cscales = np.concatenate([np.geomspace(0.1, 20, 35), np.geomspace(20, 0.1, 35)])
fig, ax = plt.subplots(figsize=(7, 5))


def draw_moons(i):
    ax.clear()
    lp = _model.batched_log_path_probs(Gs, Pc["W"], Pc["b"],
                                       jnp.asarray(lgc + np.log(cscales[i])))
    w = np.exp(np.asarray(lp))
    pk = jax.nn.softmax(Pc["leaf_logits"], axis=-1)
    prob1 = (w @ np.asarray(pk))[:, 1].reshape(gx.shape)
    ax.contourf(gx, gy, prob1, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.75)
    ax.contour(gx, gy, prob1, levels=[0.5], colors=NAVY, linewidths=2)
    ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu_r", s=10, edgecolor="k", lw=0.3)
    ax.set_title(f"P(class 1 | x),  γ × {cscales[i]:5.2f}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()


save_gif(FuncAnimation(fig, draw_moons, frames=len(cscales)), "anim_moons.gif", fps=9)
plt.close(fig)

# ---------------------------------------------------------------- Static: heteroscedastic sensor
R, Vmax, K0, Ea, Tref = 8.314e-3, 1.0, 25.0, 50.0, 300.0
true_mean = lambda c, T: Vmax * c / (K0 * np.exp(Ea / R * (1 / T - 1 / Tref)) + c)
true_sd = lambda c, T: 0.02 + 0.10 * true_mean(c, T)
rng = np.random.default_rng(0)
c = rng.uniform(0, 100, 2000); T = rng.uniform(280, 320, 2000)
X = np.c_[c, T]; y = true_mean(c, T) + rng.normal(0, true_sd(c, T))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=400, random_state=0)
hm = DifferentiableTreeRegressor(depth=5, n_epochs=150, gamma_max=5.0,
                                 delta_gamma=0.05, random_state=0).fit(Xtr, ytr)
m, s, sa, sr = hm.predict(Xte, return_components=True)
st = true_sd(Xte[:, 0], Xte[:, 1])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
ax.scatter(st, sa, s=10, alpha=0.5, color=MID)
lim = [0, max(st.max(), sa.max()) * 1.05]
ax.plot(lim, lim, "--", color=NAVY, lw=1)
ax.set_xlabel("true noise σ(c, T)"); ax.set_ylabel("predicted aleatoric σ")
ax.set_title("Learned noise tracks the truth")
ax = axes[1]
nominal = np.linspace(0.05, 0.99, 30)
from scipy.stats import norm
emp = []
for q in nominal:
    z = norm.ppf(0.5 + q / 2)
    emp.append(np.mean(np.abs(yte - m) <= z * s))
ax.plot(nominal, emp, "o-", color=MID, ms=4)
ax.plot([0, 1], [0, 1], "--", color=NAVY, lw=1)
ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
ax.set_title("Calibrated prediction intervals")
fig.tight_layout(); fig.savefig(OUT / "hetero.png", dpi=150); plt.close(fig)
print("wrote hetero.png")

# ---------------------------------------------------------------- Multivariate: conditioning + inverse design
def true_means(X):
    a, b_, c_ = X[:, 0], X[:, 1], X[:, 2]
    y_1 = 1.5 + 0.8 * np.tanh(2.0 * a - c_) - 0.5 * b_
    y_2 = 0.4 * y_1 + 0.6 * np.cos(1.5 * b_) + 0.2 * c_
    return np.stack([y_1, y_2], axis=1)


rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (1500, 3)); Yt = true_means(X)
Sd = np.stack([0.05 + 0.10 * np.abs(Yt[:, 0]), 0.05 + 0.08 * np.abs(Yt[:, 1])], 1)
Y = Yt + rng.normal(0, Sd)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=300, random_state=0)
Ytt = true_means(Xte)
mm = DifferentiableTreeRegressor(depth=5, n_epochs=150, gamma_max=5.0, delta_gamma=0.05,
                                 sigma_init=0.5, random_state=0).fit(Xtr, Ytr)
Yp, cov = mm.predict_full_cov(Xte)
cm, cc, _ = mm.predict_conditional(Xte, observed={0: Ytt[:, 0]})
sd_marg = np.sqrt(cov[:, 1, 1]); sd_cond = np.sqrt(cc[:, 0, 0])
fig, ax = plt.subplots(figsize=(5.5, 4.4))
ax.scatter(sd_marg, sd_cond, s=10, alpha=0.5, color=MID)
lim = [0, sd_marg.max() * 1.05]
ax.plot(lim, lim, "--", color=NAVY, lw=1)
ax.set_xlabel("σ(y₂ | x)"); ax.set_ylabel("σ(y₂ | x, y₁)")
rm = np.sqrt(np.mean((Yp[:, 1] - Ytt[:, 1]) ** 2)); rc = np.sqrt(np.mean((cm[:, 0] - Ytt[:, 1]) ** 2))
ax.set_title(f"Observing y₁ shrinks σ(y₂)\nmedian ratio {np.median(sd_cond / sd_marg):.2f}", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "conditioning.png", dpi=150); plt.close(fig)
print("wrote conditioning.png", rm, rc)

# GIF 4: multi-start inverse design on a 2D slice of log p(y* | x)
Pm = mm._params_jax()
y_star = np.array([1.8, 1.0])
yt = jnp.asarray((y_star - mm.y_mean_) / mm.y_std_)
obs = jnp.asarray([0, 1], dtype=jnp.int32)
x_best, _, _, _ = mm.inverse_design({0: 1.8, 1: 1.0}, n_starts=24, n_steps=600,
                                    learning_rate=0.05,
                                    bounds=(np.full(3, -1.0), np.full(3, 1.0)),
                                    random_state=0)
x3 = float(x_best[2])


def ll_raw(x):  # x in original units, (3,)
    xs = (x - jnp.asarray(mm.x_mean_)) / jnp.asarray(mm.x_std_)
    return _model.inverse_design_log_lik(xs, yt, obs, Pm["W"], Pm["b"], Pm["log_gamma"],
                                         Pm["leaf_mu"], Pm["leaf_A"], Pm["leaf_chol_raw"],
                                         Pm["log_alpha"])


ll_v = jax.jit(jax.vmap(ll_raw))
g = np.linspace(-1, 1, 120)
A, B = np.meshgrid(g, g)
L = np.asarray(ll_v(jnp.asarray(np.c_[A.ravel(), B.ravel(), np.full(A.size, x3)]))).reshape(A.shape)

grad2 = jax.jit(jax.vmap(jax.grad(lambda z: ll_raw(jnp.array([z[0], z[1], x3])))))
starts = np.random.default_rng(3).uniform(-0.95, 0.95, (14, 2))
z = jnp.asarray(starts); o2 = optax.adam(0.03); st2 = o2.init(z)
traj = [np.asarray(z)]
for _ in range(240):
    gz = -grad2(z)
    u, st2 = o2.update(gz, st2, z)
    z = jnp.clip(optax.apply_updates(z, u), -1, 1)
    traj.append(np.asarray(z))
traj = np.array(traj)

fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.pcolormesh(A, B, np.maximum(L, np.percentile(L, 5)), cmap="Blues", shading="auto")
fig.colorbar(im, ax=ax, label="log p(y* | x)")
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
ax.set_title(f"Inverse design: y* = (1.8, 1.0), x₃ = {x3:.2f}", fontsize=12)
lines = [ax.plot([], [], "-", color=ORANGE, lw=1.2)[0] for _ in starts]
dots = ax.scatter(starts[:, 0], starts[:, 1], s=30, color=ORANGE, edgecolor="k", zorder=3)
fig.tight_layout()
frames = list(range(0, len(traj), 4)) + [len(traj) - 1] * 12


def draw_inv(k):
    for j, ln in enumerate(lines):
        ln.set_data(traj[: k + 1, j, 0], traj[: k + 1, j, 1])
    dots.set_offsets(traj[k])
    return lines + [dots]


save_gif(FuncAnimation(fig, draw_inv, frames=frames), "anim_inverse.gif", fps=12)
plt.close(fig)


# ---------------------------------------------------------------- Piecewise-affine surrogates for MILP
# Same problem as notebooks/piecewise_linear_milp.ipynb.
from difftree import export


def truth_pw(X):
    u = X[:, 0] + 0.02 * X[:, 1] - 1.0           # oblique kink
    y0 = np.where(u > 0, 3.0 * u, -1.0 * u) + 0.01 * X[:, 1]
    y1 = np.where(u > 0, 0.8, 0.2)
    return np.stack([y0, y1], axis=1)


rng = np.random.default_rng(0)
lo, hi = np.array([0.0, -50.0]), np.array([2.0, 50.0])
X_te = rng.uniform(lo, hi, (4000, 2)); Y_te = truth_pw(X_te)
Xp = rng.uniform(lo, hi, (60, 2))
Yp = truth_pw(Xp) + 0.01 * rng.normal(size=(60, 2))
common = dict(depth=2, n_epochs=300, batch_size=32, hardening=True, random_state=0)
aff = dict(leaf_type="affine", constant_outputs=[1], **common)
variants = {
    "discontinuous": dict(),
    "continuous, untied": dict(continuous=True),
    "continuous, tied": dict(continuous=True, tied_splits=True),
}
fits2d = {k: DifferentiableTreeRegressor(**aff, **kw).fit(Xp, Yp) for k, kw in variants.items()}
const = DifferentiableTreeRegressor(**common).fit(Xp, Yp)
for name, mod in (("constant", const), ("affine", fits2d["discontinuous"])):
    r = np.sqrt(np.mean((mod.predict(X_te, hard=True) - Y_te) ** 2, axis=0))
    print(f"{name} leaves: hard RMSE {np.round(r, 3)}")

# Figure: constant vs affine leaves along a line across the kink (x1 = 0)
xs = np.linspace(lo[0], hi[0], 600)
S0 = np.c_[xs, np.zeros_like(xs)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for k, (ax, lab) in enumerate(zip(axes, ("y₀: piecewise linear", "y₁: kept constant per leaf"))):
    ax.plot(xs, truth_pw(S0)[:, k], ":", color="0.4", lw=2.5, label="truth")
    ax.plot(xs, const.predict(S0, hard=True)[:, k], color=ORANGE, lw=2, label="constant leaves")
    ax.plot(xs, fits2d["discontinuous"].predict(S0, hard=True)[:, k], color=NAVY, lw=2, label="affine leaves")
    ax.set(xlabel="x₀  (x₁ = 0)", title=lab)
axes[0].legend(fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "leaves_affine.png", dpi=150); plt.close(fig)
print("wrote leaves_affine.png")

# Figure: hardening makes the soft and crisp models agree (1-D, affine leaves)
import copy
rh = np.random.default_rng(0)
xh = rh.uniform(-2, 2, (300, 1)); yh = np.sin(2 * xh[:, 0]) + 0.03 * rh.normal(size=300)
soft = DifferentiableTreeRegressor(depth=3, n_epochs=200, leaf_type="affine", gamma_max=2.0,
                                   random_state=0).fit(xh, yh)
hard = copy.deepcopy(soft).harden(xh, yh)
xg = np.linspace(-2, 2, 1500)[:, None]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, mod, title in ((axes[0], soft, "Before hardening"), (axes[1], hard, "After harden()")):
    ax.scatter(xh[:, 0], yh, s=6, color="0.75")
    ax.plot(xg[:, 0], mod.predict(xg), color=SKY, lw=3, label="soft  predict()")
    ax.plot(xg[:, 0], mod.predict(xg, hard=True), color=NAVY, lw=1.6, label="crisp  predict(hard=True)")
    gap = mod.soft_hard_gap(xh, yh)
    ax.set(xlabel="x", title=f"{title}: crisp RMSE {gap['rmse_hard'][0]:.3f}")
axes[0].set_ylabel("y"); axes[0].legend(fontsize=11, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "hardening.png", dpi=150); plt.close(fig)
print("wrote hardening.png")

# Figure: leaf polytopes and the exported surface
mp = fits2d["discontinuous"]
polys = export.leaf_polytopes(mp, bounds=(lo, hi), X=Xp)
g0, g1 = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
G = np.stack([g0.ravel(), g1.ravel()], axis=1)
region = np.full(G.shape[0], np.nan)
for k, lp in enumerate(polys):
    region[lp.contains(G)] = k
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].pcolormesh(g0, g1, region.reshape(g0.shape), cmap=matplotlib.colors.ListedColormap([ICE, SKY]),
                   shading="auto")
axes[0].scatter(Xp[:, 0], Xp[:, 1], s=12, color=NAVY)
u0 = np.linspace(lo[1], hi[1], 2)
axes[0].plot(1.0 - 0.02 * u0, u0, "--", color=ORANGE, lw=2, label="true kink")
axes[0].set(xlabel="x₀", ylabel="x₁", title=f"{len(polys)} non-empty leaf polytopes of {2 ** mp.depth}")
axes[0].legend(loc="upper right")
pred = export.predict_spec(export.to_spec(mp, bounds=(lo, hi)), G)[:, 0]
im = axes[1].pcolormesh(g0, g1, pred.reshape(g0.shape), cmap="Blues", shading="auto")
axes[1].contour(g0, g1, pred.reshape(g0.shape), levels=12, colors=NAVY, linewidths=0.6)
fig.colorbar(im, ax=axes[1], label="y₀ from the exported spec")
axes[1].set(xlabel="x₀", title="Exported piecewise-affine y₀")
fig.tight_layout(); fig.savefig(OUT / "milp_polytopes.png", dpi=150); plt.close(fig)
print("wrote milp_polytopes.png")

# Figure: jumps vs continuity, sliced across the discontinuous tree's worst split
f3 = lambda Z: np.abs(Z[:, 0]) + Z[:, 1] + np.maximum(Z[:, 2], 0)
r3 = np.random.default_rng(1)
X3 = r3.uniform(-2, 2, (600, 3)); y3 = f3(X3) + 0.02 * r3.normal(size=600)
X3_te = r3.uniform(-2, 2, (5000, 3))
kw3 = dict(depth=3, n_epochs=80, leaf_type="affine", hardening=True, n_hard_epochs=20, random_state=0)
fits3d = {k: DifferentiableTreeRegressor(**kw3, **kw).fit(X3, y3) for k, kw in variants.items()}
for k in variants:
    ra = np.sqrt(np.mean((fits2d[k].predict(X_te, hard=True)[:, 0] - Y_te[:, 0]) ** 2))
    rb = np.sqrt(np.mean((fits3d[k].predict(X3_te, hard=True) - f3(X3_te)) ** 2))
    print(f"{k:>20s}: 2-D RMSE {ra:.3f} jump {fits2d[k].hardening_report_['max_boundary_jump'][0]:.2e} | "
          f"3-D RMSE {rb:.3f} jump {fits3d[k].hardening_report_['max_boundary_jump'][0]:.2e}")


def worst_slice(model, half_width=1.0, n=2000):
    j = export.boundary_jumps(model)[0]
    w = model.params_["W"][j["node"]]
    step = (w / np.linalg.norm(w)) * model.x_std_
    t = np.linspace(-half_width, half_width, n)
    return t, j["worst_point"][0][None, :] + t[:, None] * step[None, :]


styles = {"discontinuous": (MID, "-"), "continuous, untied": (ORANGE, "--"), "continuous, tied": (NAVY, "-")}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, fits, f_true, title in (
    (axes[0], fits2d, lambda Z: truth_pw(Z)[:, 0], "2-D oblique kink (y₀), depth 2"),
    (axes[1], fits3d, f3, "|x₀| + x₁ + max(x₂, 0), depth 3"),
):
    t, S = worst_slice(fits["discontinuous"])
    ax.plot(t, f_true(S), ":", color="0.4", lw=2.5, label="truth")
    for k, mod in fits.items():
        c, ls = styles[k]
        p = mod.predict(S, hard=True)
        ax.plot(t, p[:, 0] if p.ndim == 2 else p, color=c, ls=ls, lw=2, label=k)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set(xlabel="distance across the worst split (standardized x)", ylabel="crisp prediction", title=title)
    ax.legend(fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "jumps_continuity.png", dpi=150); plt.close(fig)
print("wrote jumps_continuity.png")

# Figure: a MILP solution on a split boundary, and its repair
j = export.boundary_jumps(mp, bounds=(lo, hi))[0]
x_b = j["worst_point"][0]
pl = {lp.leaf: lp for lp in export.leaf_polytopes(mp, bounds=(lo, hi))}
routed = mp.apply(x_b[None])[0]
chosen = j["right"] if routed == j["left"] else j["left"]
x_fix = export.repair_solution(mp, x_b, chosen, bounds=(lo, hi))
w = mp.params_["W"][j["node"]]
step = (w / np.linalg.norm(w)) * mp.x_std_
t = np.linspace(-0.3, 0.3, 1200)
S = x_b[None, :] + t[:, None] * step[None, :]
fig, ax = plt.subplots(figsize=(8, 4.6))
for leaf, side, c in ((j["left"], t <= 0, MID), (j["right"], t > 0, MID)):
    yl = pl[leaf].predict(S)[:, 0]
    ax.plot(t, yl, ":", color="0.6", lw=1.2)
ax.plot(t, mp.predict(S, hard=True)[:, 0], color=NAVY, lw=2.5, label="crisp model")
ax.plot(t, truth_pw(S)[:, 0], color=ORANGE, lw=1, alpha=0.6, label="truth")
v_milp = pl[chosen].predict(x_b[None])[0, 0]
v_crisp = mp.predict(x_b[None], hard=True)[0, 0]
ax.plot(0, v_milp, "o", ms=11, color=ORANGE, mec="k", label=f"MILP objective (leaf {chosen}): {v_milp:.3f}")
ax.plot(0, v_crisp, "s", ms=10, color=SKY, mec="k", label=f"predict(hard=True) at that x: {v_crisp:.3f}")
ax.annotate("repair_solution:\nmove δ into leaf %d" % chosen, xy=(0.012, v_milp), xytext=(0.09, v_milp - 0.25),
            arrowprops=dict(arrowstyle="->", color=NAVY), color=NAVY, fontsize=12)
ax.axvline(0, color="0.75", lw=0.8)
ax.set(xlabel="distance across split (standardized x)", ylabel="y₀",
       title="A MILP optimum can sit on a split boundary")
ax.legend(fontsize=10, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "milp_repair.png", dpi=150); plt.close(fig)
print("wrote milp_repair.png", v_milp, v_crisp, mp.predict(x_fix[None], hard=True)[0, 0])
