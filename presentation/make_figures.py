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
        params["leaf_mu"], params["leaf_chol_raw"], params["log_alpha"],
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
mask = {k: (k != "log_gamma" and k != "log_alpha") for k in params}


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
                                         Pm["leaf_mu"], Pm["leaf_chol_raw"], Pm["log_alpha"])


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
