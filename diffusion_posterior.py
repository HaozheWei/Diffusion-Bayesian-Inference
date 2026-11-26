#!/usr/bin/env python3
"""
Amortised diffusion-model posterior sampling vs MCMC on a bimodal inverse problem.

Problem ("where did the fire start?"): a source at unknown 2D location theta emits a signal
and two sensors report noisy distances to it. With only two sensors the posterior has two
mirror-image modes (reflection across the line through the sensors).

Methods compared on the SAME observations, against the EXACT posterior (computed on a grid):
  * conditional score-based diffusion model, trained once on simulated (theta, y) pairs
    (amortised: new observations need no further forward-model evaluations)
  * random-walk Metropolis (RWM) and MALA, run from scratch for every observation

Everything is NumPy (the score network and its backprop are written by hand) so it runs
anywhere. See diffusion_posterior_torch.py for the PyTorch version of the model.
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

# =====================================================================================
# 1. Bayesian inverse problem
# =====================================================================================
SENSORS = np.array([[-1.5, 0.0], [1.5, 0.0]])
NOISE_SD = 0.15
PRIOR_SD = 2.0


def forward_model(theta):
    """theta (..., 2) -> distances to the two sensors (..., 2)."""
    d = theta[..., None, :] - SENSORS
    return np.sqrt((d ** 2).sum(-1))


def log_post(theta, y):
    r = forward_model(theta) - y
    return -0.5 * (theta ** 2).sum(-1) / PRIOR_SD ** 2 - 0.5 * (r ** 2).sum(-1) / NOISE_SD ** 2


def grad_log_post(theta, y):
    d = theta[..., None, :] - SENSORS
    f = np.sqrt((d ** 2).sum(-1)) + 1e-9
    r = f - y
    g_lik = -((r / f)[..., None] * d).sum(-2) / NOISE_SD ** 2
    return g_lik - theta / PRIOR_SD ** 2


def simulate(n, rng):
    theta = rng.normal(0.0, PRIOR_SD, (n, 2))
    y = forward_model(theta) + NOISE_SD * rng.normal(size=(n, 2))
    return theta, y


# ---- exact reference posterior on a grid ----
GRID_L, GRID_N = 7.0, 561
_gx = np.linspace(-GRID_L, GRID_L, GRID_N)
GRID = np.stack(np.meshgrid(_gx, _gx, indexing="ij"), -1).reshape(-1, 2)
GRID_H = _gx[1] - _gx[0]


def grid_posterior(y):
    lp = log_post(GRID, y[None, :])
    p = np.exp(lp - lp.max())
    return p / p.sum()


def sample_grid(p, n, rng):
    idx = rng.choice(p.size, n, p=p)
    return GRID[idx] + rng.uniform(-GRID_H / 2, GRID_H / 2, (n, 2))


def upper_mass(p):
    return p[GRID[:, 1] > 0].sum()


def bimodal_gap_mass(p, w=0.3):
    return p[np.abs(GRID[:, 1]) < w].sum()


# =====================================================================================
# 2. Metrics
# =====================================================================================
N_PROJ, N_Q = 200, 500
_d = np.random.default_rng(12345).normal(size=(N_PROJ, 2))
DIRS = _d / np.linalg.norm(_d, axis=1, keepdims=True)
QLEVELS = (np.arange(N_Q) + 0.5) / N_Q


def mode_balance_err(samples):
    return float(abs((samples[:, 1] > 0).mean() - 0.5))


def fold(a):
    """Reflect samples into the upper half-plane. The two modes are mirror images, so this overlays them:
    sliced-W2 on folded samples measures within-mode SHAPE accuracy, independent of the mode weights
    (which mode_balance_err measures separately). Plain sliced-W2 on a bimodal law is dominated by
    sampling noise in the mode weights."""
    b = a.copy()
    b[:, 1] = np.abs(b[:, 1])
    return b


def proj_quantiles(a):
    """Quantiles of the folded samples along fixed random directions (for sliced W2)."""
    P = np.sort(fold(a) @ DIRS.T, axis=0)
    pos = QLEVELS * (P.shape[0] - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, P.shape[0] - 1)
    w = (pos - lo)[:, None]
    return P[lo] * (1 - w) + P[hi] * w


def shape_err(a, ref_q):
    """Sliced 2-Wasserstein distance between folded samples a and the (pre-computed) reference quantiles."""
    return float(np.sqrt(((proj_quantiles(a) - ref_q) ** 2).mean()))


# =====================================================================================
# 3. MCMC baselines (vectorised over test observations and chains)
# =====================================================================================
def run_mcmc(kind, y, K, T, rng, adapt_len=1000):
    """y: (n_y, 2). Returns chains (T, n_y, K, 2). Chains start from prior draws.
    Each chain tunes its own step size online (Robbins-Monro) towards the usual optimal acceptance rates
    (a single shared step size left some prior-initialised MALA chains stuck far from the posterior)."""
    n_y = y.shape[0]
    yb = y[:, None, :]
    th = rng.normal(0.0, PRIOR_SD, (n_y, K, 2))
    lp = log_post(th, yb)
    if kind == "mala":
        g = grad_log_post(th, yb)
    log_step = np.full((n_y, K), np.log(0.3 if kind == "rwm" else 0.15))   # one step size PER CHAIN
    target = 0.30 if kind == "rwm" else 0.57
    chain = np.empty((T, n_y, K, 2))
    for i in range(T):
        s = np.exp(log_step)[..., None]
        if kind == "rwm":
            prop = th + s * rng.normal(size=th.shape)
            lp_prop = log_post(prop, yb)
            log_a = lp_prop - lp
        else:
            prop = th + 0.5 * s ** 2 * g + s * rng.normal(size=th.shape)
            lp_prop = log_post(prop, yb)
            g_prop = grad_log_post(prop, yb)
            fwd = -0.5 * (((prop - th - 0.5 * s ** 2 * g) / s) ** 2).sum(-1)
            bwd = -0.5 * (((th - prop - 0.5 * s ** 2 * g_prop) / s) ** 2).sum(-1)
            log_a = lp_prop - lp + bwd - fwd
        log_a = np.where(np.isfinite(log_a), log_a, -np.inf)
        acc = np.log(rng.uniform(size=log_a.shape)) < log_a
        th = np.where(acc[..., None], prop, th)
        lp = np.where(acc, lp_prop, lp)
        if kind == "mala":
            g = np.where(acc[..., None], g_prop, g)
        if i < adapt_len:
            log_step += 2.0 * (acc.astype(float) - target) / (i + 1) ** 0.6
        chain[i] = th
    return chain


def pooled(chain, T_use, n_out, rng):
    """Discard first half of the first T_use steps, pool chains, subsample n_out points per observation."""
    burn = T_use // 2
    c = chain[burn:T_use]                              # (t, n_y, K, 2)
    t, n_y, K, _ = c.shape
    flat = c.transpose(1, 0, 2, 3).reshape(n_y, t * K, 2)
    out = []
    for j in range(n_y):
        idx = rng.choice(flat.shape[1], n_out, replace=flat.shape[1] < n_out)
        out.append(flat[j, idx])
    return out


def rhat(chain_y):
    """Gelman-Rubin R-hat. chain_y: (t, K) scalar traces."""
    t, K = chain_y.shape
    W = chain_y.var(0, ddof=1).mean()
    Bv = t * chain_y.mean(0).var(ddof=1)
    var_hat = (t - 1) / t * W + Bv / t
    return float(np.sqrt(var_hat / max(W, 1e-12)))


def mode_switches(chain_y):
    """Number of sign changes of theta_y along each chain, averaged over chains. chain_y: (t, K)."""
    return float((np.diff(np.sign(chain_y), axis=0) != 0).sum(0).mean())


# =====================================================================================
# 4. Conditional score-based diffusion model (VP-SDE), hand-written MLP + Adam
# =====================================================================================
BETA_MIN, BETA_MAX, T_MIN = 0.1, 20.0, 2e-3
THETA_SCALE = PRIOR_SD
FREQS = 2.0 ** np.arange(5)


def alpha_sigma(t):
    la = -0.5 * (BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2)
    a = np.exp(la)
    return a, np.sqrt(1.0 - a ** 2)


def beta_fn(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def time_features(t):
    ang = np.pi * t[:, None] * FREQS[None, :]
    return np.concatenate([t[:, None], np.sin(ang), np.cos(ang)], 1)


def silu(x):
    return x * expit(x)


def silu_grad(x):
    s = expit(x)
    return s * (1.0 + x * (1.0 - s))


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32)
                  for i in range(len(sizes) - 1)]
        self.W[-1] *= 0.1
        self.b = [np.zeros(sizes[i + 1], np.float32) for i in range(len(sizes) - 1)]

    def params(self):
        return self.W + self.b

    def forward(self, x, params=None, cache=False):
        P = params if params is not None else self.params()
        L = len(P) // 2
        W, b = P[:L], P[L:]
        a, acts, pres = x, [x], []
        for i in range(L - 1):
            z = a @ W[i] + b[i]
            pres.append(z)
            a = silu(z)
            acts.append(a)
        out = a @ W[-1] + b[-1]
        return (out, acts, pres) if cache else out

    def backward(self, dout, acts, pres):
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        d = dout
        for i in range(L - 1, -1, -1):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * silu_grad(pres[i - 1])
        return gW + gb


def make_inputs(x_t, y_n, t):
    return np.concatenate([x_t, y_n, time_features(t)], 1).astype(np.float32)


def train_model(theta, y, steps, batch, lr0, rng, log=print, width=128, depth=4):
    th_n = (theta / THETA_SCALE).astype(np.float32)
    y_mu, y_sd = y.mean(0), y.std(0)
    y_n = ((y - y_mu) / y_sd).astype(np.float32)
    in_dim = 2 + 2 + 11
    net = MLP([in_dim] + [width] * depth + [2], rng)
    P = net.params()
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    ema = [p.copy() for p in P]
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    n = th_n.shape[0]
    t0 = time.time()
    losses = []
    for it in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        x0, yc = th_n[idx], y_n[idx]
        t = rng.uniform(T_MIN, 1.0, batch)
        a, s = alpha_sigma(t)
        eps = rng.normal(size=x0.shape).astype(np.float32)
        x_t = a[:, None].astype(np.float32) * x0 + s[:, None].astype(np.float32) * eps
        pred, acts, pres = net.forward(make_inputs(x_t, yc, t), cache=True)
        diff = pred - eps
        loss = float((diff ** 2).mean())
        grads = net.backward(2.0 * diff / diff.size, acts, pres)
        lr = lr0 * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * it / steps)))
        for k, (p, g) in enumerate(zip(P, grads)):
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh, vh = m[k] / (1 - b1 ** it), v[k] / (1 - b2 ** it)
            p -= lr * mh / (np.sqrt(vh) + eps_adam)
            ema[k] = 0.999 * ema[k] + 0.001 * p
        losses.append(loss)
        if it % 2000 == 0 or it == 1:
            log(f"  step {it:6d}/{steps}  loss {np.mean(losses[-500:]):.4f}  ({time.time() - t0:.0f}s)")
    return dict(ema=ema, net=net, y_mu=y_mu, y_sd=y_sd, losses=np.array(losses))


class Sampler:
    def __init__(self, model):
        self.net, self.P = model["net"], model["ema"]
        self.y_mu, self.y_sd = model["y_mu"], model["y_sd"]
        self.nfe = 0

    def eps(self, x, y_n, t):
        self.nfe += 1
        tt = np.full(x.shape[0], t)
        return self.net.forward(make_inputs(x, y_n, tt), params=self.P)

    def _grid(self, n_steps):
        u = np.linspace(1.0, 0.0, n_steps + 1)
        return T_MIN + (1.0 - T_MIN) * u ** 2         # denser steps near t=0

    def sample(self, y, n_steps, rng, method="sde"):
        """y: (N, 2) raw observations (one row per desired sample). Returns (N, 2) in theta units."""
        y_n = ((y - self.y_mu) / self.y_sd).astype(np.float32)
        x = rng.normal(size=(y.shape[0], 2)).astype(np.float32)
        ts = self._grid(n_steps)
        for i in range(n_steps):
            t, t_next = ts[i], ts[i + 1]
            h = t - t_next
            a, s = alpha_sigma(t)
            e = self.eps(x, y_n, t)
            score = -e / s
            beta = beta_fn(t)
            if method == "sde":                                   # reverse SDE, Euler-Maruyama
                x = x + h * (0.5 * beta * x + beta * score) \
                    + np.sqrt(beta * h) * rng.normal(size=x.shape).astype(np.float32)
            else:                                                 # probability-flow ODE, Heun
                d1 = 0.5 * beta * x + 0.5 * beta * score
                x_e = x + h * d1
                if i < n_steps - 1:
                    a2, s2 = alpha_sigma(t_next)
                    e2 = self.eps(x_e, y_n, t_next)
                    d2 = 0.5 * beta_fn(t_next) * x_e + 0.5 * beta_fn(t_next) * (-e2 / s2)
                    x = x + 0.5 * h * (d1 + d2)
                else:
                    x = x_e
        a, s = alpha_sigma(ts[-1])                                # final Tweedie denoising step
        e = self.eps(x, y_n, ts[-1])
        x0 = (x - s * e) / a
        return (x0 * THETA_SCALE).astype(np.float64)


# =====================================================================================
# 5. Experiment
# =====================================================================================
def self_test(rng):
    th = rng.normal(size=(5, 3, 2))
    y = rng.normal(size=(5, 1, 2)) + 2.0
    g = grad_log_post(th, y)
    num = np.zeros_like(g)
    for k in range(2):
        e = np.zeros(2)
        e[k] = 1e-6
        num[..., k] = (log_post(th + e, y) - log_post(th - e, y)) / 2e-6
    assert np.allclose(g, num, atol=1e-4), "analytic gradient mismatch"
    p = grid_posterior(np.array([2.0, 3.0]))
    assert abs(p.sum() - 1) < 1e-9
    print("self-test passed: gradient matches finite differences; grid posterior normalised")


def plot_diffusion_results(out_dir, ref_ps, th_star, samples, rows, floor, losses=None, tag=""):
    """Figures for a diffusion-sampler evaluation (used by the PyTorch script; NumPy-only, so testable anywhere).

    ref_ps  : list of exact grid posteriors, one per test observation
    th_star : (n_y, 2) true source locations
    samples : dict key -> array (n_y, n_s, 2) of posterior samples, key like "sde-100" / "ode-50"
    rows    : list of dicts with method, n_steps, nfe, shape_mean, shape_se, mode_mean
    floor   : exact-sampler noise floor of the shape error
    losses  : optional 1D array of training losses
    Returns the list of files written."""
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    written = []

    # (a) one observation: exact posterior vs samples
    keys = [k for k in ("sde-100", "ode-50") if k in samples] or list(samples)[:2]
    fig, axs = plt.subplots(1, 1 + len(keys), figsize=(3.8 * (1 + len(keys)), 3.9), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    p = ref_ps[0].reshape(GRID_N, GRID_N)
    axs[0].imshow(p.T, origin="lower", extent=[-GRID_L, GRID_L, -GRID_L, GRID_L], cmap="Greys", aspect="equal")
    axs[0].set_title("Exact posterior (grid)")
    for ax, k in zip(axs[1:], keys):
        ax.scatter(*samples[k][0].T, s=3, alpha=0.4, c="tab:blue")
        ax.set_title(f"Diffusion sampler ({k}) {tag}".strip())
    lim = min(GRID_L, max(5.0, 1.15 * max(np.abs(samples[keys[0]][0]).max(), np.abs(th_star[0]).max())))
    for ax in axs:
        ax.scatter(*SENSORS.T, marker="^", c="red", s=40, zorder=5)
        ax.scatter(*th_star[0], marker="*", c="gold", edgecolors="k", s=120, zorder=6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    fig.tight_layout()
    f = os.path.join(out_dir, "fig_posteriors.png"); fig.savefig(f, bbox_inches="tight"); plt.close(fig); written.append(f)

    # (b) accuracy vs network evaluations
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, col, lab in [("sde", "tab:blue", "reverse SDE (Euler-Maruyama)"),
                             ("ode", "tab:purple", "probability-flow ODE (Heun)")]:
        pts = sorted((r["nfe"], r["shape_mean"], r["shape_se"], r["mode_mean"]) for r in rows if r["method"] == method)
        if not pts:
            continue
        axs[0].errorbar([q[0] for q in pts], [q[1] for q in pts], [q[2] for q in pts], marker="o", c=col, label=lab)
        axs[1].plot([q[0] for q in pts], [q[3] for q in pts], marker="o", c=col, label=lab)
    axs[0].axhline(floor, c="grey", ls=":", label="exact-sampler noise floor")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("network evaluations per sample (NFE)"); axs[0].set_ylabel("Within-mode shape error (folded sliced W2)")
    axs[0].legend(fontsize=7)
    axs[1].set_xscale("log"); axs[1].set_xlabel("network evaluations per sample (NFE)")
    axs[1].set_ylabel("Mode-balance error |P(theta_y>0) - 0.5|")
    fig.tight_layout()
    f = os.path.join(out_dir, "fig_accuracy_vs_nfe.png"); fig.savefig(f, bbox_inches="tight"); plt.close(fig); written.append(f)

    # (c) training loss
    if losses is not None and len(losses) > 1:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        k = max(1, len(losses) // 200)
        ax.plot(np.convolve(np.asarray(losses), np.ones(k) / k, mode="valid"))
        ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.set_title(f"Training loss {tag}".strip())
        fig.tight_layout()
        f = os.path.join(out_dir, "fig_training_loss.png"); fig.savefig(f, bbox_inches="tight"); plt.close(fig); written.append(f)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.steps, args.n_test = 20_000, 3_000, 4
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    self_test(rng)
    res = {"config": vars(args)}

    # ---------- train (amortised) ----------
    ckpt = os.path.join(args.out, "model.npz")
    if args.stage == "eval":
        rng = np.random.default_rng(args.seed + 1)
    th_tr, y_tr = simulate(args.n_train, rng)               # n_train forward-model evaluations
    if (os.path.exists(ckpt) and not args.retrain and not args.quick and args.stage != "train") \
            or (args.stage == "eval"):
        z = np.load(ckpt, allow_pickle=True)
        model = z["model"].item()
        net = MLP([15] + [128] * 4 + [2], np.random.default_rng(0))
        model["net"] = net
        print("loaded cached model")
    else:
        print(f"training on {args.n_train} simulated (theta, y) pairs, {args.steps} steps")
        t0 = time.time()
        model = train_model(th_tr, y_tr, args.steps, 256, 2e-3, rng)
        res["train_seconds"] = time.time() - t0
        if not args.quick:
            save = {k: v for k, v in model.items() if k != "net"}
            np.savez(ckpt, model=np.array(save, dtype=object))
    res["train_sim_evals"] = args.n_train
    if args.stage == "train":
        print("training stage done; model cached at", ckpt)
        return
    sampler = Sampler(model)

    # ---------- test observations: strongly bimodal posteriors ----------
    th_star, y_test, refs, ref_ps = [], [], [], []
    while len(y_test) < args.n_test:
        th, y = simulate(1, rng)
        p = grid_posterior(y[0])
        if bimodal_gap_mass(p) < 0.02:
            th_star.append(th[0]); y_test.append(y[0]); ref_ps.append(p)
    th_star, y_test = np.array(th_star), np.array(y_test)
    n_y, n_s = args.n_test, 2000
    refs = [sample_grid(p, 10000, rng) for p in ref_ps]
    ref_q = [proj_quantiles(r) for r in refs]
    floor = np.mean([shape_err(sample_grid(p, n_s, rng), ref_q[j]) for j, p in enumerate(ref_ps)])
    res["floor_sw2"] = float(floor)
    print(f"{n_y} strongly bimodal test observations; exact-sampler noise floor (folded SW2) = {floor:.4f}")

    # ---------- diffusion: accuracy vs number of network evaluations ----------
    configs = [("sde", 25), ("sde", 100), ("sde", 250), ("ode", 5), ("ode", 10), ("ode", 25), ("ode", 50)]
    y_rep = np.repeat(y_test, n_s, axis=0)
    diff_res, diff_samples = {}, {}
    for method, n_steps in configs:
        sampler.nfe = 0
        t0 = time.time()
        out = sampler.sample(y_rep, n_steps, rng, method)
        dt = time.time() - t0
        S = out.reshape(n_y, n_s, 2)
        sw = [shape_err(S[j], ref_q[j]) for j in range(n_y)]
        mb = [mode_balance_err(S[j]) for j in range(n_y)]
        key = f"{method}-{n_steps}"
        diff_res[key] = dict(method=method, n_steps=n_steps, nfe=sampler.nfe,
                             sw2_mean=float(np.mean(sw)), sw2_se=float(np.std(sw, ddof=1) / np.sqrt(n_y)),
                             mode_err_mean=float(np.mean(mb)), seconds_total=dt,
                             seconds_per_obs=dt / n_y)
        diff_samples[key] = S
        print(f"  diffusion {key:8s} NFE={sampler.nfe:4d}  shape-err={np.mean(sw):.4f}  mode-balance err={np.mean(mb):.3f}"
              f"  ({dt:.1f}s for {n_y} observations x {n_s} samples)")
    res["diffusion"] = diff_res

    # ---------- MCMC ----------
    budgets = [1_000, 4_000, 16_000, 64_000]
    mcmc_res = {}
    mcmc_chains = {}
    for kind in ["rwm", "mala"]:
        cost = 1 if kind == "rwm" else 2        # MALA step = one posterior eval + one gradient eval
        for K in [4, 100]:
            T_max = budgets[-1] // (K * cost)
            t0 = time.time()
            chain = run_mcmc(kind, y_test, K, T_max, rng)
            dt = time.time() - t0
            mcmc_chains[(kind, K)] = chain
            for B in budgets:
                T_use = B // (K * cost)
                if T_use < 4:
                    continue
                S = pooled(chain, T_use, n_s, rng)
                sw = [shape_err(S[j], ref_q[j]) for j in range(n_y)]
                mb = [mode_balance_err(S[j]) for j in range(n_y)]
                mcmc_res[f"{kind}-K{K}-B{B}"] = dict(kind=kind, K=K, budget=B, T=T_use,
                                                     sw2_mean=float(np.mean(sw)),
                                                     sw2_se=float(np.std(sw, ddof=1) / np.sqrt(n_y)),
                                                     mode_err_mean=float(np.mean(mb)))
            print(f"  {kind} K={K:3d}: T={T_max} steps/chain, {dt:.1f}s")
    res["mcmc"] = mcmc_res

    # ---------- mixing diagnostics, K=4 chains at largest budget ----------
    diag = {}
    for kind in ["rwm", "mala"]:
        ch = mcmc_chains[(kind, 4)]
        T = ch.shape[0]
        half = ch[T // 2:]
        rh = [rhat(half[:, j, :, 1]) for j in range(n_y)]
        sw_ = [mode_switches(half[:, j, :, 1]) for j in range(n_y)]
        both = [(np.unique(np.sign(half[:, j, k, 1])).size == 2) for j in range(n_y) for k in range(4)]
        diag[kind] = dict(rhat_median=float(np.median(rh)), rhat_max=float(np.max(rh)),
                          switches_per_chain_mean=float(np.mean(sw_)),
                          frac_chains_visiting_both_modes=float(np.mean(both)))
        print(f"  {kind} (4 chains, {T} steps): median R-hat(theta_y)={np.median(rh):.2f}, "
              f"mean mode switches/chain={np.mean(sw_):.2f}, chains visiting both modes={np.mean(both):.0%}")
    res["mcmc_diagnostics"] = diag

    # ---------- calibration of the diffusion posterior (simulation-based calibration) ----------
    n_cases, L = (150, 99) if args.quick else (400, 99)
    th_c, y_c = simulate(n_cases, rng)
    S = sampler.sample(np.repeat(y_c, L, axis=0), 100, rng, "sde").reshape(n_cases, L, 2)
    stats = {"theta_x": lambda a: a[..., 0], "theta_y": lambda a: a[..., 1],
             "dist_to_origin": lambda a: np.linalg.norm(a, axis=-1)}
    ranks = {}
    from scipy.stats import kstest
    for name, f in stats.items():
        r = (f(S) < f(th_c)[:, None]).sum(1)
        ranks[name] = r
        ks = kstest((r + rng.uniform(size=r.shape)) / (L + 1), "uniform")
        res.setdefault("sbc", {})[name] = dict(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        print(f"  SBC {name:15s}: KS stat={ks.statistic:.3f}, p={ks.pvalue:.3f}")

    # =============================== figures ===============================
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    # Fig 1: one observation, all methods
    j = 0
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
    p = ref_ps[j].reshape(GRID_N, GRID_N)
    axs[0].imshow(p.T, origin="lower", extent=[-GRID_L, GRID_L, -GRID_L, GRID_L], cmap="Greys", aspect="equal")
    axs[0].set_title("Exact posterior (grid)")
    S_d = diff_samples["sde-250"][j]
    axs[1].scatter(*S_d.T, s=3, alpha=0.4, c="tab:blue")
    axs[1].set_title("Diffusion sampler (SDE, 250 steps)")
    ch = mcmc_chains[("rwm", 4)]
    half = ch[ch.shape[0] // 2:, j]
    for k in range(4):
        axs[2].scatter(*half[::8, k].T, s=2, alpha=0.5, label=f"chain {k + 1}")
    axs[2].set_title(f"RWM, 4 chains x {ch.shape[0]} steps")
    ch = mcmc_chains[("mala", 4)]
    half = ch[ch.shape[0] // 2:, j]
    for k in range(4):
        axs[3].scatter(*half[::8, k].T, s=2, alpha=0.5)
    axs[3].set_title(f"MALA, 4 chains x {ch.shape[0]} steps")
    lim = min(GRID_L, max(5.0, 1.15 * max(np.abs(S_d).max(), np.abs(th_star[j]).max())))
    for a in axs:
        a.scatter(*SENSORS.T, marker="^", c="red", s=40, zorder=5)
        a.scatter(*th_star[j], marker="*", c="gold", edgecolors="k", s=120, zorder=6)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    axs[2].legend(loc="lower right", fontsize=7, markerscale=3)
    fig.suptitle("Same observation, same budget order: red = sensors, star = true source", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1_posteriors.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 2: accuracy vs cost
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    best = diff_res["sde-250"]
    for ax, key, lab in [(axs[0], "sw2", "Within-mode shape error: folded sliced W2 (lower = better)"),
                         (axs[1], "mode_err", "Mode-balance error |P(theta_y>0) - 0.5|")]:
        for kind, col in [("rwm", "tab:orange"), ("mala", "tab:green")]:
            for K, ls in [(4, "-"), (100, "--")]:
                pts = [(v["budget"], v[f"{key}_mean"]) for v in mcmc_res.values() if v["kind"] == kind and v["K"] == K]
                pts.sort()
                if pts:
                    ax.plot(*zip(*pts), ls, marker="o", c=col, label=f"{kind.upper()} {K} chains")
        ax.axhline(best[f"{key}_mean"], c="tab:blue", lw=2, label="Diffusion (amortised, 0 evals per new y)")
        if key == "sw2":
            ax.axhline(floor, c="grey", ls=":", label="exact-sampler noise floor")
        ax.set_xscale("log"); ax.set_yscale("log" if key == "sw2" else "linear")
        ax.set_xlabel("forward-model evaluations spent on ONE observation (MALA step = 2)")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=7)
    axs[0].set_title(f"Diffusion needs {args.n_train:,} simulator calls once, then none")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_accuracy_vs_cost.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 3: diffusion accuracy vs NFE
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for method, col in [("sde", "tab:blue"), ("ode", "tab:purple")]:
        pts = sorted((v["nfe"], v["sw2_mean"], v["sw2_se"]) for v in diff_res.values() if v["method"] == method)
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], marker="o", c=col,
                    label="reverse SDE (Euler-Maruyama)" if method == "sde" else "probability-flow ODE (Heun)")
    ax.axhline(floor, c="grey", ls=":", label="exact-sampler noise floor")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("network evaluations per sample (NFE)"); ax.set_ylabel("Within-mode shape error (folded sliced W2)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_diffusion_vs_nfe.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 4: SBC + training loss
    fig, axs = plt.subplots(1, 4, figsize=(14, 2.9))
    for ax, (name, r) in zip(axs[:3], ranks.items()):
        ax.hist(r, bins=np.arange(-0.5, L + 1.5, (L + 1) / 10), color="tab:blue", alpha=0.8)
        ax.axhline(n_cases / 10, c="k", ls="--", lw=1)
        ax.set_title(f"SBC ranks: {name}")
    ls = model["losses"]
    k = max(1, len(ls) // 200)
    axs[3].plot(np.convolve(ls, np.ones(k) / k, mode="valid"))
    axs[3].set_title("training loss"); axs[3].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_calibration.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("done; results in", args.out)


if __name__ == "__main__":
    main()
