#!/usr/bin/env python3
"""
PyTorch port of the conditional diffusion model in diffusion_posterior.py.

STATUS: run on CPU and checked against the NumPy implementation. With the He initialisation below, converged
sampler errors agree with NumPy to within training-seed variation (see "PyTorch port check" and "Robustness
checks" in the README). Expected for the 100-step SDE sampler: shape error about 0.012 to 0.014 and mode-balance
error about 0.01.

It reuses the problem definition, exact grid posterior, metrics and plotting from diffusion_posterior.py,
and writes figures, results.json and the trained model to ./results_torch.

    python diffusion_posterior_torch.py                 # full run (under a minute on CPU)
    python diffusion_posterior_torch.py --steps 3000    # quick look
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import diffusion_posterior as dp

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FREQS = torch.tensor(dp.FREQS, dtype=torch.float32)


def alpha_sigma(t):
    """VP-SDE marginal: x_t = alpha(t) x_0 + sigma(t) eps."""
    log_a = -0.5 * (dp.BETA_MIN * t + 0.5 * (dp.BETA_MAX - dp.BETA_MIN) * t ** 2)
    a = torch.exp(log_a)
    return a, torch.sqrt(1.0 - a ** 2)


def time_features(t):
    ang = torch.pi * t[:, None] * FREQS.to(t.device)[None, :]
    return torch.cat([t[:, None], torch.sin(ang), torch.cos(ang)], 1)          # (N, 11)


class EpsNet(nn.Module):
    """Predicts the noise eps from (x_t, observation y, time t). Input dim = 2 + 2 + 11."""

    def __init__(self, width=128, depth=4):
        super().__init__()
        layers, d = [], 15
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        # He (Kaiming) normal init and zero biases, to match the NumPy version. PyTorch's default nn.Linear init
        # has about 6x smaller weight variance, which measurably worsened the converged sampler error
        # (see "Robustness checks" in the README).
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)

    def forward(self, x, y_n, t):
        return self.net(torch.cat([x, y_n, time_features(t)], 1))


def train(theta, y, steps=30_000, batch=256, lr=2e-3, seed=0):
    """Denoising score matching (epsilon-prediction) on simulated (theta, y) pairs."""
    torch.manual_seed(seed)
    th = torch.tensor(theta / dp.THETA_SCALE, dtype=torch.float32, device=DEV)
    y_mu, y_sd = y.mean(0), y.std(0)
    yn = torch.tensor((y - y_mu) / y_sd, dtype=torch.float32, device=DEV)
    net = EpsNet().to(DEV)
    ema = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * it / steps)))
    t0 = time.time()
    losses = []
    for it in range(1, steps + 1):
        idx = torch.randint(0, th.shape[0], (batch,), device=DEV)
        t = dp.T_MIN + (1.0 - dp.T_MIN) * torch.rand(batch, device=DEV)
        a, s = alpha_sigma(t)
        eps = torch.randn(batch, 2, device=DEV)
        x_t = a[:, None] * th[idx] + s[:, None] * eps
        loss = ((net(x_t, yn[idx], t) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        with torch.no_grad():                                  # exponential moving average of the weights
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(p, alpha=0.001)
        if it % 2000 == 0 or it == 1:
            print(f"  step {it:6d}/{steps}  loss {np.mean(losses[-500:]):.4f}  ({time.time() - t0:.0f}s)")
    return ema.eval(), y_mu, y_sd, np.array(losses)


@torch.no_grad()
def sample(ema, y_mu, y_sd, y, n_steps, method="sde"):
    """Draw one posterior sample per row of y (N, 2). method: 'sde' (Euler-Maruyama) or 'ode' (Heun)."""
    yn = torch.tensor((y - y_mu) / y_sd, dtype=torch.float32, device=DEV)
    x = torch.randn(y.shape[0], 2, device=DEV)
    ts = dp.T_MIN + (1.0 - dp.T_MIN) * np.linspace(1.0, 0.0, n_steps + 1) ** 2      # denser near t=0

    def beta(t):
        return dp.BETA_MIN + t * (dp.BETA_MAX - dp.BETA_MIN)

    def score(x_, t):
        tt = torch.full((x_.shape[0],), float(t), device=DEV)
        _, s = alpha_sigma(tt)
        return -ema(x_, yn, tt) / s[:, None]

    for i in range(n_steps):
        t, t_next = float(ts[i]), float(ts[i + 1])
        h = t - t_next
        sc, b = score(x, t), beta(t)
        if method == "sde":                                   # reverse-time SDE
            x = x + h * (0.5 * b * x + b * sc) + (b * h) ** 0.5 * torch.randn_like(x)
        else:                                                 # probability-flow ODE, Heun's method
            d1 = 0.5 * b * x + 0.5 * b * sc
            x_e = x + h * d1
            if i < n_steps - 1:
                d2 = 0.5 * beta(t_next) * x_e + 0.5 * beta(t_next) * score(x_e, t_next)
                x = x + 0.5 * h * (d1 + d2)
            else:
                x = x_e
    tt = torch.full((x.shape[0],), float(ts[-1]), device=DEV)       # final Tweedie denoising step
    a, s = alpha_sigma(tt)
    x0 = (x - s[:, None] * ema(x, yn, tt)) / a[:, None]
    return (x0 * dp.THETA_SCALE).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--train_seed", type=int, default=0,
                    help="seeds the network init, minibatches and noise only; the training data and test set stay fixed")
    ap.add_argument("--out", default="results_torch", help="folder for figures, results.json and the model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    theta, y = dp.simulate(args.n_train, rng)
    print(f"device: {DEV}; training on {args.n_train} simulated pairs for {args.steps} steps")
    t0 = time.time()
    ema, y_mu, y_sd, losses = train(theta, y, args.steps, seed=args.train_seed)
    train_seconds = time.time() - t0
    torch.save({"ema": ema.state_dict(), "y_mu": y_mu, "y_sd": y_sd}, os.path.join(args.out, "model.pt"))

    obs, th_star, ref_ps = [], [], []                          # strongly bimodal test observations
    while len(obs) < args.n_test:
        th, yy = dp.simulate(1, rng)
        p = dp.grid_posterior(yy[0])
        if dp.bimodal_gap_mass(p) < 0.02:
            obs.append(yy[0]); th_star.append(th[0]); ref_ps.append(p)
    obs, th_star, n_s = np.array(obs), np.array(th_star), 2000
    n_y = len(obs)
    ref_q = [dp.proj_quantiles(dp.sample_grid(p, 10_000, rng)) for p in ref_ps]
    floor = float(np.mean([dp.shape_err(dp.sample_grid(p, n_s, rng), ref_q[j]) for j, p in enumerate(ref_ps)]))
    print(f"exact-sampler noise floor (folded SW2) = {floor:.4f}")

    samples, rows = {}, []
    for method, n_steps in [("sde", 25), ("sde", 100), ("sde", 250), ("ode", 10), ("ode", 25), ("ode", 50)]:
        S = sample(ema, y_mu, y_sd, np.repeat(obs, n_s, axis=0), n_steps, method).reshape(n_y, n_s, 2)
        sw = [dp.shape_err(S[j], ref_q[j]) for j in range(n_y)]
        mb = [dp.mode_balance_err(S[j]) for j in range(n_y)]
        nfe = n_steps + 1 if method == "sde" else 2 * n_steps       # same accounting as the NumPy script
        rows.append(dict(method=method, n_steps=n_steps, nfe=nfe, shape_mean=float(np.mean(sw)),
                         shape_se=float(np.std(sw, ddof=1) / np.sqrt(n_y)), mode_mean=float(np.mean(mb))))
        samples[f"{method}-{n_steps}"] = S
        print(f"  {method}-{n_steps:<4d} NFE={nfe:4d}  shape-err={np.mean(sw):.4f}  mode-balance err={np.mean(mb):.3f}")

    files = dp.plot_diffusion_results(args.out, ref_ps, th_star, samples, rows, floor, losses, tag="(PyTorch)")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(config=vars(args), device=DEV, train_seconds=train_seconds, floor_shape_err=floor,
                       diffusion=rows), f, indent=2)
    print("figures and results written to:", args.out)
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
