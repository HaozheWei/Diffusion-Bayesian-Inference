"""How much does the diffusion result vary with the TRAINING seed alone?
Same training pairs, same test observations, same samplers; only the network init / minibatch / noise draws change.

Usage (run from this folder; each trained model takes about 1.5 minutes in NumPy):
    python seed_check.py cached          # evaluate results/model.npz (create it first: python diffusion_posterior.py --stage train)
    python seed_check.py 11              # train a new model with training seed 11 and evaluate it
    python seed_check.py 11 torch        # same, but with PyTorch's default nn.Linear initialisation instead of He init
Writes seedcheck_<name>.json next to the script."""
import sys, json, numpy as np, importlib.util, time
spec = importlib.util.spec_from_file_location("dp", "diffusion_posterior.py"); dp = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
tag = sys.argv[1]                                  # 'cached' or an integer seed
INIT = sys.argv[2] if len(sys.argv) > 2 else "he"  # 'he' (NumPy script default) or 'torch' (PyTorch nn.Linear default)
if INIT == "torch":
    def _torch_style_init(self, sizes, rng):
        # nn.Linear default: weights and biases ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)); last-layer weight scaled by 0.1 as in the torch script
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            bound = 1.0 / np.sqrt(sizes[i])
            self.W.append(rng.uniform(-bound, bound, (sizes[i], sizes[i + 1])).astype(np.float32))
            self.b.append(rng.uniform(-bound, bound, sizes[i + 1]).astype(np.float32))
        self.W[-1] *= 0.1
    dp.MLP.__init__ = _torch_style_init
    tag_out = tag + "_torchinit"
else:
    tag_out = tag
th_tr, y_tr = dp.simulate(100_000, np.random.default_rng(1))
if tag == "cached":
    model = np.load("results/model.npz", allow_pickle=True)["model"].item()
    model["net"] = dp.MLP([15] + [128]*4 + [2], np.random.default_rng(0))
else:
    t0 = time.time()
    model = dp.train_model(th_tr, y_tr, 30_000, 256, 2e-3, np.random.default_rng(int(tag)), log=lambda *a: None)
    print(f"trained seed {tag} in {time.time()-t0:.0f}s, final loss {model['losses'][-500:].mean():.4f}")
sampler = dp.Sampler(model)
rng = np.random.default_rng(100)                   # fixed test set, independent of the training seed
obs, ps = [], []
while len(obs) < 12:
    th, y = dp.simulate(1, rng); p = dp.grid_posterior(y[0])
    if dp.bimodal_gap_mass(p) < 0.02: obs.append(y[0]); ps.append(p)
obs = np.array(obs); n_s = 2000
ref_q = [dp.proj_quantiles(dp.sample_grid(p, 10000, rng)) for p in ps]
out = {}
for method, n in [("sde", 100), ("sde", 250), ("ode", 50)]:
    S = sampler.sample(np.repeat(obs, n_s, 0), n, np.random.default_rng(5), method).reshape(12, n_s, 2)
    out[f"{method}-{n}"] = [float(np.mean([dp.shape_err(S[j], ref_q[j]) for j in range(12)])),
                            float(np.mean([dp.mode_balance_err(S[j]) for j in range(12)]))]
print(tag_out, json.dumps({k: [round(v[0], 4), round(v[1], 3)] for k, v in out.items()}))
json.dump(out, open(f"seedcheck_{tag_out}.json", "w"))
