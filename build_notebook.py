"""Build ladder.ipynb from ladder_core.py plus the cells below.

Usage: python build_notebook.py
Edit ladder_core.py or this file, never the notebook directly.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
core = (HERE / "ladder_core.py").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Do models' self-reports track their choices — and does that improve with scale?

**Question.** A model can be asked what it would enjoy (a *stated* preference) or made to choose between two
tasks and complete one (a *revealed* preference). Its activations can also be read with a linear probe. How
well do these three measurements agree, and how does agreement change along a ladder of model sizes from the
same family?

**Why it matters.** If self-reports are to serve as evidence about a model's preferences — for welfare
assessment or for safety monitoring — they have to track what the model actually does. Prior work documents a
stated–revealed gap in individual models (Gu et al., 2025; *Mind the Gap*, 2026) but not how it scales, and
does not compare self-reports against a white-box readout.

**Design.**
- **Models:** Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B (same family, tokenizer and chat template).
- **Tasks:** 300 tasks sampled from the canonical test split of Gilg et al. (2026) — WildChat, Alpaca, MATH,
  BailBench and STRESS-TEST — so results are comparable to published work. Identical tasks for every model.
- **Revealed:** both orders of 2,901 pairs; the model names *Task A* or *Task B* and completes it. Choice
  probabilities are read from the logits after prefilling `Task`. Utilities are fit with a Thurstonian model
  that includes an explicit **position-bias** term.
- **Stated:** 1–5 enjoyment ratings and yes/no questions, with **controls** — reversed scale anchors and the
  negated question — because at small scale these formats are dominated by artifacts.
- **Probe:** ridge regression from the last-prompt-token activation to revealed utilities, against a
  **content-only baseline** (MiniLM embeddings of the task text).

**Pre-specified before running the ladder:** probe layer at 50% depth (Gilg et al.'s best probe layer); the
primary stated measure is the mean of four standardized self-reports (forward rating, reversed rating flipped,
P(yes | enjoy), 1 − P(yes | dislike)).
""")

md(r"""
## 0. Setup

The notebook picks its mode automatically:
- **GPU available** (e.g. Colab, *Runtime → Change runtime type → T4 GPU*): measures any models whose results
  are missing, then analyses.
- **No GPU, results present** in `results/`: analysis only.
- **No GPU, no results:** a small **smoke test** (0.5B, 30 tasks) to check the code runs.
""")
code(r'''
import sys, subprocess, gc, json, time
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate"], check=False)
''')
code(r'''
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
          "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
N_TASKS, K_PAIRS = 300, 20          # each task appears in ~K_PAIRS comparisons, both orders
PROBE_FRACTIONS = (0.25, 0.5, 0.625, 0.75, 0.875)
PRIMARY_FRACTION = 0.5               # pre-specified
SAVE_TO_DRIVE = True                 # Colab runtimes are wiped on disconnect; Drive keeps results

if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = Path("/content/drive/MyDrive/ladder_results")
else:
    RESULTS = Path("results")
have_all = all((RESULTS / m.split("/")[-1] / "meta.json").exists() for m in MODELS)
MODE = "measure" if torch.cuda.is_available() else ("analyze" if have_all else "smoke")
if MODE == "smoke":
    MODELS, N_TASKS, K_PAIRS, RESULTS = MODELS[:1], 30, 6, Path("results_smoke")
DATA, FIGURES = Path("data"), Path("figures")
for p in (RESULTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
print("mode:", MODE, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("models:", [m.split("/")[-1] for m in MODELS], "| tasks:", N_TASKS, "| pairs per task:", K_PAIRS)
''')

md("## 1. Pipeline code\n\nThe full measurement code, kept in one cell so the notebook is self-contained. Skim it; the sections below call into it.")
code(core)

md("## 2. Tasks\n\nThe same tasks, in the same order, for every model. Tasks over 256 tokens are excluded (8 of 1,000).")
code(r'''
download_data(DATA)
_tok = AutoTokenizer.from_pretrained(MODELS[0])
TASKS = sample_tasks(DATA, _tok, n=N_TASKS, seed=0)
PAIRS = make_pairs(len(TASKS), K_PAIRS, seed=0)
task_df = pd.DataFrame(TASKS)
print(len(TASKS), "tasks,", len(PAIRS), "pairs ->", 2 * len(PAIRS), "revealed prompts per model")
task_df.groupby("origin").size().to_frame("n").T
''')
code("task_df.topic.value_counts().head(8).to_frame('n').T")

md("## 3. Measure each model\n\nResults are saved per model as soon as each finishes, so a Colab disconnect only loses the model in progress. Re-running skips finished models.")
code(r'''
def batch_size(name):
    if MODE == "smoke": return 8
    size = next((v for k, v in PARAMS_B.items() if f"-{k}-" in name), 7.6)
    return 32 if size <= 1.5 else 16 if size <= 3.1 else 8

def run_model(name):
    short = name.split("/")[-1]
    out = RESULTS / short
    if (out / "meta.json").exists():
        print(f"{short}: already done"); return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = load_model(name)
    bs = batch_size(name)
    L = model.config.num_hidden_layers
    layer_ids = [int(f * L) for f in PROBE_FRACTIONS]   # one per fraction, in order

    rev = measure_revealed(tok, model, TASKS, PAIRS, bs=bs);        print(f"  revealed {time.time()-t0:.0f}s")
    st = measure_stated(tok, model, TASKS, bs=bs);                  print(f"  stated   {time.time()-t0:.0f}s")
    acts = last_token_acts(tok, model, [chat_text(tok, t["text"]) for t in TASKS], layer_ids, bs=bs)
    print(f"  probe activations {time.time()-t0:.0f}s")

    spot = []                                                       # does it do the task it names?
    for i, j in PAIRS[:4]:
        ids = tok(chat_text(tok, revealed_prompt(TASKS[i]["text"], TASKS[j]["text"])),
                  return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=False)
        spot.append({"task_a": TASKS[i]["text"][:80], "task_b": TASKS[j]["text"][:80],
                     "reply": tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)})

    np.save(out / "revealed.npy", rev)
    np.savez(out / "stated.npz", **st)
    np.save(out / "acts.npy", acts.astype(np.float16))
    json.dump({"model": name, "n_layers": L, "layer_ids": layer_ids, "fractions": list(PROBE_FRACTIONS),
               "dtype": str(model.dtype), "quantized_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
               "seconds": time.time() - t0, "spot_checks": spot}, open(out / "meta.json", "w"), indent=1)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"{short}: done in {time.time()-t0:.0f}s")
''')
code(r'''
for name in MODELS:
    if MODE == "analyze":
        print(name.split("/")[-1], "-> results found, analysis only")
    else:
        print("==", name); run_model(name)
''')

md(r"""
## 4. Analysis

Everything below reads only from the saved results, so it runs without a GPU.

For each model: fit revealed utilities, then correlate each other measurement with them —
- **probe**: out-of-fold predictions of a ridge probe at the pre-specified 50%-depth layer (5-fold CV);
- **template**: the published 1–5 rating template alone;
- **composite**: the pre-specified controlled composite of four self-reports;
- **content**: out-of-fold predictions of the same ridge probe on MiniLM embeddings of the task text.

Each is computed on all 300 tasks and again **excluding the 35 harmful requests**, since a large share of
agreement could come from models simply disfavouring harmful tasks. Intervals are 95% bootstrap CIs over tasks.
""")
code(r'''
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
harm = np.array([t["topic"] == "harmful_request" for t in TASKS])
EMB = content_embeddings([t["text"] for t in TASKS])        # content-only baseline features

def cv_pred(X, y, folds=5, seed=0):
    """Out-of-fold predictions of a standardized ridge probe (alpha by inner CV)."""
    p = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13)))
        p[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return p

def boot_ci(a, b, mask, B=2000, seed=0):
    idx, rng = np.where(mask)[0], np.random.default_rng(seed)
    v = [corr(a[s], b[s]) for s in (rng.choice(idx, len(idx)) for _ in range(B))]
    return np.percentile(v, [2.5, 97.5])

def analyze(short):
    d = RESULTS / short
    rev, st = np.load(d / "revealed.npy"), dict(np.load(d / "stated.npz"))
    acts, meta = np.load(d / "acts.npy").astype(np.float32), json.load(open(d / "meta.json"))
    n, half = len(TASKS), len(rev) // 2
    mu, beta = fit_thurstonian(n, rev[:, 0].astype(int), rev[:, 1].astype(int), rev[:, 2])
    _, rel = split_half_reliability(n, rev)
    parts = {"rate_fwd": z(st["rate_fwd"]), "rate_rev (flipped)": z(6 - st["rate_rev"]),
             "yes_enjoy": z(st["yes_enjoy"]), "1 - yes_dislike": z(1 - st["yes_dislike"])}
    meas = {"probe": cv_pred(acts[:, meta["fractions"].index(PRIMARY_FRACTION)], mu),
            "template": st["rate_gilg"], "composite": sum(parts.values()) / 4, "content": cv_pred(EMB, mu)}
    return {"short": short, "params": next(v for k, v in PARAMS_B.items() if f"-{k}-" in meta["model"]),
            "meta": meta, "st": st, "rev": rev, "mu": mu, "beta": beta, "reliability": rel, "parts": parts,
            "meas": meas, "probe_by_depth": {f: corr(cv_pred(acts[:, k], mu), mu) for k, f in enumerate(meta["fractions"])},
            "swap": np.mean((rev[:half, 2] > .5) == (rev[half:, 2] < .5))}

done = [m.split("/")[-1] for m in MODELS if (RESULTS / m.split("/")[-1] / "meta.json").exists()]
A = [analyze(s) for s in done]
ALL, BENIGN = np.ones(len(TASKS), bool), ~harm
P = np.array([a["params"] for a in A]); LBL = [a["short"].split("-")[1] for a in A]
print("analysed:", LBL)
''')

md("### Main table")
code(r'''
cohen = lambda x: (x[harm].mean() - x[~harm].mean()) / x.std()
table = pd.DataFrame([{
    "model": l, "mass_AB": a["rev"][:, 3].mean(), "P_first": a["rev"][:, 2].mean(), "beta": a["beta"],
    "swap_survival": a["swap"], "reliability": a["reliability"],
    "anchor_ctrl": corr(a["st"]["rate_fwd"], a["st"]["rate_rev"]),
    "polarity_ctrl": corr(a["st"]["yes_enjoy"], a["st"]["yes_dislike"]),
    **{f"r_{k}": corr(v, a["mu"]) for k, v in a["meas"].items()},
    "harm_d_revealed": cohen(a["mu"]), "harm_d_stated": cohen(a["meas"]["composite"]),
} for l, a in zip(LBL, A)]).set_index("model")
table.round(3)
''')
md(r"""
- **`mass_AB`** — probability on valid answers (`A`/`B`). Near 1 means the model follows the format.
- **`P_first`, `beta`** — position bias. `P_first` = 0.5 and `beta` = 0 mean none.
- **`swap_survival`** — share of pairs whose winner survives swapping the order.
- **`reliability`** — split-half reliability of revealed utilities (Spearman–Brown). Its square root is the
  ceiling on any correlation with them.
- **`anchor_ctrl`, `polarity_ctrl`** — controls on the stated formats. **Both should be negative**; positive
  means the format is answered by token preference rather than meaning.
- **`r_*`** — correlation of each measurement with revealed utilities.
- **`harm_d_*`** — Cohen's d of harmful requests vs. the rest (negative = harmful tasks dispreferred).
""")

md("### Headline: agreement with revealed preferences, by model size")
code(r'''
from matplotlib.ticker import NullFormatter, NullLocator

def logx(ax):
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(P); ax.set_xticklabels(LBL); ax.set_xlim(0.38, 10); ax.grid(alpha=.3)
    ax.set_xlabel("Qwen2.5-Instruct model size")

STY = {"probe": dict(c="#1f6fb4", marker="o", ls="-", label="linear probe on activations"),
       "template": dict(c="#d9822b", marker="^", ls="-", label="self-report: 1–5 rating (published template)"),
       "composite": dict(c="#d9822b", marker="s", ls="--", label="self-report: controlled composite (pre-specified)"),
       "content": dict(c="0.55", marker="x", ls=":", label="content-only baseline (MiniLM)")}
R = {(k, s): np.array([corr(a["meas"][k][m], a["mu"][m]) for a in A]) for k in STY for s, m in (("all", ALL), ("benign", BENIGN))}
CI = {(k, s): np.array([boot_ci(a["meas"][k], a["mu"], m) for a in A]) for k in STY for s, m in (("all", ALL), ("benign", BENIGN))}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
for ax, s, title in zip(axes, ("all", "benign"), (f"All tasks (n = {len(TASKS)})", f"Excluding harmful requests (n = {BENIGN.sum()})")):
    ax.fill_between(P, 0, np.sqrt(np.clip([a["reliability"] for a in A], 0, 1)), color="0.92",
                    label="noise ceiling (√ split-half reliability)")
    for k, sty in STY.items():
        y, ci = R[k, s], CI[k, s]
        ax.errorbar(P, y, yerr=[y - ci[:, 0], ci[:, 1] - y], capsize=3, lw=1.6, ms=6, **sty)
    ax.axhline(0, c="k", lw=.6); ax.set_title(title); logx(ax)
axes[0].set_ylabel("correlation with revealed preference"); axes[0].set_ylim(-0.3, 1.0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Self-reports catch up with revealed preferences only at scale; a linear probe reads them at every size", fontsize=11)
plt.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIGURES / "headline.png", dpi=180, bbox_inches="tight"); plt.show()
''')
code(r'''
rows = {}
for k in STY:
    for s in ("all", "benign"):
        rows[f"{k} ({s})"] = [f"{r:.2f} [{lo:.2f}, {hi:.2f}]" for r, (lo, hi) in zip(R[k, s], CI[k, s])]
rows["probe − content (all)"] = []
for a in A:
    rng = np.random.default_rng(0)
    v = [corr(a["meas"]["probe"][s], a["mu"][s]) - corr(a["meas"]["content"][s], a["mu"][s])
         for s in (rng.integers(0, len(TASKS), len(TASKS)) for _ in range(2000))]
    d = corr(a["meas"]["probe"], a["mu"]) - corr(a["meas"]["content"], a["mu"])
    rows["probe − content (all)"].append(f"{d:.2f} [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
pd.DataFrame(rows, index=LBL).T
''')

md("### Do the measurement formats pass their controls?")
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
axes[0].plot(P, table.anchor_ctrl, "o-", label="reversed scale anchors")
axes[0].plot(P, table.polarity_ctrl, "s-", label="negated question (enjoy vs dislike)")
axes[0].axhline(0, c="k", lw=.6); axes[0].set(ylim=(-1, 1.05), ylabel="correlation (valid format: < 0)",
                                              title="Do self-report formats pass their controls?")
axes[0].legend(fontsize=8, loc="lower left")
axes[1].plot(P, table.P_first, "o-", label="P(choose task shown first)")
axes[1].plot(P, table.swap_survival, "^-", label="winner survives order swap")
axes[1].plot(P, table.reliability, "s-", label="split-half reliability of utilities")
axes[1].axhline(.5, c="grey", ls="--", lw=.8); axes[1].set(ylim=(0, 1.05), title="Revealed choice: position bias vs. reliability")
axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
for a in axes: logx(a)
plt.tight_layout(); fig.savefig(FIGURES / "artifacts.png", dpi=180, bbox_inches="tight"); plt.show()
''')
md("### Stated composite, component by component\n\nWhy the pre-specified composite trails the single published template: two of its four components fail their controls at most sizes.")
code(r'''
pd.DataFrame({l: {k: corr(v, a["mu"]) for k, v in a["parts"].items()} for l, a in zip(LBL, A)}).T.round(3)
''')
md("### Probe by depth (exploratory — the 50% layer was the pre-specified one)")
code("pd.DataFrame({l: a['probe_by_depth'] for l, a in zip(LBL, A)}).round(3)")
md("### Spot checks: does the model perform the task it names?")
code(r'''
for l, a in zip(LBL, A):
    print("=====", l)
    for c in a["meta"]["spot_checks"]:
        print(f"  A: {c['task_a'][:60]!r}\n  B: {c['task_b'][:60]!r}\n  -> {c['reply'][:140]!r}\n")
''')
code(r'''
table.assign(**{f"{k}_{s}": R[k, s] for k in STY for s in ("all", "benign")}).to_csv(FIGURES / "summary.csv")
print("saved", FIGURES / "headline.png", FIGURES / "artifacts.png", FIGURES / "summary.csv")
''')

md(r"""
## 5. Caveats

- Revealed choices are read from logits after the model names a task, rather than parsed from full
  completions; spot checks above verify that the named task is the one performed.
- Choice probabilities are used as soft labels, where Gilg et al. sample discrete choices with active learning.
  This matters: at 1.5B the first-listed task wins 99% of the time and no pair's winner survives an order swap,
  yet the soft probabilities still yield reliable utilities.
- At 1.5B and 7B, position bias (β ≈ 2.6–2.8) sends most pairs to whichever task is listed first, so each
  task's utility rests on the minority of pairs where preference overrides position. Utilities remain
  reliable (split-half 0.93–0.97), but the effective number of comparisons is below the nominal 20 per task.
- The Thurstonian fit is simplified: one shared noise level for all tasks (Gilg et al. fit one per task),
  fit to probabilities rather than sampled choices, plus an added position-bias term.
- 7B is loaded in 8-bit to fit a T4 (see `meta.json`); the other sizes run in fp16.
- The probe vs. self-report comparison is not a fair fight: the probe learns from ~240 labelled tasks per fold,
  the self-report is zero-shot. The claim is that preference information is *present* in the activations,
  not that the model *cannot* report it.
- When a rating format ignores its anchors (forward and reversed ratings agree), its two terms cancel in the
  stated composite. This is intended: a format that fails its control contributes no signal.
- One model family; 300 tasks is a small fraction of Gilg et al.'s 10k.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"]["accelerator"] = "GPU"
nbf.write(nb, HERE / "ladder.ipynb")
print("wrote ladder.ipynb with", len(cells), "cells")
