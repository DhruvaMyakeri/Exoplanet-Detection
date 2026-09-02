# Exoplanet Detection System — Project Handoff

**Read this fully before writing any code.** Stages 0–2 are built and validated. Your job is to finish Stages 2–4. There is a mandatory pause point in Stage 2 described under "STOP CONDITION" — respect it.

---

## 0. What this project is

Build a machine learning system that decides whether a periodic dip in a star's brightness is caused by an orbiting planet or by something else.

**Working directory:**`D:\PROJECTS\exoplanet`**Hardware:** RTX 4060 (8 GB VRAM), 16 GB system RAM, Windows, PowerShell. **Python:** 3.10, global install at `C:\Users\Lenovo\AppData\Local\Programs\Python\Python310`. Do NOT use the `.venv` in the project folder — it is broken (its `pip` resolves to the global install, so it is permanently empty). Use the global Python.

Installed and verified: `lightkurve 2.6.0`, `h5py 3.16.0`, `scikit-learn 1.7.2`, `numpy 2.2.6`, `pandas 2.3.3`, `matplotlib 3.10.8`. PyTorch is NOT yet installed. Stage 3 needs it.

---

## 1. The physics, in brief

A planet cannot be imaged directly — its star is \~10⁹ times brighter. Instead we watch for **transits**: when a planet crosses in front of its star, the star dims by the ratio of disc areas.

```
depth = (R_planet / R_star)²

  Jupiter across a Sun-like star:  (69911/696000)²  = 1.0%
  Earth   across a Sun-like star:  (6371/696000)²   = 84 ppm
```

Kepler sampled \~150,000 stars every **29.4 minutes** for **4 years** (\~65,000 measurements per star). A transit lasts hours, so <1% of the array is signal.

Three obstacles, each with a fix that creates the next problem:

1. **Starspots** cause 0.1–5% brightness waves — 10–1000× larger than the transit. → **Detrend** with a Savitzky–Golay filter. New problem: if the filter window is near the transit duration, the filter eats the transit. Window must be ≫ duration (we use 2.06 days vs 2–6 hours).
2. **A single transit is marginal.** → **Fold** on the orbital period so all transits stack. SNR grows as √N. New problem: you don't know the period.
3. **Finding the period** → **Box Least Squares (BLS)**: brute-force grid search over period × phase × duration, fitting a literal rectangle. "BLS power" is how much better a box explains the data than a flat line. New problem: BLS reports a candidate on nearly every star.

**The ML task is therefore VETTING, not detection.** BLS already found the dip. The question is which dips are planets.

### What produces false positives

| Impostor                                                            | The tell                                             |
| ------------------------------------------------------------------- | ---------------------------------------------------- |
| Eclipsing binary (two stars)                                        | Depth 10–50%, far too deep                           |
| **Grazing**eclipsing binary                                         | Planet-like depth, but**V-shaped**— no flat bottom   |
| Binary detected at half its true period                             | Odd and even numbered dips have**different depths**  |
| Blend (faint background binary diluted by a bright foreground star) | Needs centroid data; hard from the light curve alone |

Two of four tells are **shape-based**. That is why a CNN on the curve can beat a tree model on summary statistics — and it is the central hypothesis of the project.

---

## 2. Data model — four distinct objects

Do not conflate these.

| Object          | Example       | Count          | What it is                                                                                        |
| --------------- | ------------- | -------------- | ------------------------------------------------------------------------------------------------- |
| **Star**        | `KIC 1026957` | 6,641          | A physical star in the Kepler field                                                               |
| **KOI**         | `K00252.01`   | 7,587          | A*candidate signal*on a star. Carries the label. Suffix`.01/.02`numbers multiple signals per star |
| **Light curve** | —             | 1 per star     | \~65,000 rows of`(time_BKJD, flux)`. Arrives as \~15 FITS files, \~1 MB each                      |
| **View**        | —             | 1 pair per KOI | Fixed-length binned array; what the network eats                                                  |

KOIs outnumber stars because multi-planet systems exist. **One light curve is folded at several different periods, once per KOI on that star.** This is why the downloader groups by `kepid`.

**Labels:**`CONFIRMED` → 1, `FALSE POSITIVE` → 0. `CANDIDATE` is NOT a third class — it means "not yet adjudicated." Exclude it from training; score it at the end as a held-out prediction set.

**Time system:**`koi_time0bk` is BKJD = BJD − 2454833, barycentre-corrected. lightkurve reports Kepler times in BKJD too, so they align directly. This has been verified (see §4).

---

## 3. Current state — files on disk

| File                 | Status                      | Purpose                                                        |
| -------------------- | --------------------------- | -------------------------------------------------------------- |
| `koi_stage1.py`      | Done, working               | Stage 0 Random Forest on catalogue features                    |
| `refetch_koi.py`     | Done                        | Pulls`koi_ephem.csv`from the NASA Exoplanet Archive            |
| `koi_ephem.csv`      | 9,564 rows                  | KOI table**with** `koi_time0bk`. Use this one                  |
| `koi_cumulative.csv` | Legacy                      | Stage 0's cache.**Lacks`koi_time0bk`**— do not use for Stage 2 |
| `lc_stage1.py`       | Done, working               | Single-star tutorial: download → detrend → BLS → fold → plot   |
| `build_views.py`     | Done, validated on 50 stars | **Stage 2 pipeline.**Needs the full run                        |
| `inspect_views.py`   | Done                        | Diagnostics on`views.h5`                                       |
| `views.h5`           | Partial (45 views)          | Delete before the full run                                     |

### `views.h5` schema

All datasets are row-aligned; index `i` is the same KOI in every one.

```
global     (N, 2001) float32   whole folded orbit, phase −0.5 … +0.5
local      (N,  201) float32   ±2.5 transit durations around the transit
label      (N,)      int8      1 = CONFIRMED, 0 = FALSE POSITIVE
name       (N,)      str       "K00252.01"
kepid      (N,)      int64     host star — REQUIRED for group splitting
disp       (N,)      str       raw disposition text
period     (N,)      float32   days
duration   (N,)      float32   hours
depth      (N,)      float32   ppm, from the catalogue
npts       (N,)      int32     cadences that entered the fold
snr        (N,)      float32   dip depth ÷ out-of-transit scatter
```

Expected final size: \~7,500 rows, \~67 MB.

### Why two views

- **Global (2001 bins over the full orbit):** fixed _phase_ resolution. Its job is context — a secondary eclipse at phase 0.5 means a binary.
- **Local (201 bins over ±2.5 durations):** fixed _duration_ resolution. Its job is shape — U vs V.

Why both are needed: a 300-day-period planet with a 10-hour transit occupies **under 3 bins** in the global view. Every shape cue is gone. The local view guarantees a 3-hour and a 10-hour transit produce the same number of in-transit bins.

### Why depth is normalised away

Each view is scaled so baseline = 0 and transit depth = −1. This is deliberate. Depth is a scalar that a tree model already handles perfectly (Stage 0 proves it). Leaving it in lets the CNN latch onto "deep = binary" and never learn shape. Depth is stored separately in `views.h5` and can be fed to a final dense layer.

---

## 4. Verified findings — do not re-litigate these

Each was established empirically in this environment. Treat as settled.

**Leakage.**`koi_fpflag_nt/ss/co/ec` and `koi_score` are OUTPUTS of the vetting process that produced the labels. Training on them gives PR-AUC 0.998 and makes the model **ignore every physical feature** (all permutation importances collapse to \~0). Never include them. Honest Stage 0 result: **PR-AUC 0.947**, ROC-AUC 0.976, **72.5% recall at 95% precision**, on a 0.35 base rate.

**Group splitting.** Split by `kepid`, not by row. Multi-planet systems put several KOIs on one star; a random row split leaks that star's systematics and stellar parameters across the train/test boundary. Use `GroupShuffleSplit`.

**Derived feature that works.**`log_dur_ratio` — the ratio of observed transit duration to the duration predicted from stellar density — ranked **2nd** in permutation importance (+0.056 ± 0.010), ahead of transit SNR. Derivation:

```
(a/R*)³ = G·ρ*·P² / (3π)          from Kepler's 3rd law
T_exp   = (P/π) · (3π / (G·ρ*·P²))^(1/3)
ρ*      = 3g / (4πGR)              from log g and stellar radius
```

Sanity check: solar density, 1-year period → 13.0 h, which is Earth's actual transit duration. Validated again against real photometry in Stage 1 (below).

**Transit masking during detrending is required.** Verified on K00002.01 (catalogue depth 6,675 ppm) with lightkurve's internal sigma clipping disabled:

```
no mask, clip OFF   5,807 ppm     ← filter ate 13% of the transit
masked,  clip OFF   6,453 ppm     ← mask alone fully recovers it
```

**In lightkurve 2.6.0, `flatten(mask=...)` with `mask=True` means EXCLUDE from the trend fit.** With default clipping ON (`niters=3, sigma=3`) the two are indistinguishable, because deep transits are 30–60σ outliers and get clipped automatically. The mask still matters for ingress/egress points and shallow transits, where clipping is unreliable.

**Stage 1 end-to-end validation (Kepler-10):**

```
recovered period    0.837491 d   vs published 0.837491 d   (exact)
recovered duration  1.80 h       vs 1.88 h predicted from stellar density (4%)
recovered depth     161 ppm      vs 160 ppm from (Rp/R*)² = (1.47 Re / 1.065 Rsun)²
per-point scatter   93 ppm       → transit is 1.6σ per cadence
stacked SNR         ~118σ        from ~1,270 folded orbits
```

**BLS grid sizing.** Astropy spaces the frequency grid as `δν = frequency_factor × d_min / T_baseline²`. With a 1,470-day baseline and `frequency_factor=1.0` this produces **194 million** trial periods and raises a ValueError. Use a two-pass search: coarse scan with `frequency_factor≈1500`, then a dense `np.linspace` grid over ±0.2% around the coarse peak. The T² scaling exists because period error accumulates over `T/P` cycles; once the drift exceeds one transit duration the stack smears out.

**BLS duration grid quality propagates into period accuracy.** With a coarse duration grid (0.02–0.15 d) the fit returned duration = 0.1 d exactly (a grid entry), depth 129 ppm (23% low), and period off by 7e-6. Refining the duration grid to 0.015-day spacing fixed all three simultaneously — an oversized box shifts its own best-fit centre, which biases the epoch, which biases the period.

**Threading vs processes.** Astroquery's download path closes `sys.stdout` on completion. With `ThreadPoolExecutor` all workers share one `sys.stdout`, so one thread's teardown breaks the other seven — 26/50 stars failed with `ValueError: I/O operation on closed file`. Rebinding `sys.stdout` does NOT fix it, because astroquery holds a reference to the original handle captured at import. **The fix is `ProcessPoolExecutor`** — each worker gets its own stdout. After this change: **0/50 failures.** Do not revert to threads.

**Fold alignment is correct.** Median offset of the local-view minimum from the centre bin is **4 bins out of 201**. A wrong epoch would give \~50. Broken down by SNR (random baseline = 10%):

```
SNR 0–3    n=25   64% centred
SNR 3–7    n= 8   62% centred
SNR 7–100  n= 8   88% centred
```

Even the noisiest bucket is 6× above chance. The overall 71% figure is an SNR effect, not a fold bug.

**Median SNR is 2.9.** Over half of all KOIs have dips barely above their own noise. This is a property of the catalogue, not a defect. Expect Stage 4 results to vary sharply with SNR — report metrics **stratified by SNR**, not pooled.

---

## 5. Stage 2 — YOUR FIRST TASK

`build_views.py` is written and validated. Run it on the full catalogue.

```powershell
cd D:\PROJECTS\exoplanet
del views.h5
del build_views.log
python build_views.py --workers 12
```

Monitor from a second terminal:

```powershell
Get-Content build_views.log -Tail 5
```

**Timing is uncertain.** Trial runs gave 6.4 s/star (8 workers) and later 14.1 s/star under identical settings — likely MAST server-side throttling. Full run is therefore **12–26 hours**. The job is resumable: if it crashes or is interrupted, rerun the same command and it skips KOIs already in `views.h5`.

### STOP CONDITION — mandatory

**While the download is running, do not proceed to Stage 3. Do not start other work. Report to the user and wait.**

Report at these moments:

1. **Immediately after launch** — confirm it started, give the observed seconds/star from the first progress line, and the projected ETA.
2. **If the rate exceeds 20 s/star** — stop and tell the user. Present the tradeoff rather than deciding alone: limiting `download_all()` to the first \~8 of \~15 quarters roughly halves download time, but reduces stacked SNR by ≈ √(8/15) ≈ 0.73, hitting long-period objects hardest. Given median SNR is already 2.9, this is a real data-quality cost. **The user decides.**
3. **If failures exceed 5% of stars processed** — stop and report the error text. 0/50 failed in the trial, so any systematic failure is new.
4. **On completion** — report the final `[done]` line: total views, planet/FP split, failure count, file size.

Expected on completion: roughly **7,000–7,500 views**, \~36% labelled planet, \~67 MB.

### After completion, before Stage 3

```powershell
python inspect_views.py
```

Confirm: median fold offset still ≈ 4 bins; drop rate is stated; the SNR breakdown still shows the high bucket well above the low bucket. Report the drop rate **broken down by disposition** — if drops are class-biased, the training set is biased and that must be disclosed in the final write-up, not silently absorbed.

---

## 6. Stage 3 — the CNN

Only begin after the user confirms Stage 2 output.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Architecture** (after Shallue & Vanderburg 2018, "Astronet"): two independent 1-D convolutional stacks, one per view, concatenated, then dense layers.

```
global (B, 1, 2001) ─→ conv stack ─→ flatten ─┐
                                              ├─→ concat ─→ FC ─→ sigmoid
local  (B, 1,  201) ─→ conv stack ─→ flatten ─┘
```

Suggested global branch: 5 blocks of `[Conv1d(k=5) ×2, MaxPool(2)]` with widths 16→32→64→128→256. Local branch: 2 such blocks, widths 16→32. Then two dense layers of 512 with dropout. This is a starting point, not a prescription — tune it.

**Requirements:**

- Split with `GroupShuffleSplit` on `kepid`. Same discipline as Stage 0. A random row split invalidates every number you produce.
- Hold out a **test** set that is touched exactly once, at the very end. Use a separate validation split for model selection.
- Class weighting or balanced sampling — the split is roughly 36/64.
- Augmentation: random horizontal reflection of both views (a transit is time-symmetric, so this is label-preserving). Consider small phase jitter.
- Report **PR-AUC** as the headline. Accuracy is misleading at this base rate.
- Expect well under 2 GB VRAM and a few minutes per epoch on a 4060. The GPU is not the constraint; if training is slow, the bottleneck is the data loader.
- Set seeds and report results across ≥3 seeds. With \~7,500 examples, run-to-run variance will be substantial and a single number is not a result.

**Baseline to beat:** Stage 0's PR-AUC of 0.947 from catalogue features alone. If the CNN does not beat it, say so plainly — that is a legitimate finding about whether shape adds information beyond summary statistics, and it is more interesting than a fabricated win.

---

## 7. Stage 4 — honest evaluation

This is where the project becomes a result rather than a demo.

1. **Stratify by SNR.** Median SNR is 2.9. Report PR-AUC separately for SNR <3, 3–7, and >7. A single pooled number hides everything that matters.
2. **Confound check.** If performance correlates strongly with SNR, the model may be learning "loud signal = planet" rather than shape. This is the neural-network analogue of the Stage 0 leakage lesson. Test it explicitly: train on a fixed narrow SNR band and see whether the advantage survives.
3. **CNN vs Random Forest.** Same group split, same test set. Where do they disagree? Inspect those cases individually and plot them. Disagreements are more informative than the aggregate.
4. **Odd–even auxiliary task (optional).** Build separate folded views from odd-numbered and even-numbered transits. A depth difference indicates a binary at twice the assumed period. This is a known high-value vetting signal that neither current view can express.
5. **Score the 1,977 CANDIDATE objects.** Run `build_views.py --include-candidates` to generate their views, then predict.

**Known result to compare against:** the Stage 0 Random Forest, using physics only, scored 107 candidates at p ≥ 0.9 and 639 at p ≤ 0.1.

**Interpret this carefully — do not overclaim.**`CONFIRMED` objects are systematically larger, brighter and higher-SNR than `CANDIDATE` objects, because confirmation requires follow-up observations that are easier for strong signals. So a model trained on CONFIRMED-vs-FP may be reporting "small and faint" while appearing to report "false positive." Quantify the shift by comparing the radius and SNR distributions of the three dispositions before drawing any conclusion.

The correct framing for a write-up is **not** "the model found 107 new planets." It is: "a physics-only classifier disagrees sharply with flag-based vetting on the undecided population, and here is the distribution shift that explains part of it."

---

## 8. Stage 5 — optional, blind search

Everything above is vetting: the signal was already found by NASA's pipeline. Genuine end-to-end discovery means running your own BLS over stars with no known KOI, then feeding the candidates to the CNN.

Reuse the two-pass BLS from `lc_stage1.py`. It will probably find nothing — Kepler data has been examined for over a decade. **Reporting a null result with the sensitivity limit quantified is a legitimate and honest outcome**, and more valuable than most positive claims.

---

## 9. Working principles

The user is learning this material, not just collecting code. Explanations matter as much as output.

- **Derive, don't assert.** Show the chain: goal → obvious approach → why it breaks → the fix → the new problem the fix creates.
- **Ground every claim in real numbers** from actual runs. Never invent output. If you have not run something, say so.
- **Separate clearly:** what is _established_, what is _argued_, what was _measured here_, and what that measurement actually licenses.
- **Make predictions before running things**, with numbers, and record when they are wrong. Several predictions in this project were wrong (the transit-vs-noise ratio on Kepler-10, the fold-centring threshold, the depth after fixing the duration grid) and each error taught something the correct guess would not have.
- **Silent failures are the enemy.** A filter that eats the signal, a phase shift that produces noise arrays, a leaked column — none of these raise an exception. Build the check before you need it.
- Do not turn a teaching explanation into an audit. Correct, then move on.
