## Here's how I'd make this insane (and this is where I think you'd shine)

Knowing the conversations we've had about ASTRA and JEPA...

### Upgrade 1 — Replace the CNN with a self-supervised encoder.

![Projects — Xavier Mootoo](https://images.openai.com/static-rsc-4/eytgJsY8G-AXQoCXUfifTUGzGZrHGVriCqN1dM_AQNBf-ozzDYo5SaQtWxmg3VHfYZvnZBKO69WMWBNS7XPgl2YeVmnZI3WfyTKkq2cr35e5KFGpnYVExeMnhPcavakzp0V80CllF1ziKp4iyQ7mYJNgE9IUKt3BWRiIOYTtb0w?purpose=inline)

![[Time-series] A Time Series Is Worth 64 Words: Long-Term Forecasting With Transformers (ICLR 2023, PatchTST) :: 곰곰의 일지](https://images.openai.com/static-rsc-4/bYeNoN4s5UxHLm21GrYl4ckawPCXpWwZb9ZWs2LH-TR9204jpsLZU9aeVZwf5JqX6mbLl1EmdSgMCQyStToY_FWqr0huNZvMzz0cSeHw4aX-Ksv7yo205Y0U0aVNAZyDnxHQhbeL_9z_PVT2Rihp250BFgfE30cTQD_E7X9d2gw?purpose=inline)

![HowToFindAnExoplanet-UserVersion - TESS](https://images.openai.com/static-rsc-4/Dlgh8gmDhI8nyWb2H1Sl73Cei7RzXdX6YFGCNNR0aTJsLpevYawMIREtde2yYb5YYGBIMwB38wnsi_UUEqIHm4ccWQEspqOPNLJv2CURjj1zzt7Ve_DREZ6c5quBhWBLq6fI8bQXnVTSelWQ4NJs4-srPShu7EkHe7wBZZO_IvA?purpose=inline)

**5**

Instead of supervised labels first:

```
Millions of unlabeled TESS light curves
        │
Masked reconstruction / JEPA-style objective
        │
Latent representation of stellar behavior
        │
Fine-tune on planet candidates
```

Why this is better:

- TESS has millions of stars.
- Labels are tiny.
- Stellar variability is structure.
- The model learns astrophysics before planets.

This is actually publishable territory if done well.

**Difficulty:** +3.

**Coolness:** +100.

### Upgrade 2 — Learn the period instead of searching every period.

BLS brute-forces:

P∈[0.5,500] daysP \\in [0.5,500]\\text{ days}**P**∈**[**0.5**,**500**]** days

Imagine a network predicts:

(P,t0,d,σ)(P, t_0, d, \\sigma)**(**P**,**t**0\*\***,**d**,**σ**)\*\*

directly from the light curve.

Or proposes a distribution over periods.

This becomes an amortized inference problem instead of exhaustive search.

That's a genuinely interesting research idea.

### Upgrade 3 — Multi-planet world model.

![Solar Expanse - Images & Screenshots | GameGrin](https://images.openai.com/static-rsc-4/GI3Sm90zpU02yYsx88dxNsvY9NWMFmuJ3qNziAE8oKk2C1EHtdEA0V_afGtsqd0Tert85AD_lWNoD2sAZsgl63_sFQ86n1PF4x2Au_d5bXDD52YGgaTIyD_I6YlTqgEAbyNpTXqC3GNfmcZvLLgWH1ao3Jd6IQbhBwyooxBtJX4?purpose=inline)

![Orbital Dynamics of Planetary Systems | Eric Ford](https://images.openai.com/static-rsc-4/0GFGlA5QQi7BNztlAIjhT0cZaJaZsQ0Wd9shy6g6IuhSMHVwMLXtubvoNcVgqgs6d5dCOoBJq8SntZzQim2QKUP5dTuSufrfnwN78aeDBSK1jzINB8LzFqSf1IIYbVJSlU8LWOECYUHxwS6AMUcK2vbYCzdd9QjXvudXjDPivCQ?purpose=inline)

![ESA - Detecting exoplanets with transit-timing variations](https://images.openai.com/static-rsc-4/dhhWo8mxsYbvLXPgWi81EYx8cWI-kMBl0owS40ukt4K-eCQ1g0eOXgYHJS_JYCrfOiD9DPV3F-Bp-Wn65XnjVBlWo0i8EXNHFOqNusccdTi-Xc4igi9PqD5kaPSwVkwrKPghRbqe8lwlKVpG8vyE_3Zb6HTmKTOxKqZROc4horE?purpose=inline)

**6**

Don't stop after one transit.

Predict:

- number of planets,
- orbital periods,
- radii,
- inclination,
- transit timing variations.

Now the latent state is:

z={Pi,Ri,ii,ei,…}z = \\{P_i, R_i, i_i, e_i, \\ldots\\}**z**=**{**P**i\*\***,**R**i\***\*,**i**i\*\***,**e**i\***\*,**…**}**

That's much closer to a world model.

### Upgrade 4 — Uncertainty everywhere.

Right now it's binary.

Instead output:

| **Quantity**       | **Distribution**      |
| ------------------ | --------------------- |
| Period             | Mean + variance       |
| Transit depth      | Posterior             |
| Planet probability | Calibrated confidence |

Scientists care about calibrated uncertainty way more than accuracy.

## If this landed on GitHub, here's what I'd think

Typical GitHub exoplanet project

⭐ 6/10

Your pipeline as described

⭐ 8.5–9/10

Self-supervised + uncertainty + candidate discovery

⭐ 10/10 "I'd read this README."

The difference is huge.

## What I'd cut from the document

A couple things felt too tutorial-ish.

### I'd shorten the astronomy lesson.

The sections explaining:

- why planets dim stars,
- ingress/egress,
- area ratios,

are good for learning, but they don't belong in your project README.

Readers care more about:

> **Problem → Pipeline → Results → Architecture.**

Keep the physics in docs.

### I'd spend more space on evaluation.

I want to see metrics like:

![Precision-Recall Curve - ML - GeeksforGeeks](https://images.openai.com/static-rsc-4/rAb-MSSFXQo4l7uuByFk1WN9M5AVzhsvhp-GKw2d3iPdwGWYg9xyyhbKSSwYhuuPHzJLN5zVE0P9ermowwz79K8ETrjTAJq4pstnjdN5NYv0oTWUgYS-6-y0F9wL5gol0BrETjYfoOMQ1yRLd3zOppfKDB7sGcKseS-2W_-nGt4?purpose=inline)

![Checking your browser - reCAPTCHA](https://images.openai.com/static-rsc-4/usV4tv9EU_7n9HWqKNeuNbJSuQpOLwrihwk07cUy4RVcIR9UPUiH7a2U9xTn9XB6GIXDGb60z51EmYuf-1Q-kATC0nJc29RSX9CwteK1ycMQeh_KArs0ra8v5ZaD-O-4fX-PYyMvtsLIeF4iZaef18BfiYdGCAyGYFBAE6wZYCg?purpose=inline)

![Explainable and fair anti-money laundering models using a reproducible SHAP framework for financial institutions | Discover Artificial Intelligence | Springer Nature Link](https://images.openai.com/static-rsc-4/yamZbGrX8V0tV0gHL_K2X9trcSwNQeuaFJ8D0McUsThphE7UKKAAFo9Rcp-bNAe04XCnwqylihrQPwiKZPpcaTfoVCHKvMx8Duzt0L9jO8ImaeAn_hhwq7oPG8E851iUOQaWnhLPNN6q12QZPAaUUS7KNqzrjM4wB1f-L9mHQS4?purpose=inline)

**5**

- ROC-AUC
- PR-AUC (more important because positives are rare)
- Recall at fixed false-positive rate.
- Calibration curve.
- Confusion matrix by planet size.

That screams "I know ML."

## What I'd add that almost nobody adds

### An interactive candidate explorer.

![Software: LATTE — Nora Eisner](https://images.openai.com/static-rsc-4/PolgKT9PMy4KVNray6uQVEo5B_sNpY4vpSGtnwhmfxB_Ll63Kcmu8nV505uG9RcG4g7yO7r0-AWazwS3mhB4FMSwubdzi4IWarqtJ8g6Zbp0p1IbpdQkM8qmUj3NzUfbuL-YchmqcysrH4TRjpaR9F6H8tMjcI71VzFnycNxQDo?purpose=inline)

![Interactive light curve viewer now live! | MAST](https://images.openai.com/static-rsc-4/DvENyYo9M-bPryANBf0Xn5w5FaDHVYfufsOlCN76IJTjFMtJs27isz_pp-eYZSLyWRDSkiG6jllHK6cCJlsBg0wbcZdp4c5weInv0ftWgkPFras5_eSuGFSn0Waich37Sd5QxLfHWumVu083BW7IT47WD9EnfBtrUDsY9gGQTew?purpose=inline)

![Podcast CB SyR 234: Supremacía cuántica, Naukas 2019, Planet Hunters TESS y más noticias - La Ciencia de la Mula Francis](https://images.openai.com/static-rsc-4/5Nc9ip8XPspU5__AeaGbocWuR09ToRVxIHdpSZFz6R56QAsEXdDacZiCQuSAKfR1zOlfNiNkdeuLSI-tbkYpcU2Jv-omLokht6LATKrejHR1Dg7B12lZ5Nrwy1U3UC8GgPN2KV_PDNiEW2HEYzmb7NFUrqRDzviYzAjIvZflo9k?purpose=inline)

**4**

Imagine your repo has a dashboard where you can:

- search a TIC/Kepler ID,
- see raw light curve,
- detrended light curve,
- BLS periodogram,
- folded transit,
- model prediction,
- uncertainty,
- nearest neighbors in latent space.

Now it's not just a model.

It's a tool.

That's memorable.

## One thing I want to push you on

Because you've been talking about ASTRA and world models...

This project **should not become ASTRA-lite.**

ASTRA is about **orbital dynamics of satellites and space objects**.

This project is about **astronomical signal processing**.

Different scientific problems.

If you start stuffing JEPA, world models, orbital simulation, LLM agents, RAG, etc. into exoplanet detection, it'll become kitchen sink.

Keep this project laser-focused:

> **Build the best open-source AI pipeline for discovering and vetting transiting exoplanets from Kepler/TESS light curves.**

Then steal ideas from ASTRA (self-supervised learning, uncertainty, latent dynamics) only where they genuinely improve that goal.

## The version I'd build if I were optimizing for top-tier internships / research

Best version

# 9.8/10

**Name:**OrbitalSight (or whatever you call it)

**Pipeline:**

- NASA Kepler/TESS FITS ingestion.
- Quality masks + detrending + normalization.
- BLS candidate generation.
- Self-supervised encoder pretrained on unlabeled stars.
- Planet vetting head with calibrated uncertainty.
- Candidate ranking against NASA catalogs.
- Interactive explorer/dashboard.

**Stretch goal:** discover unlabeled TESS candidates and compare against later catalog releases.

That is the kind of project I'd expect from someone applying for ML research or space AI work.
