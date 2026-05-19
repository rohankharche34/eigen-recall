# Adaptive Spectral Tempering for Precision-Controlled Associative Memory
## Anvil P-04 · PCAM Precision Agent

Adaptive Spectral Tempering (AST) is a geometry-aware precision controller developed for the Anvil Hackathon problem:

P-04 · Precision-Controlled Associative Memory

Original benchmark/problem statement:
https://github.com/Sauhard74/Anvil-P-E/tree/main/bench-p04-pcam

This repository contains:
- the adaptive precision controller,
- experiments,
- analysis,
- and design exploration built on top of the provided benchmark infrastructure.

The benchmark provides a frozen Precision-Controlled Associative Memory (PCAM) system. The only controllable component is a diagonal precision vector Π predicted for each corrupted query.

This project explores how adaptive precision modulation can:
- improve retrieval under corruption,
- stabilize associative memory dynamics,
- reduce pathological anisotropy,
- and preserve discriminative spectral structure.

Pure Python · NumPy only · CPU only.

--------------------------------------------------------------------------------
BACKGROUND
--------------------------------------------------------------------------------

Associative memory systems often exhibit highly anisotropic retrieval dynamics.

Some spectral directions dominate retrieval because their eigenvalues are significantly larger than others.

This creates an important tradeoff:

- Large eigenvalues improve retrieval and semantic discrimination
- Large eigenvalues also create unstable and anisotropic dynamics

Flattening the spectrum entirely improves isotropy but destroys semantic structure.

Leaving the spectrum untouched preserves retrieval but causes dynamical imbalance.

The objective therefore becomes:

    controlled spectral tempering

instead of:
- complete whitening,
- or full isotropic cancellation.

The controller attempts to:
- preserve useful dominant eigendirections,
- reduce pathological spectral imbalance,
- stabilize retrieval geometry,
- and improve corruption robustness.

--------------------------------------------------------------------------------
CORE IDEA
--------------------------------------------------------------------------------

The controller dynamically estimates:

1. Query reliability
2. Soft memory retrieval confidence
3. Local retrieval geometry
4. Local variance structure
5. Spectral imbalance

The predicted precision vector is then adapted per-query.

The current design combines:
- soft nearest-memory retrieval,
- attractor reconstruction,
- reliability-aware precision,
- local geometry estimation,
- and mild spectral compression.

Instead of flattening eigenvalues completely, the system attempts to gently compress spectral disparity while preserving discriminative structure.

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

```txt
bench-p04-pcam/
|
|-- adapter.py
|     Adapter abstract base class.
|     Defines the interface every precision controller must implement.
|
|-- pcam_model.py
|     Frozen Precision-Controlled Associative Memory (PCAM) dynamics.
|     Contains:
|     - energy function
|     - retrieval dynamics
|     - gradients
|     - Hessian estimation
|
|-- data.py
|     Synthetic pattern generation and corruption pipeline.
|     Creates:
|     - twin-pair memory patterns
|     - Gaussian corruption
|     - mask corruption
|
|-- metrics.py
|     Evaluation metrics.
|     Computes:
|     - retrieval accuracy
|     - anisotropy spread
|     - spectral reduction statistics
|
|-- harness.py
|     Multi-seed evaluation harness.
|     Handles:
|     - seed regeneration
|     - anti-gaming evaluation
|     - aggregation
|     - score computation
|
|-- run.py
|     Full benchmark runner CLI.
|
|-- self_check.py
|     Fast local evaluation loop for development and debugging.
|
|-- requirements.txt
|     Python dependencies.
|
|-- README.md
|     Project documentation and benchmark overview.
|
|-- adapters/
|     Precision controller implementations.
|     |
|     |-- dummy.py
|     |     Π = I baseline controller.
|     |
|     |-- variance.py
|     |     Naive variance-based precision controller.
|     |
|     |-- class_conditional.py
|     |     Paper-inspired class-conditional precision controller.
|     |
|     |-- myteam.py
|           Adaptive Spectral Tempering (AST) controller.
|           Main project implementation.
|
|-- experiments/
|     Experimental logs, analysis scripts, and metric tracking.
|
|-- results/
|     Benchmark outputs and evaluation summaries.
|
|-- docs/
      Research notes, derivations, and design exploration.
```

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

Clone repository:

    git clone <repo-url>

Move into project:

    cd bench-p04-pcam

Install dependencies:

    pip install -r requirements.txt

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

Run baseline controller:

    python self_check.py --adapter adapters.dummy:DummyAgent --quick

Run AST controller:

    python self_check.py --adapter adapters.myteam:Engine

Run multi-seed evaluation:

    python self_check.py \
        --adapter adapters.myteam:Engine \
        --seeds 1 2 3 4 5 42 101

Run full harness:

    python run.py --adapter adapters.myteam:Engine

--------------------------------------------------------------------------------
PRECISION CONTROLLER PIPELINE
--------------------------------------------------------------------------------

The controller performs the following steps:

1. Normalize corrupted query
2. Compute cosine similarities
3. Perform soft retrieval using temperature-scaled softmax
4. Estimate denoised attractor state
5. Estimate coordinate reliability
6. Estimate local geometry
7. Apply spectral tempering
8. Predict adaptive precision vector
9. Normalize and stabilize output

--------------------------------------------------------------------------------
SOFT ATTRACTOR RECONSTRUCTION
--------------------------------------------------------------------------------

The controller reconstructs a denoised attractor estimate:

    x_hat = weights @ self.X

where retrieval weights are computed using softmax similarity.

The attractor acts as a soft reconstruction of the corrupted memory.

Coordinates agreeing strongly with the attractor are considered more reliable and receive higher precision.

--------------------------------------------------------------------------------
LOCAL GEOMETRY ESTIMATION
--------------------------------------------------------------------------------

Local geometry is estimated using weighted local variance:

    local_var = Σ w_i (x_i - x_hat)^2

High local variance:
- indicates unstable directions,
- reduces precision,
- dampens overconfident updates.

Low local variance:
- indicates stable retrieval structure,
- receives slightly higher precision.

This acts as a weak spectral conditioning prior.

--------------------------------------------------------------------------------
SPECTRAL INTERPRETATION
--------------------------------------------------------------------------------

The controller can be interpreted geometrically as performing:

    partial spectral compression

instead of:
- full whitening,
- inverse-Hessian cancellation,
- or complete isotropic flattening.

The objective is to:
- reduce pathological eigenvalue disparity,
- preserve dominant semantic directions,
- maintain retrieval robustness,
- and stabilize convergence dynamics.

The project experimentally explores the tradeoff between:
- retrieval quality,
- and isotropic dynamics.

--------------------------------------------------------------------------------
BENCHMARK EVALUATION
--------------------------------------------------------------------------------

The benchmark evaluates:

| Category               | Weight |
|------------------------|--------|
| Retrieval Accuracy     | 70 pts |
| Anisotropy Reduction   | 20 pts |
| Code Quality           | 10 pts |

--------------------------------------------------------------------------------
RETRIEVAL METRIC
--------------------------------------------------------------------------------

Measured relative to Π = I baseline.

Interpretation:

| Mean Δ         | Meaning                  |
|----------------|--------------------------|
| Δ ≤ 0.00       | baseline or worse        |
| 0.00 – 0.02    | weak signal              |
| 0.02 – 0.05    | meaningful improvement   |
| 0.05 – 0.08    | strong                   |
| ≥ 0.08         | full marks               |

--------------------------------------------------------------------------------
ANISOTROPY METRIC
--------------------------------------------------------------------------------

Measures spread reduction in retrieval dynamics.

| Reduction      | Meaning                  |
|----------------|--------------------------|
| ≤ 1.0×         | baseline                 |
| 1.0× – 1.5×    | mild conditioning        |
| 1.5× – 3.0×    | geometry-aware           |
| 3.0× – 5.0×    | strong conditioning      |
| ≥ 5.0×         | full marks               |

--------------------------------------------------------------------------------
CURRENT EXPERIMENTAL OBSERVATIONS
--------------------------------------------------------------------------------

Empirical testing revealed an important tradeoff:

- Strong spectral flattening improves isotropy but harms retrieval
- Weak conditioning preserves retrieval but barely affects spread
- Mild spectral compression gives the best balance

This suggests:
- dominant eigendirections are important for discrimination,
- but uncontrolled spectral dominance destabilizes retrieval dynamics.

The current controller therefore focuses on:
- preserving semantic geometry,
- while softly tempering spectral imbalance.

--------------------------------------------------------------------------------
ANTI-GAMING EVALUATION
--------------------------------------------------------------------------------

The benchmark includes:
- canonical public seeds,
- randomized multi-seed regeneration,
- held-out adversarial evaluation.

Each seed regenerates:
- memory patterns,
- corruption,
- model parameters,
- and adapter instances.

The held-out evaluation uses:
- larger dimensions,
- PCA-MNIST structured memories,
- harder corruption,
- stronger anisotropy.

The controller is designed to generalize across both synthetic and structured retrieval settings.

--------------------------------------------------------------------------------
DESIGN PHILOSOPHY
--------------------------------------------------------------------------------

The project intentionally avoids:
- hardcoded seed tuning,
- benchmark memorization,
- brittle inverse-Hessian whitening,
- aggressive isotropy enforcement.

Instead, the controller emphasizes:
- local geometry,
- corruption robustness,
- retrieval stability,
- and cross-seed generalization.

Because every benchmark eventually evolves into an arms race between geometry and whatever terrible shortcuts humans discover at 3 AM.

--------------------------------------------------------------------------------
CONSTRAINTS
--------------------------------------------------------------------------------

- NumPy only
- CPU only
- No GPU
- One-pass inference only
- No iterative refinement
- Positive diagonal precision only
- No modification of PCAM dynamics

The harness automatically:
- clips precision values,
- enforces mean(Π) = 1,
- and projects outputs into the valid constraint set.

--------------------------------------------------------------------------------
RESEARCH CONNECTIONS
--------------------------------------------------------------------------------

Relevant research directions include:
- associative memory
- Hopfield networks
- spectral conditioning
- covariance shrinkage
- representation isotropy
- Hessian conditioning
- information geometry
- partial whitening
- neural collapse

The project particularly explores:
- retrieval geometry,
- eigenspectrum shaping,
- and adaptive spectral tempering.

--------------------------------------------------------------------------------
ABOUT THE BENCHMARK
--------------------------------------------------------------------------------

This project was developed for:

    Anvil Hackathon
    Problem P-04 · Precision-Controlled Associative Memory

The original benchmark and PCAM framework were provided as part of the Anvil Hackathon problem statement:

https://github.com/Sauhard74/Anvil-P-E/tree/main/bench-p04-pcam

--------------------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------------------

This repository contains original work built for the Anvil Hackathon benchmark.

Refer to the original benchmark repository and competition rules for usage restrictions related to the provided framework and assets.
