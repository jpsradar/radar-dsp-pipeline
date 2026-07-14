# Radar DSP Pipeline

A reproducible radar DSP pipeline for **signal-level analysis, detection behavior, numerical robustness, and system trade-offs**.

This repository shows how practical DSP decisions—filtering, windowing, FFT processing, recursive-filter implementation, and detection strategies—impact **detectability, probability of detection (Pd), false alarms (Pfa), numerical stability, and required SNR**.

The focus is on **engineering behavior and measurable consequences**, not on isolated formulas or toy examples.

---

## Scope

This pipeline answers questions such as:

- When does a signal become detectable in the spectral domain?
- How do filtering and windowing affect detectability?
- What is the impact of leakage and noise bandwidth on detection?
- How do thresholds translate into empirical Pd and Pfa?
- How do DSP choices affect system-level detection performance?
- How do IIR pole location, arithmetic precision, and filter structure affect numerical robustness?

It is a **DSP-to-detection bridge**, not a full radar simulator.

---

## Pipeline structure

The repository is structured as a clean separation between:

- `src/` → reusable DSP library
- `scripts/` → executable pipeline stages
- `notebooks/` → reproducible walkthrough
- `tests/` → engineering invariants

### DSP modules (`src/`)

- `signals.py` → signal generation and SNR-controlled noise
- `fft_tools.py` → FFT, spectral peak detection, and noise-floor estimation
- `filters.py` → FIR/IIR design and application
- `windows.py` → windowing, coherent gain, and ENBW
- `detection.py` → thresholding, Pd/Pfa estimation, and Doppler peak detection
- `iir_stability.py` → pole-radius experiments, direct-form vs SOS behavior, and numerical robustness

No plotting. No CLI. Pure logic.

---

### Pipeline stages (`scripts/`)

1. **01 — Signal vs noise (FFT)**  
   SNR → spectral visibility → detectability

2. **02 — Filtering**  
   FIR vs IIR → noise shaping → detectability impact

3. **03 — Windowing**  
   leakage vs resolution vs ENBW trade-offs

4. **04 — Detection**  
   thresholding → empirical Pd/Pfa

5. **05 — System trade-offs**  
   filtering + windowing + detection strategy → Pd/Pfa/SNR trade-offs

6. **06 — IIR stability and numerical robustness**  
   pole radius + direct form vs SOS + float32 vs float64 → bounded response, numerical overflow, and divergence

All outputs are generated under:

    figures/generated_plots/

---

## Reproducible walkthrough

Run:

    jupyter notebook notebooks/radar_dsp_pipeline_walkthrough.ipynb

Then:

    Kernel → Restart & Run All

Individual stages can also be executed directly:

    python scripts/01_signals_noise_fft.py
    python scripts/02_fir_iir_filters.py
    python scripts/03_windowing_leakage.py
    python scripts/04_detection_doppler.py
    python scripts/05_system_tradeoffs.py
    python scripts/06_iir_stability_demo.py

---

## Testing (engineering invariants)

Run:

    python -m pytest -q

Tests validate:

- signal power and SNR consistency
- FFT peak correctness and scaling
- filter passband/stopband behavior
- window coherent gain and ENBW
- detection Pd/Pfa and threshold logic
- IIR stability inputs, impulse-response behavior, and dtype preservation

---

## Modeling approach

- deterministic signal generation
- explicit SNR control
- correct handling of linear vs dB domains
- detection evaluated at a defined cell under test
- Monte Carlo used for measurement
- explicit comparison between direct-form and second-order-section IIR implementations
- controlled float32 vs float64 numerical experiments

---

## What this is not

This repository does not include:

- full radar equation modeling
- tracking systems
- advanced clutter models
- real-time constraints
- hardware fixed-point implementation

It is a **controlled DSP analysis pipeline**.

---

## Summary

This project demonstrates:

- clean DSP architecture (library + pipeline)
- reproducible experiments
- engineering-level validation via tests
- system-level interpretation of DSP choices
- practical IIR stability and numerical robustness analysis
