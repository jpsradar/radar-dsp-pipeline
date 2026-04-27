# Radar DSP Pipeline

A reproducible radar DSP pipeline for **signal-level analysis, detection behavior, and system trade-offs**.

This repository is designed to show how practical DSP decisions—filtering, windowing, FFT processing, and detection strategies—impact **detectability, probability of detection (Pd), false alarms (Pfa), and required SNR**.

The focus is on **engineering behavior and measurable consequences**, not on isolated formulas or toy examples.

---

## Scope

This pipeline answers questions such as:

- When does a signal become detectable in the spectral domain?
- How do filtering and windowing affect detectability?
- What is the impact of leakage and noise bandwidth on detection?
- How do thresholds translate into empirical Pd and Pfa?
- How do DSP choices affect system-level detection performance?

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
- `fft_tools.py` → FFT, spectral peak detection, noise floor estimation  
- `filters.py` → FIR/IIR design and application  
- `windows.py` → windowing, coherent gain, ENBW  
- `detection.py` → thresholding, Pd/Pfa estimation, Doppler peak detection  

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

All outputs are generated under:

```
figures/generated_plots/
```

---

## Reproducible walkthrough

Run:

```bash
jupyter notebook notebooks/radar_dsp_pipeline_walkthrough.ipynb
```

Then:

```
Kernel → Restart & Run All
```

---

## Testing (engineering invariants)

Run:

```bash
python -m pytest -q
```

Tests validate:

- Signal power and SNR consistency  
- FFT peak correctness and scaling  
- Filter passband/stopband behavior  
- Window coherent gain and ENBW  
- Detection Pd/Pfa and threshold logic  

---

## Modeling approach

- Deterministic signal generation  
- Explicit SNR control  
- Correct handling of linear vs dB domains  
- Detection evaluated at a defined cell under test  
- Monte Carlo used for measurement  

---

## What this is not

This repository does not include:

- full radar equation modeling  
- tracking systems  
- advanced clutter models  
- real-time constraints  

It is a **controlled DSP analysis pipeline**.

---

## Summary

This project demonstrates:

- clean DSP architecture (library + pipeline)  
- reproducible experiments  
- engineering-level validation via tests  
- system-level interpretation of DSP choices  

