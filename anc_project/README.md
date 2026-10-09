# LMS active noise cancellation simulation

This project studies feedforward ANC with an ideal unity secondary path. It is a numerical model, not a physical loudspeaker demonstration. Run implementation before using its results in your term paper.

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python anc_lms.py
```

Tested with NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. Script regenerates all figures and numerical results beside itself. Runtime can take roughly a minute depending on computer. No audio files or external datasets required.

## Understand the experiment

1. Generate white Gaussian noise, then pass it through a fourth-order 600 Hz Butterworth low-pass filter at 8000 samples/s. Discard one second of filter startup and normalize reference RMS to one. This is synthetic low-frequency noise, not a recorded engine or fan.
2. Pass reference x through a fixed, causal, 16-tap FIR primary path P. First two coefficients are zero, representing a two-sample delay. All coefficients are stated in source code.
3. Add independent white noise v with RMS 5% of primary disturbance RMS. Error microphone receives d = P*x + v. Reference cannot predict v, creating an irreducible expected noise floor.
4. Adaptive filter estimates correlated disturbance: y(n) = w(n)^T u(n), where u(n) = [x(n), x(n-1), ...]. Physical anti-noise has sign -y. Unity secondary path gives e(n) = d(n) - y(n).
5. Update weights after evaluating current error: w(n+1) = w(n) + 2*mu*e(n)*u(n). Start weights and delay buffer at zero. The factor 2 matters: sources defining w += mu*e*u use twice our mu for equivalent updates.

Our expected objective is J = E[e^2]. Its instantaneous gradient is -2*e*u. Descending this gradient gives the LMS update above. Real ANC requires including secondary-path dynamics; ordinary LMS is justified here by unity-path assumption. With S(z) present, error depends on S(z)y and its gradient uses the reference filtered by S(z), motivating filtered-x LMS.

## Measurement definitions

- MSE: mean(e^2), in arbitrary amplitude units squared. Report both all five seconds and final second [4,5).
- Noise attenuation: 10*log10(mean(d^2)/mean(e^2)), using matching windows. Positive values mean less noise; negative values mean amplification.
- Requested SNR improvement equals this attenuation if an unchanged desired signal is hypothetically present. No desired signal is simulated, so absolute input/output SNR is undefined. Label reported metric noise attenuation or equivalent SNR improvement, not conventional SNR.
- Learning curves are 50 ms block averages of squared error for one realization. These are empirical curves, not ensemble estimates of E[e^2].
- PSD curves use Welch estimation on final second with 1024-sample segments and SciPy defaults (Hann window, 50% overlap, density scaling).

Five fixed seeds provide repeatability and modest sensitivity checking. Each seed uses identical x,d across all parameters. Sweep: lengths 8,16,32; mu 0.001,0.005,0.01,0.05. Run is marked divergent when weights become nonfinite or exceed 1e6 in absolute magnitude; this numerical cutoff does not define mathematical stability. Finite five-second runs also do not prove long-term stability. Summary includes number of failed runs. Plot omits settings with any failed seeds.

Baseline is seed 42, 32 taps, mu 0.001, selected after the sweep showed instability at the initially proposed mu 0.01. Final-second window is an operational late-run metric; inspect curve before claiming steady state. Low-frequency correlated inputs can converge slowly along weakly excited coefficient directions. Small residual error does not imply exact identification of every FIR coefficient. An 8-tap filter cannot represent entire 16-tap path, although correlation can reduce observed error.

## Files

- `anc_lms.py`: commented simulation and checks.
- `results.csv`: each run, including failed runs and both evaluation windows.
- `summary.json`: baseline metrics, means, sample standard deviations, failure counts.
- `baseline_signals.npz`: x,d,v,y,e,w,primary,fs arrays.
- `plots/`: waveform, learning curves, spectra, parameter sweep, in PNG and vector PDF formats.

## Checks and limits

Script verifies a hand-sized LMS recursion using independent indexing, exact primary model leaves only independent noise, and baseline attenuates noise by over 10 dB. All runs are checked for nonfinite or runaway weights. No measurements are invented. Independent white noise is a modeling choice, not calibrated microphone noise. Results cannot establish real acoustic performance, spatial cancellation, perceptual quality, or robustness to an unknown secondary path.

For paper, explain unity secondary path and reference-independent floor before interpreting results. Statistical summary uses five realizations; this small sample is not a broad robustness study. No step size or filter length is universally optimal.

## Source leads for paper

Verify and read relevant sections before citing:

- B. Widrow et al., “Adaptive noise cancelling: Principles and applications,” Proceedings of the IEEE, vol. 63, no. 12, pp. 1692–1716, 1975. DOI: 10.1109/PROC.1975.10036. Author-hosted paper: https://isl-www.stanford.edu/~widrow/papers/j1975adaptivenoise.pdf
- S. M. Kuo and D. R. Morgan, “Active noise control: A tutorial review,” Proceedings of the IEEE, vol. 87, no. 6, pp. 943–975, 1999. DOI: 10.1109/5.763310. https://ieeexplore.ieee.org/document/763310/

Draft acknowledgment to edit: AI assistance was used to develop and explain simulation code. The author reviewed the implementation, reproduced the results, and takes responsibility for the final work.

## Explicit input and output comparison

`plots/signal_comparison.png` and `.pdf` show reference input x, disturbance d, adaptive filter output y, and final residual e in the same 50 ms final-second interval, with identical amplitude limits. Filter output y estimates disturbance; system output e = d-y is noise remaining after cancellation. Therefore attenuation compares d against e, not x against y. `signal_comparison.csv` exports mean-square power and RMS of all four signals over the full record and final second. New metric fields report before/after RMS, power, and percentage reductions in the final second. RMS reduction and power reduction are different measures; neither is perceived loudness reduction.
