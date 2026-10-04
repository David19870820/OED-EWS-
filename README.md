# Early Warning Signal Pipeline Validation

Code accompanying the manuscript *Early warning signals in psychopathology: the channel problem*.

This repository contains the synthetic pipeline validation reported in the manuscript. It documents three traps
that can destroy or fabricate early warning signals (EWS) based on critical slowing down, and provides repairs.

## The three traps

1. **Linear detrending removes the signal.** The signal resides in the low-frequency component of the series;
   linear detrending removes the baseline around which the system is supposed to recover.
2. **Radix-2 FFT phase randomization on non-power-of-two lengths produces NaN surrogates.** When every surrogate
   statistic is NaN, the comparison is false for each surrogate, the count collapses to zero, and the p value is
   computed as (0 + 1) / (N + 1). Any series then appears significant.
3. **Unguarded NaN comparisons produce the same false positive**, independently of the FFT length issue.

## Requirements

Node.js version 24. No third-party dependencies; the standard library only.

## Layout

    real_analysis/validate_ews3.mjs    preprocessing comparison on a known CSD signal
    real_analysis/validate_ews2.mjs    alternative baseline definition
    real_analysis/diag_fft.mjs         reproduces NaN output for non-power-of-two lengths
    real_analysis/diag_fft3.mjs        verifies the zero-padding repair
    real_analysis/tv_ar_protocol.mjs   full EWS pipeline with the repaired surrogate test and numeric guards
    node_demo/simulate_ema.mjs         synthetic EMA generator used by the diagnostics
    node_demo/oed_model.mjs            model used by the simulator

## Reproduction

Run from the repository root.

    node real_analysis/validate_ews3.mjs

Expected output:

    preprocess=none     AR1-EWS sig 19/20 | mean diff=0.513
    preprocess=demean   AR1-EWS sig 19/20 | mean diff=0.513
    preprocess=detrend  AR1-EWS sig 0/20 | mean diff=0.002

    node real_analysis/diag_fft.mjs

Expected output:

    n=128 (power of 2): NaNs = 0
    n=120 (non-power): NaNs = 56 | finite=64
    real data length = 121, finite = 121
    phaseRandomize(real): NaNs = 57 | finite = 64

    node real_analysis/diag_fft3.mjs

Expected output: every surrogate value is finite, the surrogate variance is close to the original variance
(0.50 vs. 0.53 in one run), and the AR(1) structure is destroyed (original about -0.06, surrogate about -0.06).

    node real_analysis/validate_ews2.mjs

    node real_analysis/tv_ar_protocol.mjs

## Implementation notes

- The repair zero-pads the series to the next power of two before the forward transform, and truncates back to
  the original length after the inverse transform.
- Do not rescale a second time after the inverse transform, which already divides by N. Double scaling collapses
  the surrogate variance to zero and annihilates the surrogate signal.
- Apply an explicit finiteness check to the observed statistic and to every surrogate statistic before the
  comparison, and mark the test missing rather than letting a NaN enter the count.
- Surrogate counts below about 200 give a p value resolution too coarse to support a p < .05 decision.

## Scope

All results are obtained on synthetic data with known dynamics. They show that the pipeline can recover a known
signal under the correct preprocessing, and that specified implementation choices can destroy or fabricate it.
They are not evidence about the channel hypothesis, which remains to be tested on real data.

## License

MIT
