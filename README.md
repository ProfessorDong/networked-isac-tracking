# Distributed Tracking for Networked ISAC over Quantized, Lossy Backhaul

Code and results for the paper

> L. Dong, "Distributed Tracking for Networked ISAC over Quantized, Lossy Backhaul,"
> submitted to *IEEE Transactions on Signal and Information Processing over Networks*, 2026.

Base stations that perform integrated sensing and communication act as cooperating
monostatic radars and fuse their evidence by consensus over a quantized, lossy backhaul.
This repository contains the simulation code for every experiment in the paper and its
supplementary material, the stored results from which every reported number is
generated, and instructions for obtaining the public datasets.

## Repository layout

```
code/isac_core.py        CV target model, monostatic angle/range/range-rate measurements,
                         anchored HCMCI information messages, requantized consensus with
                         subtractive dither and link failures, Theorem-2 bounds, exact
                         link-failure constants, SPD safeguard, observability Gramians
code/isac_mpc.py         belief-state predictive beamforming (certified proximal
                         SCA-WMMSE with adjoint gradient)
code/v2x_radar_io.py     loader for recorded V2X-Radar 4-D radar detections
code/exp_*.py            one driver per experiment (writes results/exp_*.json)
code/make_numbers.py     results/*.json -> numbers.tex (every number quoted in the paper)
code/make_figures.py     results/*.json -> figs/data/*.dat (data plotted in the figures)
results/exp_*.json       stored results; each rerun file records source hashes and versions
numbers.tex              generated macros (as used in the submitted manuscript)
figs/data/*.dat          generated figure data (as used in the submitted manuscript)
data/                    place the datasets here (not included; see below)
```

## Experiments

| Driver | Paper section | Data | Approx. runtime |
|---|---|---|---|
| `exp_V_frozen_check.py` | VII-A, S-VII (frozen-message check of Thms. 2-3) | synthetic | 2 min |
| `exp_B_distributed_gap.py` | VII-A (consensus bounds, fusion rules) | synthetic | 5 min |
| `exp_D_rate_distortion.py` | VII-B (Thm. 3, Cor. 3, frame dependence) | synthetic | <1 min |
| `exp_S_network_scale.py` | VII-C, S-X (network size up to 64 BSs) | synthetic | 40 min |
| `exp_T_baselines.py` | VII-D, S-XI (consensus vs. flooding of quantized innovations) | synthetic | 20 min |
| `exp_U_robustness.py` | VII-E, S-XII (failure models beyond the assumptions) | synthetic | 1 min |
| `exp_J_robust_calibration.py` | V-A, S-VIII (worst-case calibration bias) | synthetic | 1 min |
| `exp_A_belief_mpc.py` | VII-F, S-IX (predictive beamforming, integrated loop) | synthetic | 20 min |
| `exp_G_v2x_radar.py` | VII-G1, S-XIII (V2X-Radar, synthesized measurements) | V2X-Radar-C | 3 min |
| `exp_R_v2x_real_radar.py` | VII-G2, S-XIV (V2X-Radar, recorded 4-D radar detections) | V2X-Radar-C | 5 min |
| `exp_I_boreas.py` | VII-G3 (Boreas trajectories) | Boreas | 3 min |
| `exp_E_deepmimo.py` | VII-G3 (DeepMIMO ray-traced maps) | DeepMIMO v4 | 1-5 min |

Runtimes are for 18 worker processes (`Pool(18)` in the drivers). Seeds are fixed, and
all trackers in a comparison share their measurement noise (common random numbers).

## Reproducing the paper's numbers

From the stored results (seconds):

```bash
cd code
python3 make_numbers.py      # writes ../numbers.tex; strict: fails on missing or non-finite values
python3 make_figures.py      # writes ../figs/data/*.dat
```

Rerunning an experiment (synthetic experiments need no data):

```bash
cd code
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u exp_B_distributed_gap.py
```

## Environment

Tested with Python 3.12.13, NumPy 2.2.6, SciPy 1.17.1, pandas 3.0.2, PyYAML 6.0.3, and
deepmimo 4.0.1 (only for `exp_E_deepmimo.py`).

```bash
pip install -r requirements.txt
```

## Datasets

The datasets are not redistributed here; obtain them from their providers under their
licenses and place them as follows.

**V2X-Radar** (L. Yang *et al.*, NeurIPS 2025 Datasets and Benchmarks):
<https://github.com/yanglei18/V2X-Radar>. Use the cooperative release V2X-Radar-C
(train, validate, test):

```
data/v2x_radar/V2X-Radar-C/{train,validate,test}/<scene>/<agent>/...
```

Agent `-1` is the roadside unit and `142` the ego vehicle. The cooperative vehicle labels
are already in the common roadside-unit frame; do not transform them again.
The radar detections are used directly in `exp_R_v2x_real_radar.py`, with label-based
(oracle) association.

**Boreas** (K. Burnett *et al.*, IJRR 2023), available from the public S3 bucket
`s3://boreas` (`aws s3 sync --no-sign-request`). Only `applanix/lidar_poses.csv` is needed
for each of the ten drives listed in `code/exp_I_boreas.py`:

```
data/boreas/<drive>/applanix/lidar_poses.csv
```

The GPS time column is in microseconds, and easting/northing are in a local frame.

**DeepMIMO v4** (A. Alkhateeb, ITA 2019), downloaded with the `deepmimo` package, for
example `deepmimo.download("city_0_newyork_3p5_s")`, for the scenarios
`city_0_newyork_3p5_s`, `city_1_losangeles_3p5`, `city_4_phoenix_3p5`, and
`city_0_newyork_28`:

```
data/deepmimo_scenarios/<scenario>/...
```

The driver sets `deepmimo.config.set("scenarios_folder", ...)` to this folder and caches
the extracted path-gain maps as `data/deepmimo_scenarios/_cache_<scenario>.npz` on first
use.

## Notes

- The tracker keeps its prediction whenever the fused information matrix is not positive
  definite; these events are counted and reported.
- The power-of-two range code supports block ranges in [2^-126, 2^127]; out-of-range
  blocks are declared coder failures.
- In the comparison with flooding, the pipelined anchor is encoded as four binary32 values
  and is assumed to be delivered by a control service; its delivery deadline is not
  simulated.

## License

Code: MIT License (see `LICENSE`). The datasets remain under their providers' licenses.

## Citation

If you use this code, please cite the paper above.
