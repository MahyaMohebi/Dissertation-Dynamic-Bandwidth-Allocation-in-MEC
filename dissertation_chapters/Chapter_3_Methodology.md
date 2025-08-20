# Chapter 3 — Methodology

## Objectives and scope

This project designs, trains, and evaluates a bandwidth prediction service and integrates it with adaptive bitrate (ABR) streaming policies under different caching conditions. The methodology spans data preparation, model training, online inference, and controlled experiments with statistical analysis.

Primary objectives:
- Learn a sequence model that predicts next-step network throughput from recent observations.
- Compare ABR policies (baselines vs. model-assisted) with and without edge caching.
- Quantify Quality of Experience (QoE) improvements and assess statistical significance.

## System overview

The system is modular:
- Data layer: CSV traces normalized into a common schema; preprocessing to windowed tensors.
- ML service: a Flask HTTP service that loads trained models and exposes /predict.
- Clients: a Windows C++ WinHTTP live client that streams features to the service; Python demos.
- ABR policies: BOLA, MPC, Pensieve placeholder, and Predictive (service-backed) policies.
- Experiment runner: executes ABR×caching experiments across traces and records QoE.
- Reporting: statistics (paired t-test, Wilcoxon) and plots for distributions and CDFs.

## Data sources and preprocessing

Sources
- ns-3/5G-LENA CSV exports (preferred when available).
- A custom dataset converted using `src/sim_adapters/custom_dataset_adapter.py` into standardized traces.
 - A lightweight C++ simulator exists for internal validation/smoke tests; it is not used for primary results reported in Chapter 4.

Standardized schema (one row per second)
- throughput_mbps, rtt_ms, loss_rate, jitter_ms, users, cell_load

Trace indexing and grouping
- `data/network_traces/trace_*.csv` holds standardized traces.
- `trace_index.csv` maps each trace to provenance metadata, including group labels (e.g., Global/Local) for stratified analysis.

Preprocessing
- Window length: 10 seconds. Features are stacked to shape (T, 10, 6) to predict the next-step throughput.
- Target: next-step throughput_mbps.
- Scaling: StandardScaler is fitted on all feature windows in `preprocess_data.py` and applied to produce `X.npy`, `y.npy`, and `scaler.pkl` under `data/processed/`.

## Learning model and training procedure

Architecture (stacked LSTM regressor)
- Input: (window=10, features=6)
- LSTM(32, return_sequences=True) → Dropout(0.2) → LSTM(16) → Dense(16, relu) → Dense(1, linear)
- Loss: Mean Squared Error; Metric: Mean Absolute Error
- Optimizer: Adam (lr=1e-3)

Training protocol
- Dataset: `data/processed/X.npy` and `y.npy` produced by preprocessing.
- Validation split: 10% held out by Keras during `fit`.
- Early stopping: patience 5, restore best weights.
- Checkpoints: best model per run saved under `models/checkpoints/`.
- Ensemble: up to 13 independent models (distinct random seeds) saved under `models/trained/` and later ensembled.
- Default hyperparameters: epochs=20, batch_size=64, max_models=13.

Ensembling
- `src/ml/create_ensemble.py` combines the trained models (e.g., uniform averaging of predictions) to improve robustness and reduce variance.

## Online inference service

- The Flask service at `http://127.0.0.1:8080/predict` loads the trained ensemble and exposes a JSON API.
- Input: the most recent 10×6 feature window (scaled as in training or scaled server-side using `scaler.pkl`).
- Output: predicted next-step throughput_mbps.
- Timeout handling ensures timely responses for ABR integration.

## ABR policies and caching conditions

ABR policies
- BOLA: buffer-based heuristic baseline.
- MPC: model predictive control using recent bandwidth statistics.
- Pensieve (placeholder): kept for completeness of comparisons.
- Predictive: queries the ML service for next-step throughput to inform bitrate selection.

Caching emulation
- Two conditions: EDGE_ON and EDGE_OFF.
- EDGE_ON: simulates edge cache hits with reduced latency (e.g., ~20 ms); EDGE_OFF: origin latency (e.g., ~120 ms).
- Hit probability for EDGE_ON is fixed (e.g., 0.6) to emulate partial cache effectiveness.

## Experimental design

Design
- Full factorial: ABR policies × caching conditions × traces.
- Each (policy, cache, trace) generates a streaming session simulation and per-trace QoE metrics.
- Group-aware analysis: when `trace_index.csv` provides group labels (Global/Local), results are stratified.

Primary outputs
- `results/evaluation/experiment_results.json`: per-run records including policy, cache, trace name, group, and QoE statistics.
- `results/evaluation/STATS.txt`: significance tests and summaries; plots stored in the same directory.

QoE metrics
- Aggregate QoE score per session, alongside components such as average bitrate and rebuffer ratio, computed by the project’s QoE module.

Statistical testing
- Paired t-test and Wilcoxon signed-rank test are applied to paired conditions (e.g., Predictive vs. BOLA under the same traces and cache setting).
- We report p-values per cache condition and, when applicable, within group strata.

## Live C++ client → Python service

- The Windows C++ client (`src/simulation/live_client.cc`) uses WinHTTP to send sliding 10×6 feature windows to `/predict`.
- This validates online integration and latency budget under realistic client–server interaction on Windows.

## Reproducibility and environment

- Environment isolation: Python virtual environment and pinned requirements (TensorFlow 2.10 CPU, pandas, scikit-learn, etc.).
- Determinism: fixed random seeds per model run; simulator seeds when generating synthetic traces.
- Run orchestration: `RUNBOOK.md` provides exact, prescriptive steps; outputs are written to versioned folders.

## Limitations and threats to validity

- Data representativeness: traces may not capture all network conditions; custom aggregation may induce bias.
- Label leakage: StandardScaler fitted on the entire dataset can slightly inflate validation performance; acceptable here due to downstream policy-level evaluation, but noted as a limitation.
- Simplified caching: fixed hit probability and latency are abstractions of complex cache dynamics.
- QoE model choice: specific weighting of bitrate vs. rebuffer affects absolute scores; relative comparisons remain informative.

## Ethics and responsible use

- No PII: traces are aggregated network statistics without personally identifiable information.
- Fairness: group-aware reporting (Global/Local) enables detection of disparate impacts.
- Reproducibility and transparency: code, preprocessing, and statistical procedures are documented and automated.
 - Data provenance: ns-3 scenarios are executed externally; only their CSV exports are ingested and standardized. Any synthetic simulator outputs are excluded from the main evaluation.

## Summary

We built a sequence-learning pipeline around a stacked LSTM to forecast short-horizon throughput, integrated it into ABR decision-making with and without edge caching, and evaluated QoE impacts under controlled, repeatable experiments. Statistical tests substantiate observed differences. The methodology is end-to-end reproducible via the runbook and produces artifacts suitable for analysis and reporting.

## Mathematical formulations

### Notation and windowing
- At time step t, let feature vector be x_t ∈ ℝ^F with F = 6.
- We form a sliding window of length W = 10: X_t = [x_{t−W+1}, …, x_t] ∈ ℝ^{W×F}.
- The prediction target is next-step throughput: y_t = throughput_{t+1}.

### Standardization
For each feature dimension j ∈ {1,…,F}, with training mean μ_j and std σ_j > 0:
$$x'_{t,j} = \frac{x_{t,j} - \mu_j}{\sigma_j}.$$
Window standardization is applied element-wise to obtain X'_t.

### LSTM forecasting model
Given the standardized sequence X'_t, the LSTM computes hidden states with gates:
$$\begin{aligned}
i_t &= \sigma(W_i[h_{t-1}, x'_t] + b_i),\\
f_t &= \sigma(W_f[h_{t-1}, x'_t] + b_f),\\
g_t &= \tanh(W_g[h_{t-1}, x'_t] + b_g),\\
o_t &= \sigma(W_o[h_{t-1}, x'_t] + b_o),\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t,\\
h_t &= o_t \odot \tanh(c_t),
\end{aligned}$$
stacked over two LSTM layers, followed by dense layers to yield the prediction
$$\hat y_t = f_\theta(X'_t).$$

### Training objective and metric
- Loss (MSE):
$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{n=1}^N\big(y_n - \hat y_n\big)^2.$$
- Metric (MAE):
$$\text{MAE} = \frac{1}{N}\sum_{n=1}^N\big|y_n - \hat y_n\big|.$$

### Ensemble aggregation
With M independently trained models, the ensemble prediction is the uniform mean:
$$\hat y^{(ens)} = \frac{1}{M}\sum_{m=1}^M \hat y^{(m)}.$$

### Caching latency model
Let Z ∼ Bernoulli(p_hit) indicate an edge cache hit. Effective segment latency L_eff is
$$L_{eff} = Z\,L_{edge} + (1-Z)\,L_{origin}}, \quad \mathbb{E}[L_{eff}] = p_{hit} L_{edge} + (1-p_{hit}) L_{origin}.$$

### QoE formulation
For a session with T segments, bitrate sequence {b_t}, rebuffer time {r_t} (seconds during segment t), and bitrate switching penalty term S = \frac{1}{T-1}\sum_{t=2}^T |b_t - b_{t-1}|, an additive QoE takes the form
$$\text{QoE} = \alpha\, \overline{b} - \beta\, \frac{1}{T}\sum_{t=1}^T r_t - \gamma\, S,$$
where \overline{b} = \frac{1}{T}\sum_{t=1}^T b_t. Coefficients (α, β, γ) weight bitrate utility, rebuffer penalty, and smoothness; in our reporting we summarize average bitrate and rebuffer ratio alongside the composite QoE.

### Statistical tests
- Paired t-test on per-trace differences d_i = m^{(A)}_i − m^{(B)}_i (e.g., QoE under Predictive vs. BOLA):
$$\bar d = \frac{1}{n}\sum_{i=1}^n d_i, \quad s_d^2 = \frac{1}{n-1}\sum_{i=1}^n (d_i - \bar d)^2, \quad t = \frac{\bar d}{s_d/\sqrt{n}}.$$
Under H0: \bar d = 0, t follows a t-distribution with n−1 degrees of freedom.
- Wilcoxon signed-rank test: rank the nonzero |d_i|, apply signs, and compute
$$W = \sum_{i=1}^n \operatorname{sign}(d_i)\,R_i,$$
with two-sided p-value from the exact distribution or normal approximation for large n.

### Empirical CDFs
For measured QoE values {q_i}, the empirical CDF is
$$\hat F(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{q_i \le x\},$$
plotted per caching condition to visualize distribution shifts.
