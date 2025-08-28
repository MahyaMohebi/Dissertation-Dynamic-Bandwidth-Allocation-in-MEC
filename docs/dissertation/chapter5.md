# Chapter 5 — Evaluation, Conclusion, and Future Work

## 5.1 Evaluation of experimental outcomes

The evaluation was designed to test the central hypothesis that an edge‑assisted, prediction‑guided framework could improve user‑perceived Quality of Experience (QoE), efficiency, and fairness under variable cellular conditions. Session‑level QoE was computed using the ITU‑T P.1203 methodology to provide a perceptually calibrated primary metric [20]. This metric was complemented by analysis of constituent measures, specifically average bitrate, rebuffering frequency and duration, startup delay, and quality‑switch frequency. Cross‑user equity was summarized with Jain’s fairness index to permit scale‑independent comparison across heterogeneous conditions [24]. Experimental factors included ABR policy (BOLA, MPC, PREDICT), caching regime (EDGE_ON, EDGE_OFF), and a diverse trace corpus exceeding thirty cellular recordings; paired testing was used to assess statistical significance while controlling for trace effects.

### 5.1.1 Synthesis of observed results

A consistent performance ordering was observed across the tested conditions. MPC was observed to outperform the implemented PREDICT variant on the primary QoE metric; this finding held under both EDGE_ON and EDGE_OFF and was supported by overwhelmingly significant paired tests reported in Chapter 4. The superior median QoE for MPC was manifested through reductions in both the frequency and total duration of playback stalls. BOLA’s apparent invariance was observed to be a configuration artifact resulting from the chosen bitrate ladder and the specific trace mix; its zero variance in these experiments therefore reflects a conservative operating point rather than broad robustness.

The presence of an edge cache produced modest shifts in absolute QoE distributions but did not alter the relative ranking of ABR policies. This observation indicates that the substantial underperformance of the implemented PREDICT policy was primarily attributable to internal calibration and decision‑mapping issues rather than to external content retrieval latency caused by cache absence [38].

### 5.1.2 Diagnostic analysis of PREDICT underperformance

Three interacting mechanisms were identified from the observed logs and follow‑up analyses as responsible for the PREDICT policy’s degraded QoE.

First, forecast–action lag was observed. The interval from client feature capture to inference at the `/predict` endpoint, to the controller’s allocation computation, and finally to the ABR decision introduced non‑negligible delay relative to the dynamics of the wireless channel. In highly time‑varying conditions, forecasts were sometimes stale by the time they were acted upon, producing bitrate choices mismatched to instantaneous capacity — a known failure mode in networked control contexts [7], [17].

Second, the decision translation logic was observed to be overly conservative. The implemented rules that mapped a point forecast and allocation hint to a representation included strict safety margins and conservative tie‑breaking. In many runs this combination produced sustained selection of near‑lowest representations (≈0.3 Mbps). Although rebuffering was often minimized, the resultant persistent low quality resulted in substantially lower QoE because the adopted QoE model penalizes both stalls and low visual quality [20]. Thus, the PREDICT policy failed to achieve a satisfactory trade‑off between stall avoidance and quality preservation.

Third, a misalignment in input semantics was observed in some cases. The trained model expected features in a fixed ordering and scale; inconsistencies between the simulator’s feature mapping during inference and the semantics used at training time produced systematic forecast bias. Such biases, even when predictions were delivered with low latency, led to poor decision outcomes and compounded the effects of latency and conservative decision rules. The sensitivity of prediction‑based control to preprocessing fidelity has been reported elsewhere and requires disciplined, versioned contracts between training and inference stages [8].

## 5.2 Conclusion

An integrated system was implemented and evaluated that combined short‑horizon throughput forecasting, ONNX export and C++ inference, MEC‑assisted allocation, and client ABR adaptation within an ns‑3 / 5G‑LENA environment. The implemented experiments demonstrated feasibility: models were trained and exported, scaler metadata was persisted and consumed by an ONNX Runtime C++ predictor embedded in the simulator, and controlled sweeps were executed across ABR policies and caching regimes.

The principal empirical conclusion derived from the implemented study is that, under the tested configuration, MPC achieved substantially and significantly higher session QoE than the implemented PREDICT policy. This outcome does not imply that prediction is inherently ineffective; rather, it demonstrates that naive integration of point forecasts into lag‑prone control loops can amplify forecasting errors and result in net QoE degradation. The effectiveness of predictive, edge‑assisted adaptation was therefore observed to depend critically on: (a) semantic fidelity of preprocessing and scaler application, (b) uncertainty‑aware and utility‑aware decision rules, and (c) bound end‑to‑end inference latency relative to the control period.

The study’s stated objectives were met: a forecasting model was trained and integrated into a real‑time serving path, an allocator was implemented to translate forecasts into per‑user hints, and experiments were conducted and analyzed with standardized QoE metrics and paired statistical tests. The analysis provided measureable explanations for observed underperformance, thereby satisfying the dual aims of measurement and diagnosis.

## 5.3 Future work

The observed outcomes suggest several prioritized directions for further investigation and engineering.

1. **Probabilistic, multi‑step forecasting.** The model should be extended to produce short multi‑step horizons and associated uncertainty estimates (prediction intervals). Such richer outputs enable uncertainty‑aware allocation and ABR decisions that balance risk and reward more effectively [15], [16].

2. **Uncertainty‑aware and utility‑based ABR design.** The ABR decision logic should be redesigned to incorporate forecast uncertainty and to optimize a utility function aligned with the QoE metric. Adaptive safety margins, online parameter tuning, or lightweight online learning mechanisms could be used to close the gap between probabilistic forecasts and representation choices [21], [28].

3. **Preprocessing contracts and validation.** A formalized, versioned contract for feature extraction and scaler application should be established and tested end‑to‑end. Automated tests that validate ordering, units, and ranges at inference time are recommended to prevent systematic bias.

4. **Latency engineering and scaling.** The inference path should be optimized via model compilation (ONNX Runtime optimizations), batching strategies, and horizontal scaling. Admission control for prediction requests should be considered to protect tail latency and avoid staleness under high concurrency.

5. **Enhanced caching and content modeling.** The caching model should be extended beyond fixed hit probabilities to lifecycle‑aware, popularity‑adaptive policies and to incorporate view‑aware or tile‑based strategies for emerging modalities. The interaction between caching dynamics and allocation merits co‑optimization research [14], [19], [23].

6. **Broader and stress‑test evaluation.** The trace corpus should be expanded to include extreme mobility scenarios, flash‑crowd events, and a wider variety of geography and user mixes. Benchmarks should be extended to include additional advanced baselines and energy‑efficiency metrics [25], [27], [32].

By pursuing these directions, it is anticipated that a predictive edge‑assisted controller can be developed that reliably reduces stalls and stabilizes quality in a manner that exceeds well‑tuned reactive baselines, particularly when probabilistic forecasts and robust decision logic are jointly employed. Implementation engineering — specifically ensuring preprocessing fidelity and bounded latency — is expected to be decisive in realizing these gains.

## 5.4 Closing remark

The results observed in this study emphasize that predictive capabilities must be integrated with rigorous systems engineering to yield practical QoE improvements. Forecast accuracy is necessary but not sufficient; semantic consistency, uncertainty handling, and latency control are equally essential. The recommendations and future work outlined above provide actionable steps toward realizing predictive, edge‑assisted streaming systems that are both effective and deployable in realistic cellular settings.
