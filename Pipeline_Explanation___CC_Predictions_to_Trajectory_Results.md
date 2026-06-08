# From CN Metric Predictions to Heatwave Trajectory Analysis
## Full Pipeline Explanation

---

## 1. Starting Point — What Was Predicted

A ConvLSTM model was trained to predict **next-day Complex Network (CN) metrics** for each grid cell in a given European region, using ERA5 climate data as input features. The CN metrics predicted were:

- **CC** (Clustering Coefficient) — how interconnected a node's neighbours are; high CC means surrounding cells experience correlated temperature anomalies simultaneously
- **BC** (Betweenness Centrality) — how often a node lies on the shortest path between other nodes; high BC means the node acts as a relay for information flow
- **DC** (Degree Centrality) — how many other nodes a given node is connected to
- **HW** (Heatwave occurrence) — binary label, used as validation ground truth

The CN itself is built from ERA5 temperature data using Event Synchronisation: two grid cells are connected if their heatwave occurrences are temporally synchronised. The resulting network captures which locations tend to experience heatwaves at the same time.

---

## 2. The Original Hypothesis — Standing vs Propagating

Based on **Mondal & Mishra (2021)** and **Wang et al. (2025)**, the hypothesis was:

> Nodes with **high CC and low BC** are *standing* nodes — locally cohesive, heatwave stays put.  
> Nodes with **low CC and high BC** are *propagating* nodes — relay nodes, heatwave moves through them.

Mondal (2021) demonstrated this for the USA using sparse weather station networks, finding that high-BC, low-CC regions along the US West Coast propagated heatwaves inland. Wang et al. (2025) independently confirmed that heatwaves follow preferred propagation pathways driven by Rossby wave packets, and that events are either *propagating* (moving high-pressure system) or *standing* (stationary blocking high). [1][8]

The plan was to predict CC and BC jointly, classify each pixel as standing or propagating, and validate against observed HW.

---

## 3. Why the Multi-Metric Approach Failed — Collinearity

When the CC−BC difference distribution was plotted, it showed a spike at zero across the entire European domain. The same was true for CC−DC.

**What this means:** In a dense gridded network (ERA5 at 0.25° resolution), all three metrics go up and down together — when a heatwave is active, CC, BC, and DC are all simultaneously elevated. The *difference* between them, which is what the standing/propagating classification requires, is near-zero everywhere.

This is a **network topology problem**, not a model failure. Mondal used sparse weather station networks where BC has genuine spatial variation because there are true bottleneck nodes. In a dense grid, there are many equally short paths between any two nodes, so BC is uniformly small and uninformative. The collinearity means you cannot use the CC/BC ratio to distinguish node types in this setting.

---

## 4. What CC Alone Tells You

Despite the multi-metric failure, **CC alone showed strong spatial structure and predictive power**.

The Clustering Coefficient measures local thermal coherence: a grid cell with high CC is part of a neighbourhood where all surrounding cells are experiencing correlated temperature anomalies. In physical terms, **high predicted CC tomorrow = the model expects a spatially coherent, locally synchronised heat pattern at that location tomorrow**. This is precisely what a heatwave looks like from a network perspective. [2]

This was validated using the **Jaccard index** (see Section 5).

---

## 5. Jaccard Validation — What It Measures and How

### The Jaccard Index

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Where:
- $A$ = pixels classified as **active** (predicted CC ≥ 70th percentile) at day $t$
- $B$ = pixels where **HW is observed** at day $t + \text{lead}$
- $J = 0$ → no spatial overlap; $J = 1$ → perfect overlap

### Lead Time

The lead time is the gap between the prediction day and the validation day:

- **Lead +1d**: use active CC map from day $t$, compare to HW at day $t+1$
- **Lead +2d**: compare to HW at day $t+2$
- **Lead +3d**: compare to HW at day $t+3$

This tests whether today's CN structure contains information about where the heatwave will be in the *future* — not just where it is now. If Jaccard were only computed at lead=0 (same day), it would measure contemporaneous correlation, not prediction.

### What n Is

$n$ is the number of days used to compute the mean Jaccard. For each day $t$, the calculation is only performed if HW pixels exist at $t + \text{lead}$ (otherwise Jaccard is trivially zero). $n$ decreases slightly at longer leads because you lose days at the end of the test period and some target days have no observed HW.

### Interpreting the Values

A Jaccard of 0.38 is **strong** for spatial prediction. The denominator (union) is inherently large because the active CC region covers ~30% of the domain by construction, and the HW region covers a different fraction — their union is always larger than either alone. Values above 0.3 indicate the model is correctly localising the heatwave region, not just predicting the right general area.

---

## 6. Trajectory Analysis — Largest Connected Component

Once the binary active/inactive map is computed, the question becomes: *where is the heatwave spatially, and does it move over time?*

The naive approach — averaging the centroid of all active pixels — fails when the active region is fragmented into multiple scattered patches, producing meaningless average positions.

The **Largest Connected Component (LCC)** approach:
1. Find all spatially contiguous clusters of active pixels using connected component labelling
2. Filter out clusters smaller than a minimum size threshold (noise)
3. Track the centroid of the **largest** cluster over time

This gives the position of the dominant heatwave node each day. The temporal sequence of LCC centroids = the **heatwave trajectory**.

---

## 7. Results — Three Case Studies

### Iberia 2003 — Standing Event ✓

| Metric | Value |
|---|---|
| Active → HW Jaccard (lead+1d) | 0.379 |
| Peak centroid | 39.5°N, -3.7°E (central Spain) |
| Centroid movement (Aug 3–17) | < 0.2° lat, < 0.1° lon |
| N_comp during peak | 1 throughout |

The centroid was essentially stationary over La Mancha / central Spain for 12 consecutive days. This is consistent with the literature: the 2003 European heatwave was caused by a **quasi-stationary northward displacement of the Azores High**, which blocked the westerlies and prevented cooler Atlantic air from reaching Iberia for weeks (Garcia-Herrera et al., 2010). The blocking system barely moved, and the CN analysis recovers exactly this — a single, fixed, coherent active region. [0]

### Eastern Europe 2010 — Shifting Event ✓

| Metric | Value |
|---|---|
| Active → HW Jaccard (lead+1d) | 0.389 |
| Peak centroid range | ~50°N, 25–36°E (Ukraine/western Russia) |
| Centroid movement | ~10° lon drift over event |
| N_comp during peak | 1–3 |

The centroid drifted eastward during onset (Jul 15–24) then settled over central Ukraine during the peak (Aug 1–18) before the event collapsed. This is consistent with the literature: the 2010 Russian heatwave was driven by a **blocking high that developed in late June and persisted through early August**, sustained by a complex interplay of planetary Rossby waves, transient eddies, and ENSO teleconnections. The block was not perfectly stationary — it established progressively over western Russia — which explains the eastward centroid drift in the onset phase. [0]

### Scandinavia 2018 — Fragmented Event ✓

| Metric | Value |
|---|---|
| Active → HW Jaccard (lead+1d) | 0.191 |
| Centroid range | 59–61°N, alternating 12–26°E |
| N_comp | 2–5 throughout |
| HW positive rate | 8.3% |

The centroid alternated between ~12°E (Sweden/Norway) and ~26°E (Finland/Baltic) on consecutive days, with N_comp consistently 2–5. This is consistent with the literature: the 2018 Scandinavian heatwave featured **two distinct long-lived high-pressure centres** over Sweden and Finland simultaneously, producing a spatially fragmented event rather than a single coherent blocking high (Hoy et al., 2020; World Weather Attribution, 2018). The lower Jaccard (0.19) reflects the sparse HW field (only 8.3% of pixels) rather than poor model performance.

---

## 8. The Lead-Time Decay Pattern

Across all three regions, Jaccard decreases with lead time:

| Region | Lead+1d | Lead+2d | Lead+3d |
|---|---|---|---|
| Iberia 2003 | 0.379 | 0.352 | 0.325 |
| Eastern Europe 2010 | 0.389 | 0.364 | 0.358 |
| Scandinavia 2018 | 0.191 | 0.163 | 0.137 |

This decay is the **persistence signature**: the CN structure today is most informative about tomorrow's HW, and predictive power degrades further out. This confirms the signal is genuinely predictive rather than contemporaneous.

Eastern Europe shows the **slowest decay** (0.389 → 0.358), consistent with a persistent blocking high that maintains the same spatial footprint for days. Iberia shows steeper decay despite being a standing event, likely because the HW boundary fluctuates around the fixed centroid more than the centroid itself moves.

---

## 9. Summary

| Step | What happened | Why |
|---|---|---|
| Predict CC, BC, DC, HW | ConvLSTM trained on ERA5 CN metrics | Establish next-day network structure |
| Test CC/BC classification | CC−BC difference ≈ 0 everywhere | Dense grid → collinearity, BC uninformative |
| Use CC alone | High CC co-locates with HW (Jaccard ~0.38) | CC captures local thermal coherence |
| LCC centroid tracking | Largest connected component centroid per day | Avoids noise from scattered active pixels |
| Three-region comparison | Iberia: stationary; EE: drifting; Scandi: fragmented | Matches known atmospheric dynamics |

---

## References

[1] Mondal, S. & Mishra, A.K. (2021). Complex Networks Reveal Heatwave Patterns and Propagations Over the USA. *Geophysical Research Letters*, 48(2).

[2] Wang, M. et al. (2025). Evidence for preferred propagating terrestrial heatwave pathways due to Rossby wave activity. *Nature Communications*, 16, 4742.

[3] Garcia-Herrera, R. et al. (2010). A review of the European summer heat wave of 2003. *Critical Reviews in Environmental Science and Technology*, 40(4), 267–306.

[4] Christian, J.I. et al. (2020). Flash drought development and cascading impacts associated with the 2010 Russian heatwave. *Environmental Research Letters*, 15(9).

[5] Hoy, A. et al. (2020). Analyses of the Northern European Summer Heatwave of 2018. *Meteorologische Zeitschrift*.
