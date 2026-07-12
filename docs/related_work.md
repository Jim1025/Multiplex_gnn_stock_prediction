# Related Work — MAGNET Paper §2 Draft

Reference material for the MAGNET paper's Related Work section. Groups
prior work into three themes (cross-market GNN, lead-lag relationships,
stock-level GNN baselines), then closes with MAGNET's positioning
against each.

Bibliography key format: `[TYPE][YEAR][AUTHOR]` where TYPE ∈
{JRN, CONF, WS} for journal / conference / workshop. Full BibTeX
entries live in `references.bib` (TBD).

---

## §2.1 Cross-Market / Cross-Border Graph Neural Networks

The dominant recent direction in cross-market stock prediction is to
model multiple markets as an integrated heterogeneous graph and let a
learnable attention mechanism discover cross-market signals.

### MEIG (Bukhari, Maqsood & Sattar, 2025) — direct competitor, same venue; implemented as baseline (core form)

> Bukhari, M., Maqsood, M., & Sattar, A. (2025).
> **A novel inter-intra graph neural networks for stock price forecasting
> modeling cross-border relationships.**
> *Expert Systems with Applications*, 286, 127907.
> https://www.sciencedirect.com/science/article/abs/pii/S0957417425015295

- Full name: Macro-Event Driven Inter-Intra Graph Neural Network (MEIG)
- **Intra-graph** per market + **inter-graph** across markets; edges in
  both are formed by **correlation thresholding** (their Eq. 6:
  corr > θ_inter), i.e. statistical edge discovery between distinct
  companies — not learned, not identity-based
- **CGAT** (Cross-Graph Attention Layer, their Eq. 21-23): despite the
  name, the mechanism is **three free learnable scalars** (one per graph
  branch) passed through a softmax and used to mix the three parallel
  GCN branch outputs — graph-level, input-independent, fixed after
  training; not query-key attention, not per-edge, not per-node
- Pipeline order: spectral GCN (spatial) first, then LSTM 64/32
  (temporal) — the reverse of MAGNET's LSTM-then-GAT
- Augmented with **42.74M tweets** for event sentiment + 5 macro
  indicators per country as node attributes
- Evaluation: price-level regression (MAE/MSE/RMSE) on 5 markets
  (SP500 / FTSE-100 / SSE-50 / KSE-100 / BIST-100), pairwise market
  experiments; baselines are SVR / RF / LSTM / BiLSTM only (no GNN,
  no cross-market method); train/test protocol, optimizer and seeds
  not disclosed
- Published in the same target venue as MAGNET (ESWA), same year

**MAGNET baseline result (our modality-reduced implementation)**:
We implement MEIG's graph + CGAT core on our dataset — correlation
graphs estimated in the T=20 input window on log returns (the paper
does not disclose its correlation window), θ = 0.3, three parallel
dense-GCN branches, 3-scalar CGAT, GCN-then-LSTM order — with the
macro and tweet-sentiment node features removed for input parity
across all models (the paper's own Tables 5-8 include no-macro
ablation rows, the same precedent as MAN-SF's no-text variant).
Best of 4-config grid = **0.023 IC** vs MAGNET's 0.072 (Δ = +0.049).
Two diagnostic findings: (1) at lr 1e-3 the best-val checkpoint lands
on an epoch-2 validation spike (val 0.071) that collapses on test
(−0.002, gap 0.074) — the noisy 20-day correlation edges provide no
stable structure to select on; (2) the learned CGAT weights never
differentiate, ending at [0.335, 0.333, 0.332] ≈ uniform — with only
three graph-level scalars over noisy statistical graphs, the mixing
weights receive almost no usable gradient signal, whereas MAGNET's
per-node per-dimension gate meaningfully specializes.

**MAGNET differentiation**:
- **Edge source**: correlation thresholding between distinct entities
  (statistical estimate, window- and θ-sensitive, spurious-correlation
  prone) vs verified dual-listing identity (zero estimation error)
- **Mixing granularity**: 3 input-independent graph-level scalars vs
  per-node per-dimension input-dependent gate — MAGNET fixes the
  hard-to-estimate part (topology) and spends its learning capacity on
  a finer, easier target
- **Task rigor**: price-level regression (persistence-dominated, no
  naive baseline reported) vs next-day return IC under walk-forward
- Reproducibility: MAGNET achieves IC=0.072 without tweet or macro data,
  suggesting the cross-market mechanism itself carries substantial signal

### ASTGCN (Gong, Wang, Zhou & Xie, 2025)

> Gong, J., Wang, G.-J., Zhou, Y., & Xie, C. (2025).
> **Cross-market volatility forecasting with attention-based
> spatial–temporal graph convolutional networks.**
> *Journal of Empirical Finance*, 83, 100961.
> https://www.sciencedirect.com/science/article/abs/pii/S0927539825000611

- Attention-based Spatial-Temporal GCN forecasting volatility of **18
  financial markets**
- Nodes = markets (not stocks), edges = pairwise market correlations
- Finds cross-market correlations differ between tranquil/turmoil periods
- Journal of Empirical Finance — top-tier empirical finance journal

**MAGNET differentiation**: stock-level prediction (finer granularity),
return prediction (directly tradeable), and focused pair-wise structure
vs broad market network.

### US-China Bipartite (Zhu et al., 2026)

> A Bipartite Graph Approach to U.S.-China Cross-Market Return Forecasting
> arXiv:2603.10559 (2026).
> https://arxiv.org/abs/2603.10559

- Directed bipartite graph: US nodes → China nodes
- Edges selected via **rolling-window hypothesis testing** (not learned)
- Feature-selection layer for downstream Lasso/RF/ensemble methods
- Documents pronounced directional asymmetry: US previous-close returns
  predict China intraday returns, reverse effect limited

**MAGNET differentiation**: End-to-end neural training vs two-stage
(statistical edge selection + traditional ML). Stock-level vs
market-index-level. **Reinforcing evidence**: this paper's finding
of US-leading-Asia asymmetry independently supports MAGNET's ADR-TW
structural assumption.

### HGT (Hu, Dong, Wang & Sun, 2020) — implemented as baseline

> Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020).
> **Heterogeneous Graph Transformer.**
> *Proceedings of The Web Conference 2020*, pp. 2704–2710.
> https://arxiv.org/abs/2003.01332

- General-purpose heterogeneous GNN with meta-relation-aware attention
- Type-specific Q/K/V projections × relation-specific weights
- Originally for OAG (Open Academic Graph) and Amazon Review
- We adapt it to ADR-TW: 2 node types + 3 relation types (L1 intra,
  L2 intra, A12 cross with reverse edges)

**MAGNET baseline result**: Best of 4-config hyperparameter grid = 0.028
IC vs MAGNET's 0.072. Note both models learn data-dependent cross-market
weights — the difference is the **search space of the learning target**.
HGT's attention must distribute over all incoming edges (the single
cross-market edge competes with 2-4 intra-market edges inside one
softmax, per head, per layer), whereas MAGNET restricts learning to an
element-wise gate over exactly two pre-aligned representations at the
same node index. See §6 for why the restricted learning target
generalizes better (+0.045 IC).

---

## §2.2 Lead-Lag Relationships in Financial Markets

The premise that some markets/stocks lead others in price discovery is
well established in international finance. MAGNET's ADR-TW pairing
exploits the natural lead-lag arising from non-overlapping trading
sessions (NYSE 09:30-16:00 EST precedes TWSE 09:00-13:30 CST next day).

### DeltaLag (Zhou et al., 2025) — implemented as baseline

> Zhou, W. et al. (2025).
> **DeltaLag: Learning Dynamic Lead-Lag Patterns in Financial Markets.**
> *Proceedings of the 6th ACM International Conference on AI in Finance
> (ICAIF 2025)*.
> https://arxiv.org/abs/2511.00390

- First end-to-end deep learning method for dynamic lead-lag discovery
- Sparsified cross-attention identifies daily leader-lagger pairs +
  pair-specific lag values (τ ∈ [1, l_max])
- Extracts lag-aligned raw features from leaders for lagger prediction
- Beats a range of temporal / spatio-temporal SOTA baselines
- **Scope note**: original evaluation is single-market U.S. equities
  (S&P 500 / NASDAQ / NYSE universes, overlapping trading sessions);
  its lead-lag arises from cross-stock information diffusion within one
  market, **not** cross-border trading-session ordering. DeltaLag does
  not itself claim cross-market prediction.

**MAGNET baseline result (our cross-market adaptation)**: DeltaLag is
natively single-market; we adapt it to the cross-market setting by
adding the 7 ADR nodes to its candidate pool (14 candidates for each
TW target). Its pair-selection mechanism is pool-agnostic, so this
extension exercises the method exactly as designed — it simply gains
the opportunity to discover the ADR-TW lead-lag on its own. Best of
4-config hyperparameter grid = 0.030 IC vs MAGNET's 0.072. DeltaLag
must solve a 65-way discrete search (13 candidates × 5 lags) per
target per day, and fails to reliably recover the domain-verified
ADR-TW pairing from 1150 training samples. This supports MAGNET's
core thesis: **fix the hard-to-learn decisions (pair topology, lag)
with domain knowledge and restrict learning to the easy one
(per-dimension mixing weights)** — see §6.

### Classical Lead-Lag Literature

- Lo & MacKinlay (1990) — original documentation of size-based lead-lag
- Hou (2007) — industry lead-lag effects
- Baur & Fry (2009) — international spillover asymmetry
- These establish lead-lag as a real (not spurious) phenomenon, justifying
  encoding it as structural prior rather than discovering it from data.

---

## §2.3 Stock-Level GNN Methods (Single-Market)

Prior work on stock-level graph neural networks has explored various
graph constructions and attention mechanisms — but almost exclusively
within a single market universe.

### Adv-ALSTM (Feng, Chen, He, Ding, Sun & Chua, 2019) — implemented as baseline

> Feng, F., Chen, H., He, X., Ding, J., Sun, M., & Chua, T.-S. (2019).
> **Enhancing stock movement prediction with adversarial training.**
> *IJCAI 2019*, pp. 5843-5849.
> https://arxiv.org/abs/1810.09936

- LSTM + Temporal Attention + FGSM adversarial training
- Original task: binary classification on ACL18 / KDD17 US-stock datasets
- Serves as strong single-market non-graph baseline

**MAGNET baseline result**: 0.034 IC — well above LSTM-only (-0.005) but
still 0.038 below MAGNET, confirming cross-market information contributes
beyond what temporal attention captures.

### HATS (Kim, So, Jeong, Lee, Kim & Kang, 2019) — implemented as baseline

> Kim, R., So, C. H., Jeong, M., Lee, S., Kim, J., & Kang, J. (2019).
> **HATS: A hierarchical graph attention network for stock movement
> prediction.**
> *NeurIPS 2019 Workshop on Robust AI in Financial Services*.
> https://arxiv.org/abs/1908.07999

- Hierarchical GAT: stock-level → relation-level attention
- Original uses Wikidata K≈57 relation types
- We adapt with single relation (industry from PAIR_MAP)

**MAGNET baseline result**: 0.019 IC. Notably underperforms Adv-ALSTM,
suggesting single-market hierarchy (industry axis) is a weaker
inductive bias than cross-market axis (0.053 below MAGNET).

### MAN-SF (Sawhney, Agarwal, Wadhwa & Shah, 2020) — implemented as baseline

> Sawhney, R., Agarwal, S., Wadhwa, A., & Shah, R. R. (2020).
> **Deep attentive learning for stock movement prediction from social
> media text and company correlations.**
> *EMNLP 2020*, pp. 8415-8426.
> https://aclanthology.org/2020.emnlp-main.676/

- Multi-modal Attention Network: fuses (price GRU, tweet BERT,
  correlation GCN)
- Original task: binary classification on S&P 500 with Twitter data
- We adopt the paper's own **no-text variant** — see original Ablation
  Table 4 which reports this setting
- Also swap GCNConv → GATv2 due to MPS numerical bug

**MAGNET baseline result**: 0.033 IC. Multi-modal fusion within a single
market matches Adv-ALSTM performance, both 0.039 below MAGNET — evidence
that **the axis of fusion (market vs modality) matters more than the
fusion mechanism itself**.

### Additional Contemporaneous Work

Cited but not re-implemented (either due to data requirements or being
concurrent single-market advances):

- **MASTER** (Li, Liu, Shen, Wang, Chen & Huang, 2024, AAAI) —
  Market-Guided Stock Transformer, intra/inter-stock alternation with
  market-info gating for feature selection. Single-market despite the
  "market-guided" name (market = broad index conditioning, not
  cross-market fusion).
  https://ojs.aaai.org/index.php/AAAI/article/view/27767

- **MDGNN** (2024) — Multi-Relational Dynamic GNN. Multiplex edges among
  stocks/industries/investment-banks. Related to MAGNET's multiplex
  design but focuses on within-market multi-relational structure.
  https://arxiv.org/abs/2402.06633

- **THGNN** — Temporal-Heterogeneous GNN combining Transformer temporal
  encoder with GAT relational model.

- **STHAN-SR** (Sawhney, Agarwal, Wadhwa & Shah, 2021, AAAI) —
  Spatio-Temporal Hypergraph Attention, extends MAN-SF with
  hypergraph structure. Single-market.

---

## §2.4 Positioning of MAGNET

MAGNET occupies a distinct methodological position at the intersection
of three themes above. Table below summarizes the differentiation:

| Prior work theme | Representative method | MAGNET's departure |
|---|---|---|
| Statistical cross-market edges + coarse mixing | MEIG (corr-threshold edges, 3-scalar CGAT) | **Pair topology fixed** by dual-listing identity (vs correlation estimate); mixing learned at per-node per-dimension granularity (vs 3 input-independent graph-level scalars) |
| Learned cross-market attention | HGT (meta-relation) | **Pair topology fixed** by node-order alignment; learning restricted to a per-dimension mixing gate over the pre-aligned pair (vs attention competing across all edges) |
| Dynamic learned lead-lag | DeltaLag (single-market origin; adapted by us to a cross-market pool) | **Pair and lag fixed** by domain knowledge (dual-listing identity, trading-session ordering); only mixing weights are learned (vs joint discrete search over pair × lag) |
| Single-market fusion mechanisms | MAN-SF, HATS, MASTER | **Cross-market fusion** as primary axis, single-market treated as secondary intra-graph structure |
| Broad multi-market GNN | ASTGCN (18 markets) | **Focused verified-pair study** with cleaner ablation methodology |

**Core contributions specific to MAGNET**:

1. **Structural A12 diagonal coupling** — the ADR-TW pair topology and
   its 1-day lag are fixed by node-order alignment and trading-session
   ordering; **learning is restricted to a per-dimension mixing gate**
   over the two pre-aligned representations (the gate itself remains
   fully data-dependent). Empirically validated against three
   alternatives spanning the design space: HGT, whose attention
   distributes over all incoming edges (Δ = +0.045 IC); DeltaLag, which
   additionally learns pair selection and lag values (Δ = +0.043 IC);
   and MEIG-core, which replaces identity pairing with
   correlation-thresholded edges and graph-level scalar mixing
   (Δ = +0.049 IC).

2. **Multi-mechanism ablation** — clean isolation of L1/L2/A12/fusion
   contributions via the Stage 0 ablation table. Show that IC 0.072 →
   0.003 when A12 is severed, providing rare causal-level evidence in
   finance ML literature.

3. **Focused verified dual-listing** — restriction to 7 pairs of ADR-TW
   cross-listings of identical companies, avoiding confounders from
   cross-country correlation between distinct entities that plague
   broader-scope work.

4. **Generalization signature** — MAGNET's val→test IC gap of 0.005
   (vs 0.04-0.07 for competing GNN baselines) demonstrates the
   structural inductive bias provides **implicit regularization**,
   an insight relevant to the broader finance-ML small-data regime.

---

## Bibliography (to be formatted as BibTeX in `references.bib`)

Order roughly by citation frequency in the paper text.

- [CONF][2020][Hu-HGT] Hu et al., 2020, WWW — HGT
- [JRN][2025][Bukhari-MEIG] Bukhari, Maqsood & Sattar, 2025, ESWA — MEIG
- [CONF][2025][Zhou-DeltaLag] Zhou et al., 2025, ICAIF — DeltaLag
- [JRN][2025][Gong-ASTGCN] Gong et al., 2025, JEF — ASTGCN
- [CONF][2019][Feng-AdvALSTM] Feng et al., 2019, IJCAI — Adv-ALSTM
- [WS][2019][Kim-HATS] Kim et al., 2019, NeurIPS Workshop — HATS
- [CONF][2020][Sawhney-MANSF] Sawhney et al., 2020, EMNLP — MAN-SF
- [CONF][2024][Li-MASTER] Li et al., 2024, AAAI — MASTER
- [ARXIV][2024][MDGNN] MDGNN (arxiv 2402.06633)
- [ARXIV][2026][USChinaBipartite] US-China Bipartite (arxiv 2603.10559)
- [CONF][2021][Sawhney-STHANSR] Sawhney et al., 2021, AAAI — STHAN-SR
- [JRN][1990][Lo-MacKinlay] Lo & MacKinlay, 1990 — Lead-lag
- [JRN][2007][Hou-Lead] Hou, 2007 — Industry lead-lag
- [JRN][2009][Baur-Fry] Baur & Fry, 2009 — International spillover
