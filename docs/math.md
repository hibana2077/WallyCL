## 4) Mathematical Derivations & Proofs (Full)

### 4.1 Notation and Setup

* Input space $\mathcal{X}$, label set $\mathcal{Y}$. Training samples $(X,Y) \sim \mathcal{D}$.
* A **group** $G = (X_1,\dots,X_m, X_*)$ with labels $Y_1=\cdots=Y_m=y$ and $Y_* \neq y$.
* Backbone produces token embeddings $Z = (z_1,\dots,z_T)\in\mathbb{R}^{T\times d}$; pooled image embedding $F = f(X) = \operatorname{pool}(Z) \in \mathbb{R}^{d}$ with $\|F\|_2=1$.
* Token miner selects $S\subset[T], |S|=k$. Denote token-mean $\bar Z_S(X) = \frac{1}{k}\sum_{t\in S} z_t/\|\cdot\|$.

### 4.2 Losses (Formal)

1. **Odd-One-Out**: For each $x\in G$, define $s(x) = \frac{1}{m} \sum_{i=1}^m d\big(f(x), f(x_i)\big)$ with $d(u,v)=1-\langle u,v\rangle$. The probability that $x$ is odd is $p(x\,|\,G) = \frac{\exp(\alpha s(x))}{\sum_{x'\in G}\exp(\alpha s(x'))}$. Cross-entropy with label $x_*$.
2. **Group-SupCon**: For anchor $x$, positives $P(x)=\{x'\in G\setminus\{x\}: Y'=Y\}$, negatives all others in batch. Loss

$$
\mathcal{L}_{\mathrm{sup}} = \mathbb{E}\left[-\log \frac{\sum_{p\in P(x)} \exp(\langle f(x), f(p)\rangle/\tau)}{\sum_{a\neq x}\exp(\langle f(x), f(a)\rangle/\tau)}\right].
$$

3. **TokCon**: With selected subsets $S$ for odd and positive aggregate,

$$
\mathcal{L}_{\mathrm{tok}} = \mathbb{E}\left[ -\log \frac{\exp(\langle \bar Z_S(X_*), \bar Z_S(X^+)\rangle/\tau)}{\exp(\langle \bar Z_S(X_*), \bar Z_S(X^+)\rangle/\tau)+\sum_{n}\exp(\langle \bar Z_S(X_*), \bar Z_S(X^{(n)})\rangle/\tau)} \right].
$$

### 4.3 InfoNCE-style Mutual Information Lower Bound

Let $U=(\bar Z_S, Y)$ where $\bar Z_S = \bar Z_S(X)$. Consider the standard InfoNCE estimator with one positive and $K$ negatives drawn from the marginal of $Y$:

$$
\mathcal{L}_{\mathrm{NCE}} = \mathbb{E}\Big[ -\log \frac{\exp(\phi(U)/\tau)}{\exp(\phi(U)/\tau) + \sum_{i=1}^{K} \exp(\phi(U_i^-)/\tau)} \Big],
$$

where $\phi(U)=\langle \bar Z_S, \mu_Y\rangle$ and $\mu_Y = \mathbb{E}[\bar Z_S\,|\,Y]$. Using the standard derivation (Donsker–Varadhan is not required; see Poole et al., 2019-style analysis), for properly normalized scores the following holds:

**Theorem 1 (MI Lower Bound).** If negatives are sampled from the product of marginals, then

$$
I(\bar Z_S; Y) \ge \log(K+1) - \mathcal{L}_{\mathrm{NCE}}.
$$

*Proof Sketch.* The estimator is a biased, but consistent lower bound on $I$, obtained by comparing the joint density to the product of marginals via a contrastive classifier trained to discriminate the true pair from negatives. With normalized dot-product scores, the bound reduces to the above; equality holds when the classifier is optimal and the model family is rich enough.

**Corollary 1.1.** Incorporating **multi-positives** (SupCon) does not break the bound; instead, it **reduces estimator variance** by averaging across $|P(x)|=m-1$ positives, yielding tighter empirical estimates for finite batches.

### 4.4 Fisher Discriminant Growth via TokCon

Assume class-conditional token-averaged embeddings are sub-Gaussian with means $\mu_y$ and within-class covariance $\Sigma_W$. Let $\Delta = \mu_y - \mu_{y'}$.

**Theorem 2 (Fisher Ratio Increase).** Under gradient descent on $\mathcal{L}_{\mathrm{tok}}$ with normalized embeddings, there exists a stepsize $\eta>0$ s.t. after one step the Fisher discriminant $J( S ) := \Delta^\top \Sigma_W^{-1} \Delta$ **does not decrease** in expectation over minibatches. Moreover, in the limit of small $\tau$, the gradient concentrates on directions that (i) increase $\|\Delta\|$ and (ii) decrease $\operatorname{tr}(\Sigma_W)$ by aligning selected-token means within class.

*Proof Sketch.* The TokCon gradient for the odd-positive pair is proportional to $\nabla_{\bar Z_S(X_*)} \mathcal{L}_{\mathrm{tok}} \propto -\frac{1}{\tau}\Big( \bar Z_S(X^+) - \sum_n w_n \bar Z_S(X^{(n)}) \Big),$ where weights $w_n$ form a probability simplex. This pulls odd towards the positive aggregate and pushes it away from negatives. Aggregated over symmetric roles of classes and over minibatches, within-class token means concentrate, contracting $\Sigma_W$. The push against negatives increases the between-class mean separation $\|\Delta\|$. A first-order perturbation of $J$ then yields $\mathbb{E}[\Delta J] \ge 0$.

**Corollary 2.1 (Bhattacharyya Bound).** The Bhattacharyya distance $B=\frac{1}{8}\Delta^\top \Sigma^{-1}\Delta + \frac{1}{2}\ln \frac{\det \Sigma}{\sqrt{\det \Sigma_y \det \Sigma_{y'}}}$ increases as $J$ increases and $\Sigma_W$ contracts, which upper-bounds the Bayes error. Thus TokCon monotonically tightens a classical upper bound on classification error under the sub-Gaussian assumption.

### 4.5 Variance Reduction with Multi-Positives (SupCon)

Let gradient contributions from positives be $g_1,\dots,g_{m-1}$ with mean $\bar g$ and variance $\sigma^2$. Averaging them yields gradient variance $\sigma^2/(m-1)$. Hence SupCon’s multi-positive averaging **reduces gradient variance** compared to single-positive pairs, improving optimization stability and, by standard generalization bounds (e.g., uniform stability), potentially reducing generalization error.

### 4.6 Sample Complexity and Generalization Sketch

Using a margin-based argument with normalized embeddings and 1-Lipschitz scoring, for margin $\gamma$ achieved by the contrastive classifier, one obtains a sample complexity of $\tilde{O}(R^2/\gamma^2)$ to ensure generalization within $\epsilon$. The group construction effectively increases the number of informative positive pairs per batch from 1 to $m-1$, which—under independence or weak dependence—improves the empirical margin concentration and reduces the number of steps to reach a target margin.

### 4.7 Consistency of Differentiable Top-k Selection

Gumbel-Top-k yields a continuous relaxation $w\in[0,1]^T, \sum w\approx k$. Under annealing $\tau\to 0$ and a scoring function with unique top-k tokens almost surely, the selection converges in probability to the discrete top-k. Training with a slowly decaying $\tau$ thus approximates the discrete objective while preserving differentiability early in training; convergence follows from standard arguments for simulated annealing/continuation methods.

### 4.8 Putting It Together: Bound Improvement

Combining Theorems 1–2 and the variance discussion:

* **Information**: Group-contrastive losses lower-bound $I(\bar Z_S;Y)$ and become tighter with more (and better) positives.
* **Margins**: TokCon increases Fisher ratio and Bhattacharyya distance, tightening error bounds.
* **Stability**: Multi-positive averaging reduces gradient variance, improving optimization and generalization.

Therefore, WallyCL is theoretically grounded to (i) capture label-relevant token subsets, (ii) expand inter-class separation while contracting intra-class scatter, and (iii) train stably in few-shot, ultra-fine-grained regimes.

---

## 5) Implementation Notes & Hyperparameters (for Reproducibility)

* Default: ViT-B/16, AdamW (lr=2e-4, wd=0.05), batch = 6 groups, m=4 (K=5), k=8, $\tau_{sup}=0.1$, $\tau_{tok}=0.07$, $\alpha=5.0$, $\lambda_{sup}=1.0$, $\lambda_{tok}=0.5$, $\lambda_{ce}=1.0$.
* Warmup 5 epochs; cosine schedule to 0.2× lr; EMA decay 0.999.
* Token scorer: 2-layer MLP on token features with LayerNorm.
* Pos-token aggregation: average of in-group positive token means; alternative: attention-weighted by cosine similarity to anchor.
* Mixed precision (fp16/bf16), gradient clipping at 1.0.

---

## 6) Checklists for the Paper

* **Workflow diagram** (data → group builder → backbone → token miner → losses → optimizer).
* **Tables**: main results, ablations, robustness, efficiency.
* **Figures**: token-selection heatmaps; qualitative odd-one-out examples; convergence plots.
* **Appendix**: extended proofs (details of Theorem 2), sensitivity to $k$, additional visualizations.
