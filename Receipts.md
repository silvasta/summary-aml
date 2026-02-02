# Receipts

## Week 1

### Classify ML Problem Type

- Identify data: Labeled? (Supervised) Unlabeled? (Unsupervised)
  Rewards? (RL).
- Determine goal: Prediction (supervised), patterns (unsupervised),
  decisions (RL).
- Example: "Predict house prices from features" $\to$ Supervised
  regression.

### Compute Empirical Risk

- Given dataset $\{(x_i, y_i)\}_{i=1}^n$, model $f$, loss $\ell$.
- Calculate $\hat{R}(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(x_i),
  y_i)$.
- If regularized, add $\lambda \Omega(f)$.
- Hint: For MSE, expand to variance + bias terms if asked.

### Analyze Bias-Variance

- Decompose error: Compute Bias$^2$ (avg prediction error), Variance
  (spread over datasets).
- Suggest fix: High bias $\to$ more features/complex model; high
  variance $\to$ more data/regularization.
- Exam trick: Plot learning curves (train vs. test error).

### Derive Basic ERM for Linear Regression

- Model: $\hat{y} = w^T x + b$.
- Loss: Minimize $\frac{1}{n} \sum (y_i - w^T x_i - b)^2 + \lambda
  \|w\|^2$.
- Take derivatives: Set $\nabla_w = 0 \to w = (X^T X + \lambda
  I)^{-1} X^T y$ (closed-form).
- Hint: Assume centered data for simplicity.

### Identify Overfitting Signs \& Mitigation

- Signs: Low train error, high test error.
- Steps: Split data (train/val/test), apply CV, add regularization, early stopping.
- Conceptual: Explain how it relates to VC dimension (if advanced).

### General Exam Tip

Always start with definitions/formulas, then apply to given scenario. Practice with toy datasets (e.g., 2-3 points) for quick derivations.

## Week 3

### Recipe 1: Computing Fisher Information for a Model

Used for: Deriving $I(\theta)$ from log-likelihood (common in exams for Gaussian, Poisson, etc.).

1. Write the likelihood: $L(\theta|X) = \prod_{i=1}^n p(x_i|\theta)$ or log-likelihood $\ell(\theta) = \sum \log p(x_i|\theta)$.
2. Compute the score: $s(\theta) = \frac{\partial \ell}{\partial \theta}$ (scalar) or gradient vector (multivariate).
3. Option A (score form): $I(\theta) = \mathbb{E}[s(\theta) s(\theta)^\T]$ – integrate $s^2 p(x|\theta) dx$.
4. Option B (Hessian form): $I(\theta) = -\mathbb{E}\left[ \frac{\partial^2 \ell}{\partial \theta^2} \right]$ – compute 2nd derivative, then expectation.
5. For iid data: Multiply by $n$. Check: $\mathbb{E}[s(\theta)] = 0$ for unbiasedness.
6. Exam tip: Simplify for exponential families (e.g., $I(\theta) = \Var($sufficient statistic$)$).

### Recipe 2: Applying Rao-Cramér Lower Bound (CRLB) to an Estimator

Used for: Proving variance bounds or checking if estimator (e.g., MLE) is efficient.

1. Confirm estimator $\hat{\theta}$ is unbiased: Verify $\mathbb{E}[\hat{\theta}] = \theta$.
2. Compute Fisher info $I(\theta)$ using Recipe 1.
3. Apply bound: $\Var(\hat{\theta}) \geq 1/I(\theta)$ (scalar) or $\Cov(\hat{\theta}) \succeq I(\theta)^{-1}$ (matrix, check positive semidefinite).
4. For functions: If estimating $g(\theta)$, use $\Var(g(\hat{\theta})) \geq (\nabla g)^\T I^{-1} (\nabla g)$ via delta method.
5. Check efficiency: Equality holds if $\hat{\theta}$ is affine in the score (e.g., for Gaussian mean, $\hat{\mu} = \bar{x}$ achieves CRLB with $\Var = \sigma^2/n = 1/(n I(\mu))$).
6. Exam tip: If bound not achieved, discuss reasons (e.g., non-regular model, biased estimator).

### Recipe 3: Linking CRLB to Representations in ML

Used for: Conceptual questions on info bounds in feature spaces (e.g., kernel representations).

1. Define representation: Map data to features $\phi(x)$ (e.g., polynomial kernel for non-linear est.).
2. Model likelihood in feature space: $p(y|\phi(x),\theta)$ (e.g., linear regression: $y = \theta^\T \phi(x) + \epsilon$).
3. Compute Fisher info in represented space: $I(\theta) = \mathbb{E}[\phi(X) \phi(X)^\T]/\sigma^2$ (for Gaussian noise).
4. Apply CRLB: Bound variance of $\hat{\theta}$, e.g., $\Var(\hat{\theta}) \geq \sigma^2 (\mathbb{E}[\phi \phi^\T])^{-1}$.
5. Interpret for ML: Higher info (trace of $I$) means better representations for estimation; link to overfitting (e.g., high-dim $\phi$ increases variance unless regularized).
6. Exam tip: Compare to VC dimension—CRLB gives statistical limit, VC gives learnability.

### Recipe 4: Deriving CRLB for Multivariate Gaussian

Used for: Computation-heavy exam problems (e.g., mean/covariance estimation).

1. Model: $X \sim \mathcal{N}(\mu, \Sigma)$, log-likelihood $\ell = -\frac{n}{2} \log|\Sigma| - \frac{1}{2} \sum (x_i - \mu)^\T \Sigma^{-1} (x_i - \mu)$.
2. Scores: $\frac{\partial \ell}{\partial \mu} = \Sigma^{-1} \sum (x_i - \mu)$; for $\Sigma$, use vec/trick for matrix deriv.
3. Fisher: $I(\mu) = n \Sigma^{-1}$; for $\Sigma$, block-diagonal form.
4. CRLB: $\Var(\hat{\mu}) \geq \Sigma/n$; for sample cov, Wishart-based bound.
5. Check: MLE achieves for mean (efficient); not always for cov.
6. Exam tip: Use for toy data—plug in numbers, compute bound, compare to actual var.

These recipes cover ~80% of exam problem types in this section based on past patterns. If you provide excerpts from the files, I can refine further!

## Week 4 GP

I've formatted this as a markdown list for clarity. Each "recipe" focuses on exam-relevant fundamentals, such as deriving formulas or solving toy problems (common in ETH exams). These are step-by-step, like cheat-sheet instructions for quick recall.

### 1. **Recipe: Compute GP Posterior Predictive Distribution (Regression)**

- **When to use**: For questions asking to derive mean/variance for new points given training data.
- **Steps**:
  1. Define the model: $y_i = f(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, $f \sim \mathcal{GP}(0, k)$.
  2. Compute kernel matrices: $\mathbf{K}_{\mathbf{XX}}$ (train-train), $\mathbf{K}_{*\mathbf{X}}$ (test-train), $\mathbf{K}_{**}$ (test-test), $\mathbf{K}_{\mathbf{X}*}$ (train-test, transpose of $\mathbf{K}_{*\mathbf{X}}$).
  3. Add noise: $\tilde{\mathbf{K}} = \mathbf{K}_{\mathbf{XX}} + \sigma_n^2 \mathbf{I}$.
  4. Compute mean: $\bar{\mathbf{f}}_* = \mathbf{K}_{*\mathbf{X}} \tilde{\mathbf{K}}^{-1} \mathbf{y}$ (use Cholesky: Solve $\tilde{\mathbf{K}} \boldsymbol{\alpha} = \mathbf{y}$, then $\bar{\mathbf{f}}_* = \mathbf{K}_{*\mathbf{X}} \boldsymbol{\alpha}$).
  5. Compute covariance: $\cov(\mathbf{f}_*) = \mathbf{K}_{**} - \mathbf{K}_{*\mathbf{X}} \tilde{\mathbf{K}}^{-1} \mathbf{K}_{\mathbf{X}*}$.
  6. For prediction: Sample or use $\bar{\mathbf{f}}_* \pm 2\sqrt{\diag(\cov(\mathbf{f}_*))}$ for 95% CI.
- **Exam Tip**: If kernel params unknown, mention optimizing log-marginal via gradients.

### 2. **Recipe: Optimize GP Hyperparameters (Log-Marginal Likelihood)**

- **When to use**: For questions on model selection or deriving gradients.
- **Steps**:
  1. Write log-marginal: $\log p(\mathbf{y}) = -\frac{1}{2} \mathbf{y}^\T \tilde{\mathbf{K}}^{-1} \mathbf{y} - \frac{1}{2} \log |\tilde{\mathbf{K}}| - \frac{n}{2} \log 2\pi$.
  2. Compute gradient w.r.t. $\theta$ (e.g., $\ell$ in RBF): $\frac{\partial \log p}{\partial \theta} = -\frac{1}{2} \tr\left( (\tilde{\mathbf{K}}^{-1} - \boldsymbol{\alpha} \boldsymbol{\alpha}^\T) \frac{\partial \tilde{\mathbf{K}}}{\partial \theta} \right)$, where $\boldsymbol{\alpha} = \tilde{\mathbf{K}}^{-1} \mathbf{y}$.
  3. Use GD/SGD to maximize (or minimize negative).
  4. Check for overfitting: Validate on holdout set if n small.
- **Exam Tip**: Derive $\frac{\partial k}{\partial \ell}$ for RBF: Multiply by $\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{\ell^3}$.

### 3. **Recipe: Train an Ensemble via Bagging (e.g., Random Forest)**

- **When to use**: For conceptual or step-by-step questions on variance reduction.
- **Steps**:
  1. Bootstrap: Generate $M$ datasets by sampling n points with replacement from training set.
  2. For each dataset: Train base learner (e.g., decision tree with random subset of d features at each split).
  3. Predict: Average (regression) or majority vote (classification) over $M$ models.
  4. Evaluate: Compute OOB error (predictions on unsampled points).
  5. Feature importance: Sum impurity decreases (e.g., Gini) across trees.
- **Exam Tip**: Explain variance reduction: Bagging averages uncorrelated errors.

### 4. **Recipe: Apply Boosting (e.g., AdaBoost for Classification)**

- **When to use**: For deriving weights or error bounds in sequential learning questions.
- **Steps**:
  1. Initialize weights: $w_i = 1/n$ for all i.
  2. For m=1 to M: Train weak learner $h_m$ on weighted data, compute error $\epsilon_m = \sum_{i: h_m(\mathbf{x}_i) \neq y_i} w_i$.
  3. If $\epsilon_m > 0.5$, stop or restart.
  4. Compute alpha: $\alpha_m = \frac{1}{2} \log \frac{1 - \epsilon_m}{\epsilon_m}$.
  5. Update weights: $w_i \leftarrow w_i \exp(-\alpha_m y_i h_m(\mathbf{x}_i))$, normalize $\sum w_i = 1$.
  6. Final model: $H(\mathbf{x}) = \sign\left(\sum_m \alpha_m h_m(\mathbf{x})\right)$.
- **Exam Tip**: Derive bound: Training error $\leq \prod_m 2 \sqrt{\epsilon_m (1 - \epsilon_m)}$; minimizes exponential loss.

### 5. **Recipe: Analyze Ensemble Bias-Variance Tradeoff**

- **When to use**: Theoretical questions comparing bagging vs. boosting.
- **Steps**:
  1. Recall decomposition: Error = Bias$^2$ + Variance + Noise.
  2. For bagging: Reduces variance by averaging high-variance learners (e.g., deep trees).
  3. For boosting: Reduces bias by focusing on errors; can increase variance if overfit.
  4. Compute for toy: E.g., variance of average = (1/M) Var(single) if uncorrelated.
  5. Trick: Add diversity (e.g., random subspaces) to decorrelate.
- **Exam Tip**: Prove for AdaBoost: Weight updates emphasize hard examples, leading to margin maximization.

## Week 5 SVM

### Recipe 1: Deriving the SVM Dual Problem (Hard/Soft-Margin)

1. **Write Primal**: Start with objective (min \(\frac{1}{2} \|\mathbf{w}\|^2 + C \sum \xi_i\)) and constraints (\(y_i (\mathbf{w}^\T \mathbf{x}\_i + b) \geq 1 - \xi_i\), \(\xi_i \geq 0\)).
2. **Introduce Lagrange Multipliers**: Form Lagrangian \(L = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum \xi_i - \sum \alpha_i [y_i (\mathbf{w}^\T \mathbf{x}_i + b) - 1 + \xi_i] - \sum \mu_i \xi_i\).
3. **Take Partial Derivatives**: Set ∇L w.r.t. \(\mathbf{w}, b, \xi_i\) to 0: \(\mathbf{w} = \sum \alpha_i y_i \mathbf{x}\_i\); \(\sum \alpha_i y_i = 0\); \(\alpha_i + \mu_i = C\).
4. **Substitute Back**: Plug into L to get dual max \(\sum \alpha*i - \frac{1}{2} \sum*{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}\_i^\T \mathbf{x}\_j\), s.t. \(0 \leq \alpha_i \leq C\).
5. **Kernelize if Needed**: Replace inner products with \(k(\cdot, \cdot)\).
6. **Exam Tip**: Identify support vectors (\(\alpha_i > 0\)); compute \(b\) from any SV: \(b = y_i - \mathbf{w}^\T \mathbf{x}\_i\).

### Recipe 2: Computing SVM Margin or Decision Boundary

1. **Classify Data**: Ensure labels \(y_i \in \{-1, +1\}\); check separability.
2. **Solve for \(\mathbf{w}, b\)**: From dual solution (\(\alpha_i\)), \(\mathbf{w} = \sum \alpha_i y_i \mathbf{x}\_i\); margin \(\gamma = 1 / \|\mathbf{w}\|\) (hard) or adjust for slack.
3. **Decision Function**: \(f(\mathbf{x}) = \sign(\mathbf{w}^\T \mathbf{x} + b)\).
4. **Toy Data Example**: For 2D points, plot hyperplane \(\mathbf{w}^\T \mathbf{x} + b = 0\); verify margins (distance from plane to SVs = \(\gamma\)).
5. **Exam Tip**: For non-linear, apply kernel (e.g., RBF) and explain feature space lift.

### Recipe 3: Applying Kernel Trick to a Problem

1. **Check Linearity**: If data non-separable in input space, choose kernel (e.g., RBF for clusters).
2. **Form Kernel Matrix**: Compute \(K\_{ij} = k(\mathbf{x}\_i, \mathbf{x}\_j)\) (must be PSD).
3. **Plug into Dual**: Use in SVM dual or Kernel PCA (e.g., eigenvalues of centered K).
4. **Predict**: \(f(\mathbf{x}) = \sign(\sum \alpha_i y_i k(\mathbf{x}\_i, \mathbf{x}) + b)\).
5. **Exam Tip**: Justify kernel choice (e.g., polynomial for degree-d relations); compute small matrix for toy data.

### Recipe 4: Building/ Analyzing an Ensemble (Bagging or Random Forest)

1. **Bootstrap Samples**: Generate B datasets by sampling with replacement (size n from n data).
2. **Train Base Models**: Fit high-variance learners (e.g., unpruned trees) on each; for RF, subsample features at splits.
3. **Aggregate**: Average predictions (regression) or vote (classification); compute OOB error on unused samples.
4. **Variance Reduction**: Explain: Averaging uncorrelated models ↓ variance by 1/B.
5. **Exam Tip**: On toy data, compute ensemble prediction and compare to single model error.

### Recipe 5: Running AdaBoost or Gradient Boosting Iteration

1. **Initialize**: Weights \(w_i = 1/n\); initial model \(f_0 = 0\) (GB) or uniform (AdaBoost).
2. **Train Weak Learner**: Fit \(h_t\) to weighted data; compute error \(\epsilon_t = \sum w_i \mathbf{1}(y_i \neq h_t(\mathbf{x}\_i))\).
3. **Update**: For AdaBoost, \(\alpha*t = \frac{1}{2} \ln((1-\epsilon_t)/\epsilon_t)\); reweight \(w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(\mathbf{x}\_i))\). For GB, fit to residuals \(r_i = y_i - f*{t-1}(\mathbf{x}\_i)\), add \(\nu h_t\).
4. **Final Model**: Sum weighted predictions; stop if error < threshold.
5. **Exam Tip**: Compute 1–2 iterations on toy data; discuss overfitting (e.g., monitor validation error).

### Recipe 6: Comparing SVM vs. Ensembles (Conceptual Question)

1. **Identify Strengths**: SVM: Good for high-dim data, margins → generalization. Ensembles: Handle non-linearity, reduce bias/variance via aggregation.
2. **When to Use**: SVM for small/medium data with clear margins; ensembles for large/noisy data (e.g., RF robust to outliers).
3. **Bias-Variance Link**: SVM controls variance via regularization (C); boosting ↓ bias, bagging ↓ variance.
4. **Exam Tip**: Reference bounds (e.g., AdaBoost exponential decay) or examples (e.g., SVM fails on XOR without kernel, ensembles handle via trees).

## Week 6 NN

### Recipe 1: Deriving Backpropagation Gradients for an MLP

1. **Define the Network**: Write the forward pass for each layer: $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$, $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$. Set $\mathbf{a}^{(0)} = \mathbf{x}$, output $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$.
2. **Choose Loss**: E.g., MSE: $\mathcal{L} = \frac{1}{2} \|\hat{\mathbf{y}} - \mathbf{y}\|^2$ or CE with softmax.
3. **Compute Output Delta**: $\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}} = (\hat{\mathbf{y}} - \mathbf{y}) \odot \sigma'(\mathbf{z}^{(L)})$.
4. **Propagate Backwards**: For $l = L-1$ to $1$: $\delta^{(l)} = (\mathbf{W}^{(l+1)\T} \delta^{(l+1)}) \odot \sigma'(\mathbf{z}^{(l)})$.
5. **Weight Gradients**: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} \mathbf{a}^{(l-1)\T}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}$.
6. **Update (if asked)**: Apply GD: $\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$.
   - **Exam Tip**: Watch for activation derivatives; use chain rule explicitly for proofs. Common pitfall: Forgetting transpose in propagation.

### Recipe 2: Computing Scaled Dot-Product Attention

1. **Input Setup**: Given sequences $\mathbf{Q} (n \times d_k)$, $\mathbf{K} (m \times d_k)$, $\mathbf{V} (m \times d_v)$.
2. **Scores**: Compute raw scores $\mathbf{S} = \mathbf{Q} \mathbf{K}^\T / \sqrt{d_k}$.
3. **Mask (if needed)**: For causal, set future positions in $\mathbf{S}$ to $-\infty$.
4. **Softmax**: $\mathbf{A} = \softmax(\mathbf{S})$ (row-wise).
5. **Output**: $\mathbf{O} = \mathbf{A} \mathbf{V}$.
6. **For Self-Attention**: Set $\mathbf{Q} = \mathbf{K} = \mathbf{V}$ (project input if needed).
   - **Exam Tip**: For toy examples (e.g., 2 tokens), compute numerically. Derive why scaling $\sqrt{d_k}$ stabilizes gradients (variance argument).

### Recipe 3: Building/Explaining a Transformer Layer

1. **Input**: Embed sequence + add positional encoding: $\mathbf{X} = \text{Embed}(\text{tokens}) + \text{PE}$.
2. **Self-Attention Block**: Compute multi-head attention on $\mathbf{X}$, add residual: $\mathbf{X}' = \mathbf{X} + \text{MultiHead}(\mathbf{X})$.
3. **LayerNorm**: $\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}')$.
4. **FFN Block**: $\mathbf{X}''' = \mathbf{X}'' + \text{FFN}(\mathbf{X}'')$, then another LayerNorm.
5. **For Decoder**: Add masked self-attn, then enc-dec attn (queries from decoder, keys/values from encoder).
6. **Output/Generation**: For inference, use argmax or sampling from softmax.
   - **Exam Tip**: Compare to RNN: Transformers are parallelizable, better for long seqs. Derive positional encoding to show wavelength variation.

### Recipe 4: Handling Common Tricks/Optimizations

1. **Vanishing Gradients**: Identify issue (e.g., sigmoid saturation), suggest fix (ReLU/LSTM).
2. **Initialization**: For layer with $n_{in}$ inputs, init $\mathbf{W} \sim \mathcal{N}(0, \sqrt{2/n_{in}})$ for ReLU.
3. **Regularization**: Add L2 term to loss: $\mathcal{L} + \lambda \|\mathbf{W}\|^2$; or dropout (mask neurons randomly).
4. **Bias-Variance Check**: If overfitting, increase regularization or data.
   - **Exam Tip**: For questions like "Why does attention work?", explain weighted averaging of values based on query-key similarity.

## Week 7 CV

### Recipe 1: Computing Convolution Output (e.g., for a Given Input and Kernel)

1. **Identify inputs**: Note input size ($H \times W \times C$), kernel size ($k_h \times k_w \times C$), stride $s$, padding $p$.
2. **Calculate output size**: Height = $\lfloor (H - k_h + 2p)/s \rfloor + 1$; same for width.
3. **Apply convolution**: For each output position $[i,j]$, sum over kernel: $O[i,j] = \sum_{m,n,c} I[i \cdot s + m - p, j \cdot s + n - p, c] \cdot K[m,n,c] + b$. Handle boundaries with padding (zero-pad if unspecified).
4. **Add activation/pooling if specified**: E.g., apply ReLU or max-pool.
5. **Verify**: Check for translation invariance (output shifts if input shifts).

### Recipe 2: Deriving Backpropagation Gradients in a CNN Layer

1. **Recall chain rule**: Gradient of loss $\mathcal{L}$ w.r.t. layer params = output gradient $\delta_o$ propagated back.
2. **For weights $K$**: $\frac{\partial \mathcal{L}}{\partial K[m,n,c]} = \sum_{i,j} \delta_o[i,j] \cdot I[i+m,j+n,c]$ (convolve input with $\delta_o$).
3. **For input $I$**: $\frac{\partial \mathcal{L}}{\partial I[x,y,d]} = \sum_{m,n} \delta_o[\lfloor (x-m)/s \rfloor, \lfloor (y-n)/s \rfloor] \cdot K[m,n,d]$ if in bounds (full convolution with rotated kernel).
4. **For bias $b$**: $\frac{\partial \mathcal{L}}{\partial b} = \sum_{i,j} \delta_o[i,j]$.
5. **Propagate to previous layer**: $\delta_i = $ upsampled/rotated conv of $\delta_o$ with $K$.
6. **Exam tip**: If non-linearity (e.g., ReLU), multiply by derivative (e.g., 1 if $x>0$, else 0).

### Recipe 3: Designing/Explaining a CNN Architecture for a Task (e.g., Classification vs. Detection)

1. **Analyze task**: Classification? Use conv + pool + FC + softmax. Detection? Add bounding box regression (e.g., like in Faster R-CNN).
2. **Stack layers**: Start with conv (extract features), pool (reduce dims), repeat; end with FC for output.
3. **Choose hyperparameters**: Kernel size (3x3 common), stride=1-2, padding='same' for size preservation.
4. **Incorporate tricks**: Add batch norm after conv, dropout before FC to prevent overfitting.
5. **Justify**: CNNs efficient due to param sharing (fewer weights than FC); handle spatial hierarchy.
6. **Evaluate**: Compute params (# = $k_h k_w C_{in} C_{out}$ per conv); use cross-entropy loss, IoU for detection.

### Recipe 4: Handling Output Shape and Padding/Stride Effects

1. **Input specs**: $H \times W \times C_{in}$, kernel $k \times k \times C_{in} \times C_{out}$.
2. **With stride $s$, padding $p$**: Output $H' = \lfloor (H + 2p - k)/s \rfloor + 1$.
3. **For 'valid' (no pad)**: $p=0$, shrinks output.
4. **For 'same'**: $p = \lfloor (k-1)/2 \rfloor$, preserves size if $s=1$.
5. **Exam check**: If pooling follows, repeat calc (e.g., 2x2 max-pool halves dims).
6. **Trick**: Use to control feature map size for deeper nets.

## Week 8 GNN

### 1. Deriving a GCN Layer Update

- **When**: Questions like "Derive the forward pass for a 2-layer GCN on a graph with given A and X."
- Steps:
  1. Write the graph notation: Adjacency A, features X = H^(0), degree D.
  2. Add self-loops: \tilde{A} = A + I.
  3. Normalize: \hat{A} = D^{-1/2} \tilde{A} D^{-1/2} (or asymmetric variant if specified).
  4. Layer 1: H^(1) = σ(\hat{A} H^(0) W^(0)), where σ is activation (e.g., ReLU).
  5. Layer l+1: Generalize to H^(l+1) = σ(\hat{A} H^(l) W^(l)).
  6. For output: Apply softmax if classification; explain spectral filtering if asked.

### 2. Computing Attention in GAT

- **When**: "Compute attention coefficients for node i with neighbors j,k."
- Steps:
  1. Get node features h_i, h_j, etc., and weight matrix W.
  2. Compute score e\_{ij} = LeakyReLU(a^T [W h_i || W h_j]), where || is concat, a is attention vector.
  3. Softmax over neighbors: α*{ij} = exp(e*{ij}) / ∑*{k in N(i)∪i} exp(e*{ik}).
  4. Update: h*i' = σ(∑_j α*{ij} W h_j).
  5. For multi-head: Repeat for K heads, concat or average outputs.
  6. Tip: Handle self-loops by including i in neighbors.

### 3. Solving Graph Classification with Readout

- **When**: "Design a GNN for graph-level prediction; compute embedding for a small graph."
- Steps:
  1. Run GNN layers to get node embeddings H^(L).
  2. Choose readout: e.g., global mean pooling r = (1/n) ∑_v h_v.
  3. Pass r through MLP for prediction (e.g., class logits).
  4. If hierarchical: Use pooling layers to coarsen graph (e.g., assign nodes to clusters).
  5. Evaluate: Discuss loss (e.g., cross-entropy for classification).
  6. Trick: For imbalanced graphs, use sum or attention-based readout.

### 4. Computing Entropy or Mutual Information

- **When**: "Calculate H(X) for a discrete distribution" or "Show I(X;Y) = 0 if independent."
- Steps:
  1. For entropy H(X): List probabilities p(x_i), compute -∑ p(x_i) log p(x_i) (use log2 for bits, ln for nats).
  2. For conditional H(Y|X): Average H(Y|x) over p(x), or use H(X,Y) - H(X).
  3. For I(X;Y): Compute H(X) - H(X|Y), or ∑∑ p(x,y) log [p(x,y)/(p(x)p(y))].
  4. Prove properties: e.g., I(X;Y) ≥ 0 by KL non-negativity; =0 iff independent.
  5. Continuous case: Replace sum with integral; handle differentials.
  6. Tip: Use chain rule for joint entropies in sequences/graphs.

### 5. Applying Information Theory to GNNs (e.g., InfoMax Objective)

- **When**: "Derive a loss to maximize mutual information between node and graph representations."
- Steps:
  1. Define representations: Local h_v from node, global h_G from readout.
  2. Objective: Maximize I(h_v; h_G) ≈ ∑ positive pairs - ∑ negative pairs (contrastive, e.g., Noise Contrastive Estimation).
  3. Estimator: Use Jensen-Shannon or InfoNCE loss: L = -∑ log [σ(score(h_v, h_G)) / (σ(score(h_v, h_G)) + ∑*{neg} σ(score(h_v, h*{neg})))].
  4. In GNN: Encode graph with GNN, generate negatives (e.g., shuffled graphs).
  5. Train: Minimize L to learn unsupervised embeddings.
  6. Discuss: Helps in semi-supervised settings; relates to ELBO if variational.

### 6. Proving Properties (e.g., Over-Smoothing in GNNs)

- **When**: "Prove that repeated GCN layers lead to over-smoothing."
- Steps:
  1. Recall update: H^(l+1) = \hat{A} H^(l) (ignore W, σ for simplicity).
  2. Iterative: H^(∞) converges to stationary distribution (proportional to degrees).
  3. Show variance decreases: Embeddings become similar as l increases (e.g., via Dirichlet energy or Laplacian smoothing).
  4. Mitigation: Add residuals H^(l+1) = H^(l) + σ(\hat{A} H^(l) W).
  5. Quantitative: Compute ||h_i - h_j|| → 0 for connected i,j.
  6. Tip: Reference spectral view—low-pass filtering removes high frequencies.

## Week 9 Anomaly

### 1. **Computing Z-Score or Mahalanobis for Anomaly Detection**

- **When to use**: For statistical outliers in low-dimensional data.
- Steps:
  1. Compute mean $\mu$ and std. dev. $\sigma$ (or covariance $\Sigma$) from normal data.
  2. For test point $\mathbf{x}$, calculate score: $z = \frac{x - \mu}{\sigma}$ or $D_M = \sqrt{(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)}$.
  3. Set threshold $\theta$ (e.g., 3 for z-score, or $\chi^2$ quantile).
  4. Classify: Anomaly if score > $\theta$.
  5. **Exam trick**: If multivariate, justify why Mahalanobis > Euclidean (accounts for correlations).

### 2. **Calculating Local Outlier Factor (LOF)**

- **When to use**: For density-based detection in clustered data.
- Steps:
  1. Choose $k$ (e.g., 5–20); compute distances $d(p,o)$ for all pairs.
  2. Find $N_k(p)$: $k$ nearest neighbors of $p$.
  3. Compute reachability: $\text{rd}_k(p,o) = \max(d_k(o), d(p,o))$ for each $o \in N_k(p)$.
  4. Local density: $\text{lrd}_k(p) = \left( \frac{\sum_{o} \text{rd}_k(p,o)}{|N_k(p)|} \right)^{-1}$.
  5. LOF: $\text{LOF}_k(p) = \frac{\sum_{o} \text{lrd}_k(o) / \text{lrd}_k(p)}{|N_k(p)|}$.
  6. Interpret: >1 = outlier (lower density than neighbors).
  - **Exam trick**: On toy 2D data, sketch neighborhoods; derive why LOF=1 for uniform density.

### 3. **Building/Scoring with Isolation Forest**

- **When to use**: High-dim, unsupervised anomalies.
- Steps:
  1. Build tree: Randomly select feature and split value until isolation or depth limit.
  2. Repeat for ensemble (e.g., 100 trees).
  3. For $\mathbf{x}$, compute avg path length $E(h(\mathbf{x}))$.
  4. Normalize: $s = 2^{-E(h)/c(n)}$, with $c(n) \approx 2\ln(n-1) + 0.577 - 2(n-1)/n$.
  5. Classify: $s > 0.6$ often anomaly.
  - **Exam trick**: Derive $c(n)$ from BST avg height; explain why short paths indicate anomalies.

### 4. **Setting Up One-Class SVM**

- **When to use**: Semi-supervised, kernel-based boundaries.
- Steps:
  1. Map data to feature space $\phi(\mathbf{x})$ (e.g., RBF kernel).
  2. Solve optimization: Minimize $\frac{1}{2}\|\mathbf{w}\|^2 - \rho + \frac{1}{\nu n} \sum \xi_i$, s.t. constraints.
  3. Decision function: $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) - \rho < 0$ = anomaly.
  4. Tune $\nu$ (outlier fraction) via cross-val on normal data.
  - **Exam trick**: Derive dual form if asked; compare to standard SVM (origin as "negative" class).

### 5. **Anomaly Detection with Autoencoders**

- **When to use**: Deep learning for complex data (e.g., images).
- Steps:
  1. Train AE on normal data: Minimize $L = ||\mathbf{x} - \hat{\mathbf{x}}||^2$.
  2. Compute reconstruction error for test $\mathbf{x}$.
  3. Threshold: Mean + 3$\sigma$ from validation errors.
  4. Classify high-error points as anomalies.
  - **Exam trick**: Link to variational AE if probabilistic; discuss overfitting to normals.

### 6. **Evaluating Anomaly Detectors**

- **When to use**: Compare methods or tune thresholds.
- Steps:
  1. Generate scores for labeled data (if available).
  2. Compute ROC/PR curves; AUC-PR better for rare anomalies.
  3. Choose threshold at desired precision/recall.
  4. Cross-validate on normal data for semi-supervised.
  - **Exam trick**: Explain bias in AUC-ROC for imbalance; use F1 if binary classification.

## Week 10 RL

### 1. Solving MDP Optimality (e.g., Derive/Apply Bellman Equations)

- **Step 1:** Define the MDP components: States $\mathcal{S}$, actions $\mathcal{A}$, $P$, $R$, $\gamma$.
- **Step 2:** Write the value function: $V^\pi(s) = \mathbb{E} [\sum_t \gamma^t r_t | s_0=s, \pi]$.
- **Step 3:** Derive Bellman expectation: Unroll one step to get $V^\pi(s) = \sum_a \pi(a|s) [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')]$.
- **Step 4:** For optimality, replace sum with max: $V^*(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$.
- **Step 5:** Extract policy: $\pi^*(s) = \argmax_a Q^*(s,a)$.
- **Fundamentals Tip:** If infinite horizon, justify convergence with $\gamma < 1$. Exam trick: Prove it's a fixed point.

### 2. Value Iteration on a Small MDP (e.g., Compute Optimal Values)

- **Step 1:** Initialize $V_0(s) = 0$ for all $s$ (or arbitrary).
- **Step 2:** For each iteration $k$: $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a) + \gamma V_k(s')]$.
- **Step 3:** Repeat until convergence (e.g., $\|V_{k+1} - V_k\|_\infty < \epsilon$).
- **Step 4:** Derive policy from final $V^*$: $\pi(s) = \argmax_a \sum_{s'} P(s'|s,a) [R(s,a) + \gamma V^*(s')]$.
- **Fundamentals Tip:** For toy grids, compute manually for 2-3 iterations. Exam: Discuss why it's model-based.

### 3. Policy Gradient Derivation/Update (e.g., REINFORCE on a Problem)

- **Step 1:** Parameterize policy $\pi_\theta(a|s)$ (e.g., softmax).
- **Step 2:** Objective $J(\theta) = \mathbb{E}_\pi [G]$; gradient $\nabla J = \mathbb{E} [\nabla \log \pi(a|s) G]$ (from policy grad thm).
- **Step 3:** Sample trajectory: Generate $(s_t, a_t, r_t)$ under $\pi_\theta$.
- **Step 4:** Compute est. gradient: $\hat{\nabla} J = \sum_t \nabla_\theta \log \pi(a_t|s_t) (G_t - b(s_t))$ (baseline for variance reduction).
- **Step 5:** Update $\theta \leftarrow \theta + \alpha \hat{\nabla} J$.
- **Fundamentals Tip:** Derive from log-trick: $\nabla \mathbb{E}[f] = \mathbb{E}[f \nabla \log p]$. Exam: Add advantage for actor-critic.

### 4. Q-Learning/SARSA Application (e.g., Update on a Sequence)

- **Step 1:** Initialize $Q(s,a) = 0$; choose $\alpha, \epsilon, \gamma$.
- **Step 2:** For each episode: Start at $s$, select $a$ ($\epsilon$-greedy from $Q$ for Q-Learning; from policy for SARSA).
- **Step 3:** Observe $r, s'$; for Q-Learning: target = $r + \gamma \max_{a'} Q(s',a')$; for SARSA: select $a'$ from policy, target = $r + \gamma Q(s',a')$.
- **Step 4:** Update $Q(s,a) \leftarrow Q(s,a) + \alpha ($target$ - Q(s,a))$.
- **Step 5:** Repeat until convergence. Exam: Compare off-policy (Q) vs. on-policy (SARSA) safety.

### 5. Active Learning Query Selection (e.g., Choose Point to Label)

- **Step 1:** Define setting: Labeled $\mathcal{L}$, unlabeled $\mathcal{U}$, model $f$ (e.g., classifier).
- **Step 2:** For uncertainty sampling: Compute $P(y|x)$ for each $x \in \mathcal{U}$; select $x^* = \argmax_x (0.5 - |P(y=1|x) - 0.5|)$ (margin) or entropy $H(y|x) = -\sum_y P(y|x) \log P(y|x)$.
- **Step 3:** For query-by-committee: Train $C$ models on bootstraps of $\mathcal{L}$; compute disagreement (e.g., vote entropy).
- **Step 4:** Query label for $x^*$, add to $\mathcal{L}$, retrain.
- **Fundamentals Tip:** Justify with info gain. Exam: Compute on toy data (e.g., 3 points, binary labels); discuss vs. passive learning.

## Week 11 CI

### 1. **Proving a Kernel is Positive Definite (PD)**

*Problem Type*: Theoretical proof (e.g., "Show RBF kernel is PD"). Relevant: Tests Mercer's theorem understanding; appears in 2/6 past exams.
*Steps*:

1. Recall definition: $K$ is PD if for any finite set $\{x_1, \dots, x_n\}$ and $\mathbf{c} \in \mathbb{R}^n \setminus \{0\}$, $\mathbf{c}^\T \mathbf{K} \mathbf{c} \geq 0$ (Gram matrix $\mathbf{K}_{ij} = K(x_i, x_j)$).
2. For given kernel (e.g., RBF): Express as $K(x,y) = \int \phi(x) \phi(y) d\mu$ or use known decomposition (Mercer: eigenvalues $\lambda_k \geq 0$).
3. Compute $\sum_{i,j} c_i c_j K(x_i, x_j)$ and show it's non-negative (e.g., for RBF, it's Fourier transform of positive function).
4. Conclude with Mercer's condition if symmetric and continuous.
   *Tip*: If stuck, check closure properties (sums/products of PD kernels are PD).

### 2. **Deriving the Reproducing Property or RKHS Norm**

*Problem Type*: Derivation (e.g., "Derive $\|f\|_{\mathcal{H}}^2$ for finite expansion"). Relevant: Core to kernel methods; frequent in derivations (4/6 exams).
*Steps*:

1. Assume $f \in \mathcal{H}$, with reproducing property: $f(x) = \langle f, K(\cdot, x) \rangle_{\mathcal{H}}$.
2. For finite rep. $f(\cdot) = \sum_{i=1}^n \alpha_i K(\cdot, x_i)$, compute inner product: $\langle f, f \rangle_{\mathcal{H}} = \sum_{i,j} \alpha_i \alpha_j \langle K(\cdot, x_i), K(\cdot, x_j) \rangle_{\mathcal{H}}$.
3. Apply reproducing: $\langle K(\cdot, x_i), K(\cdot, x_j) \rangle_{\mathcal{H}} = K(x_i, x_j)$.
4. Simplify to $\|f\|_{\mathcal{H}}^2 = \boldsymbol{\alpha}^\T \mathbf{K} \boldsymbol{\alpha}$.
5. Extend to infinite (Mercer basis): $f = \sum_k \beta_k \sqrt{\lambda_k} e_k$, $\|f\|^2 = \sum_k \beta_k^2$.
   *Tip*: Use this for regularization terms in SVM/kernel regression proofs.

### 3. **Solving Kernel Ridge Regression (KRR)**

*Problem Type*: Computation (e.g., "Fit KRR on dataset $\{ (x_i, y_i) \}$ and predict for $x_*$"). Relevant: Practical application; tested via steps (3/6 exams).
*Steps*:

1. Form Gram matrix $\mathbf{K}_{ij} = K(x_i, x_j)$ (choose kernel, e.g., RBF).
2. Solve for coefficients: $\boldsymbol{\alpha} = (\mathbf{K} + \lambda n \mathbf{I})^{-1} \mathbf{y}$ (from loss $\sum (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$).
3. For prediction: $f(x_*) = \sum_i \alpha_i K(x_*, x_i) = \mathbf{k}_*^\T \boldsymbol{\alpha}$.
4. If needed, compute RKHS norm of $\hat{f}$ as $\boldsymbol{\alpha}^\T \mathbf{K} \boldsymbol{\alpha}$.
   *Tip*: For toy data (e.g., n=2), compute manually; regularization $\lambda$ prevents overfitting.

### 4. **Checking/Computing Counterfactual Invariance**

*Problem Type*: Conceptual/application (e.g., "Is model invariant under do(X=x)? Compute P(Y|do(X))"). Relevant: Causal ML theory; emerging in recent exams (2/6).
*Steps*:

1. Draw causal graph (SCM) from problem (nodes: variables, edges: causal links).
2. Identify query: Interventional (P(Y|do(X=x))) or counterfactual (P(Y_x|y observed)).
3. Check identifiability: Use backdoor/frontdoor criteria (e.g., adjust for confounders Z if d-separated).
4. Compute: For backdoor, P(Y|do(X=x)) = \sum_z P(Y|x,z) P(z).
5. Test invariance: If E[Y|do(X=x)] = E[Y|X=x] (no effect), or model unchanged across environments.
   *Tip*: If graph has confounders, intervention != conditioning; use do-calculus rules (e.g., Rule 2 for insertion/deletion).

### 5. **Proving Invariance in Causal Models**

*Problem Type*: Proof (e.g., "Prove invariance under environment shift"). Relevant: Links to robustness; theoretical questions (1-2/6 exams).
*Steps*:

1. Define invariant predictor: f(X,E) such that E[Y|f(X),E] independent of E (environments).
2. Assume SCM with stable modules (e.g., Y = f(X) + noise, invariant to E).
3. Show via residuals: Var(Y - f(X)|E) constant across E.
4. Conclude: Invariant if no hidden confounders affect the relation.
   *Tip*: Contrast with non-invariant (e.g., spurious correlations fail under do(E=shift)).

## Week 12 VAE

### 1. Deriving the ELBO for a VAE

- **When**: Exam asks to derive variational lower bound or explain VAE objective.
- **Steps**:
  1. Start with marginal log-likelihood: $\log p_\theta(x) = \log \int p_\theta(x|z) p(z) dz$.
  2. Introduce variational posterior $q_\phi(z|x)$: Rewrite as $\log p_\theta(x) = \mathbb{E}_{q}[\log p_\theta(x|z)] + \mathbb{E}_{q}[\log \frac{p(z)}{q(z|x)}] + \KL(q(z|x) \| p(z|x))$.
  3. Note $\KL \geq 0$, so $\log p_\theta(x) \geq \ELBO = \mathbb{E}_{q}[\log p_\theta(x|z)] - \KL(q(z|x) \| p(z))$.
  4. Specify priors/likelihoods (e.g., $p(z) = \mathcal{N}(0,I)$, $p(x|z) = \mathcal{N}(\mu_\theta(z), I)$).
  5. Compute KL if Gaussian (use closed-form formula).
- **Tips**: Emphasize Jensen's inequality; for exams, derive from KL decomposition. Common trick: Assume diagonal covariance for $q$.

### 2. Applying Reparameterization Trick in VAEs

- **When**: Problem involves sampling from $q(z|x)$ for gradient computation.
- **Steps**:
  1. Parameterize encoder: $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \diag(\sigma_\phi^2(x)))$.
  2. Sample $\epsilon \sim \mathcal{N}(0,I)$.
  3. Set $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ (element-wise).
  4. Compute objective (e.g., $\log p_\theta(x|z)$) using this $z$.
  5. Backpropagate: Gradients flow through deterministic $\mu, \sigma$.
- **Tips**: Exam hint: Explains why VAEs are differentiable vs. non-variational models. Use 1 sample for low-variance estimates in training.

### 3. Computing GP Predictive Distribution

- **When**: Given training data $(X,y)$, predict for test $X_*$.
- **Steps**:
  1. Choose kernel $k$ (e.g., RBF) and compute matrices: $K = k(X,X) + \sigma^2 I$, $K_* = k(X_*,X)$, $K_{**} = k(X_*,X_*)$.
  2. Invert $K$ (or use Cholesky for stability: $L = \chol(K)$, solve $L \alpha = y$, etc.).
  3. Predictive mean: $\mu_* = K_* K^{-1} y$.
  4. Predictive covariance: $\Sigma_* = K_{**} - K_* K^{-1} K_*^T$.
  5. Sample or compute uncertainty: $f_* \sim \mathcal{N}(\mu_*,\Sigma_*)$.
- **Tips**: For small $n$ (exam toy problems), compute manually. Hint: Log-marginal for hyperparams via grid search or gradient ascent.

### 4. Explaining/Deriving Dirichlet Process Mixture

- **When**: Conceptual question on non-parametric clustering (e.g., infinite GMM).
- **Steps**:
  1. Define DP: $G \sim \DP(\alpha, H)$, base $H$ (e.g., prior over mixture params).
  2. Stick-breaking construction: Draw $v_k \sim \Beta(1,\alpha)$, set $\pi_k = v_k \prod_{j<k} (1-v_j)$, $\theta_k \sim H$.
  3. Model: For data $x_i$, assign $z_i \sim \Cat(\pi)$, $x_i \sim p(\cdot | \theta_{z_i})$.
  4. Inference: Use Gibbs sampling or variational (focus on posterior over clusters).
  5. Compare to finite mixtures: DP allows $K \to \infty$ as data grows.
- **Tips**: Exam fundamental: CRP analogy (customers join tables with prob $\propto$ size or new with $\alpha$). Useful for unknown #clusters.

### 5. Comparing VAEs to Other Models

- **When**: Short-answer on advantages/limitations (e.g., vs. GPs or standard autoencoders).
- **Steps**:
  1. State core: VAEs probabilistic (latent uncertainty) vs. deterministic autoencoders.
  2. Pros: Generative (sample new data), handles missing data via ELBO.
  3. Cons: Blurry recons (fix with better likelihoods); vs. GPs: VAEs scale better to high-dim (e.g., images) but less exact uncertainty.
  4. Link to non-parametrics: VAEs can use GP priors on latents for flexibility.
  5. Exam trick: Quantify with ELBO terms (reconstruction vs. regularization).
- **Tips**: Focus on Bayesian aspects—VAEs approximate inference, GPs exact but costly.

## Week 13 and 14

### 1. **Computing VC Dimension of a Hypothesis Class**

- **When to use**: Questions like "What is VC$(\mathcal{H})$ for linear classifiers in $\mathbb{R}^d$?" or "Prove VC = k for intervals."
- **Steps**:
  1. Recall definition: VC$(\mathcal{H})$ = max size of shatterable set $S$.
  2. Find a set of size $k$ that $\mathcal{H}$ shatters: Show all $2^k$ labelings are realizable (e.g., for intervals on $\mathbb{R}$, 2 points: all 4 labelings possible).
  3. Prove no larger set is shatterable: Show for size $k+1$, at least one labeling impossible (e.g., 3 points on line can't realize alternating labels for half-planes).
  4. If infinite, show shattering for arbitrarily large sets.
  5. Tip: Use geometric intuition (e.g., Radon theorem for linear separators).

### 2. **Proving PAC Learnability (Realizable Case)**

- **When to use**: "Show class $\mathcal{H}$ is PAC learnable" or "Is ERM consistent?"
- **Steps**:
  1. Check if $\mathcal{H}$ has finite VC-dim $d$ (or finite $|\mathcal{H}|$).
  2. State fundamental theorem: If finite VC, then PAC with sample complexity $O(\frac{d \ln(1/\epsilon) + \ln(1/\delta)}{\epsilon})$.
  3. For ERM: Show uniform convergence via VC bound: $|L(h) - \hat{L}(h)| \to 0$ as $m \to \infty$.
  4. Derive required $m$: Solve for $m$ s.t. bound $\leq \epsilon$ w.p. $1-\delta$ (e.g., using Hoeffding for finite case).
  5. Fundamentals: Assume realizability ($L(h^*)=0$); output $h$ with $\hat{L}(h)=0$.

### 3. **Deriving Sample Complexity Bounds**

- **When to use**: "What $m$ suffices for $\epsilon=0.1, \delta=0.05$ with VC=3?" or "Bound generalization error."
- **Steps**:
  1. Identify setting (realizable/agnostic, finite/infinite $\mathcal{H}$).
  2. For finite $|\mathcal{H}|$: Use $m \geq \frac{1}{\epsilon} (\ln|\mathcal{H}| + \ln(1/\delta))$.
  3. For VC $d$: Use upper bound $m = O\left( \frac{d \ln(m/\delta) + \ln(1/\delta)}{\epsilon} \right)$ (solve iteratively for $m$).
  4. Plug in numbers: E.g., approximate $\ln(m) \approx \ln(1/\epsilon)$ for large $m$.
  5. For agnostic: Square the epsilon (bound scales with $1/\epsilon^2$ due to variance).
  6. Trick: Use Sauer-Shelah to bound growth function if needed.

### 4. **Handling Agnostic PAC and ERM**

- **When to use**: "In agnostic setting, bound excess risk" or "Why does ERM work?"
- **Steps**:
  1. Recall: Minimize $\inf_{h \in \mathcal{H}} L(h) + \epsilon$.
  2. Use VC uniform convergence: Excess risk $\leq 2 \sup_h |L(h) - \hat{L}(h)| \leq O(\sqrt{d/m})$.
  3. Set bound $\leq \epsilon$: Solve $m \geq O(\frac{d + \ln(1/\delta)}{\epsilon^2})$.
  4. Prove ERM optimal: $\hat{h} = \argmin \hat{L}(h)$; bound $L(\hat{h}) \leq \min_h L(h) + 2\epsilon$.
  5. Fundamentals: No realizability; use Rademacher complexity for tighter bounds if asked.

### 5. **Computing Shattering or Growth Function**

- **When to use**: "Does $\mathcal{H}$ shatter this set?" or "Bound $\Pi_{\mathcal{H}}(m)$."
- **Steps**:
  1. For shattering: Enumerate all $2^{|S|}$ labelings; check if each has a $h \in \mathcal{H}$.
  2. For growth: If VC $d$, use $\Pi(m) \leq (em/d)^d$ for $m > d$.
  3. If $m \leq d$, $\Pi(m) = 2^m$.
  4. Trick: Use binomial sum for exact (Sauer): $\Pi(m) \leq \sum_{i=0}^d \binom{m}{i}$.
