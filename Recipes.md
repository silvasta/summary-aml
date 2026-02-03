<!-- markdownlint-configure-file { "MD013": { "line_length": 400} } -->
<!-- markdownlint-configure-file { "MD041": false } -->

# ML Recipes: Exam & Implementation Guide

## Week 1: Basics & Linear Models

### Recipe: Classify ML Problem Type

1. **Identify Data**: Labeled? (Supervised) Unlabeled? (Unsupervised) Rewards? (RL).
2. **Determine Goal**: Continuous value prediction (Regression), discrete category (Classification), finding structures (Clustering/Dimensionality Reduction), or sequential decision making (RL).
3. **Example**: "Group users by purchasing behavior" $\to$ Unsupervised clustering.

### Recipe: Compute Empirical Risk

1. **Given**: Dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, model $f$, loss $\ell$.
2. **Calculate**: $\hat{R}(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(\mathbf{x}_i), y_i)$.
3. **Regularization**: If specified, add $\lambda \Omega(f)$ (e.g., $\lambda \|\mathbf{w}\|_2^2$).
4. **Bias-Variance Note**: $MSE = \text{Bias}^2 + \text{Var} + \sigma^2$ (Irreducible noise).

### Recipe: Derive Closed-Form Linear Regression (OLS/Ridge)

1. **Objective**: Minimize $J(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2$.
2. **Expand**: $J(\mathbf{w}) = (\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w}^T \mathbf{w}$.
3. **Gradient**: $\nabla_{\mathbf{w}} J = 2\mathbf{X}^T\mathbf{X}\mathbf{w} - 2\mathbf{X}^T\mathbf{y} + 2\lambda \mathbf{w}$.
4. **Solve**: Set $\nabla_{\mathbf{w}} = 0 \implies \mathbf{w} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$.

---

## Week 3: Statistical Estimation & Information Theory

### Recipe 1: Maximum Likelihood Estimation (MLE)

1. **Likelihood**: $L(\theta) = \prod p(x_i | \theta)$.
2. **Log-Likelihood**: $\ell(\theta) = \sum \log p(x_i | \theta)$.
3. **Optimize**: Solve $\frac{\partial \ell}{\partial \theta} = 0$.
4. **Verify**: Check $\frac{\partial^2 \ell}{\partial \theta^2} < 0$ for maximum.

### Recipe 2: Computing Fisher Information $I(\theta)$

1. **Score**: $s(\theta) = \nabla_\theta \log p(x|\theta)$.
2. **Option A (Variance of Score)**: $I(\theta) = \mathbb{E}[s(\theta)^2]$.
3. **Option B (Hessian Form)**: $I(\theta) = -\mathbb{E}\left[ \frac{\partial^2}{\partial \theta^2} \log p(x|\theta) \right]$.
4. **IID Data**: Total info for $n$ samples is $n \cdot I(\theta)$.

### Recipe 3: Cramér-Rao Lower Bound (CRLB)

1. **Condition**: Estimator $\hat{\theta}$ must be unbiased ($\mathbb{E}[\hat{\theta}] = \theta$).
2. **Bound**: $\text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}$.
3. **Efficiency**: If $\text{Var}(\hat{\theta}) = \text{CRLB}$, the estimator is "efficient."

---

## Week 4 & 5: GPs, SVMs & Ensembles

### Recipe: Gaussian Process (GP) Posterior

1. **Kernel Matrices**: $K = k(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I}$, $K_* = k(\mathbf{X}, \mathbf{x}_*)$, $K_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$.
2. **Mean**: $\mu_* = K_*^T K^{-1} \mathbf{y}$.
3. **Variance**: $\sigma^2_* = K_{**} - K_*^T K^{-1} K_*$.

### Recipe: SVM Dual Formulation

1. **Primal**: $\min \frac{1}{2}\|\mathbf{w}\|^2 + C \sum \xi_i$ s.t. $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$.
2. **Dual**: $\max_{\alpha} \sum \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j)$.
3. **Constraints**: $0 \leq \alpha_i \leq C$ and $\sum \alpha_i y_i = 0$.
4. **Support Vectors**: Points where $\alpha_i > 0$. If $\alpha_i < C$, point lies on the margin.

### Recipe: AdaBoost Update

1. **Error**: $\epsilon_t = \sum w_i \mathbb{I}(y_i \neq h_t(x_i))$.
2. **Hypothesis Weight**: $\alpha_t = \frac{1}{2} \ln \frac{1-\epsilon_t}{\epsilon_t}$.
3. **Weight Update**: $w_{i} \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$, then normalize.

---

## Week 6 & 7: Neural Networks & CV

### Recipe: Backpropagation (Vectorized)

1. **Forward**: $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$, $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$.
2. **Output Error**: $\delta^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} \odot \sigma'(\mathbf{z}^{(L)})$.
3. **Hidden Error**: $\delta^{(l)} = ((\mathbf{W}^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(\mathbf{z}^{(l)})$.
4. **Gradients**: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T$.

### Recipe: Scaled Dot-Product Attention

1. **Inputs**: Queries $\mathbf{Q}$, Keys $\mathbf{K}$, Values $\mathbf{V}$.
2. **Formula**: $\text{Attn}(\mathbf{Q, K, V}) = \text{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$.
3. **Scaling**: $\sqrt{d_k}$ prevents gradients from vanishing by keeping softmax out of saturated regions.

### Recipe: CNN Output Size

1. **Dimensions**: For input $W$, kernel $K$, padding $P$, stride $S$:
2. **Output**: $W_{out} = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$.

---

## Week 8: Graph Neural Networks (GNNs)

### Recipe: GCN Layer Update

1. **Adjacency**: $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (Self-loops).
2. **Normalization**: $\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$, where $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.
3. **Propagation**: $\mathbf{H}^{(l+1)} = \sigma(\hat{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)})$.

---

## Week 10: Reinforcement Learning

### Recipe: Value Iteration

1. **Update**: $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$.
2. **Stopping**: Stop when $\|V_{k+1} - V_k\|_\infty < \epsilon$.
3. **Policy**: $\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$.

### Recipe: Q-Learning Update (Off-Policy)

1. **Target**: $TD_{target} = r + \gamma \max_{a'} Q(s', a')$.
2. **Update**: $Q(s,a) \leftarrow Q(s,a) + \eta [TD_{target} - Q(s,a)]$.

---

## Week 11: Causality & Kernels

### Recipe 1: D-Separation (Causal Graphs)

To check if $X \perp Y | Z$:

1. **Check all paths** between $X$ and $Y$. A path is blocked if:
2. **Chain/Fork**: $X \to Z \to Y$ or $X \leftarrow Z \to Y$ and $Z$ is observed.
3. **Collider**: $X \to C \leftarrow Y$. Path is blocked if **neither** $C$ nor any descendant of $C$ is observed.
4. **Result**: If all paths are blocked, $X \perp Y | Z$.

### Recipe 2: The Backdoor Criterion

To calculate $P(Y | do(X))$:

1. Find set $Z$ that blocks all paths from $X$ to $Y$ containing an arrow into $X$.
2. Ensure $Z$ contains no descendants of $X$.
3. **Adjust**: $P(Y | do(X)) = \sum_z P(Y | X, z) P(z)$.

---

## Week 12: Generative Models (VAE)

### Recipe: Evidence Lower Bound (ELBO)

1. **Decomposition**: $\log p(\mathbf{x}) = \text{ELBO} + \text{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$.
2. **Formula**: $\text{ELBO} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$.
3. **Interpretation**: [Reconstruction Term] - [Regularization Term].

---

## Week 13 & 14: Learning Theory

### Recipe: VC Dimension Calculation

1. **Lower Bound**: Find $d$ points that **can** be shattered (all $2^d$ labelings possible).
2. **Upper Bound**: Prove that **no** set of $d+1$ points can be shattered.
3. **Result**: $VC(\mathcal{H}) = d$.
4. **Example**: Linear classifier in $\mathbb{R}^2$: 3 points can be shattered $\to$ $VC = 3$.

### Recipe: Sample Complexity (PAC Learning)

1. **Finite $\mathcal{H}$ (Agnostic)**: $n \geq \frac{1}{2\epsilon^2} \ln \frac{2|\mathcal{H}|}{\delta}$.
2. **Infinite $\mathcal{H}$ (VC-bound)**: $n \geq O\left( \frac{d + \ln(1/\delta)}{\epsilon^2} \right)$.
