# Deep Learning Vocabulary Reference
### SIAM Life Sciences 2026 — Deep Learning Section

---

## Guiding Principles

By the end of this section, you should be able to:

- **Speak the language.** Become conversant in the core vocabulary of deep learning — layers, loss functions, activation functions, gradient descent, overfitting — well enough to read papers and follow talks in the field.

- **Understand the interpretability spectrum.** Recognize that "black box" is not an inevitable property of deep learning, and be able to distinguish three strategies for recovering interpretability: inspecting the architecture, explaining predictions post-hoc, and embedding biological structure directly into training.

- **See where deep learning meets mechanistic modeling.** Understand how known biological structure (an ODE, a conservation law, a rate equation) can be encoded as a constraint on a neural network, and what this buys you over classical parameter fitting.

---

## Core Building Blocks

**Neuron**
The basic computational unit. A neuron takes a weighted sum of its inputs, adds a bias, and passes the result through an activation function. Inspired loosely by biological neurons, but the analogy is shallow — think of it as a parameterized function rather than a cell.

**Weights and Biases**
The *learnable parameters* of the network. Each connection between neurons has a weight (how much to scale that input). Each neuron also has a bias (a constant offset). Training = finding the values of weights and biases that minimize the loss.

**Activation Function**
A nonlinear function applied to a neuron's output. Without activations, stacking layers would collapse to a single linear transformation. Common choices:
- **ReLU** (Rectified Linear Unit): f(x) = max(0, x) — simple, fast, widely used in hidden layers
- **Sigmoid**: f(x) = 1/(1 + e^−x) — squashes output to (0,1), used for binary classification outputs
- **Softmax**: generalizes sigmoid to multiple classes

---

## Neurons in Depth

**The full picture: weights, bias, and activation together**

For a neuron with inputs x₁, …, xₙ, the computation is:

```
z = w₁x₁ + w₂x₂ + … + wₙxₙ + b     ← linear combination (the "pre-activation")
a = f(z)                              ← activation function applied to z
```

In vector form: **z = wᵀx + b**, then **a = f(z)**.

The weights and bias define a *hyperplane* in input space. The activation function then warps that into a nonlinear decision surface. Without f (or with f = identity), stacking layers collapses to a single affine map — you gain nothing from depth.

**Single neuron, one incoming edge**

```
x ──[w]──→ ( z = wx + b ) ──f──→ a
```

With sigmoid activation: a = 1/(1 + e^{−(wx+b)}) — this is exactly **logistic regression**.
With linear activation: a = wx + b — this is **linear regression**.

A single neuron *is* a generalized linear model. Neural networks extend this by composing many such units.

---

## Training Concepts

**Data Splits: Train / Validation / Test**

Traditional ML commonly uses an 80/20 train/test split. Deep learning typically uses a three-way split:

| Split | Typical fraction | Role |
|-------|-----------------|------|
| Train | 60% | Model sees this; weights are updated on it |
| Validation | 20% | Evaluated during training to monitor overfitting and tune hyperparameters |
| Test | 20% | Held out completely; touched only once at the very end |

The critical distinction: **validation influences decisions made during training** (e.g., when to stop, which learning rate to use, how many layers to try). If you make any decision based on a data split, that split is functionally a validation set, not a test set. Using the test set to make decisions contaminates it.

For small datasets — common in biology — a fixed split may be too noisy. **k-fold cross-validation** is preferred: partition the data into k folds, train k times each with a different fold held out, and average performance.

**Loss Function**
A measure of how wrong the network's predictions are. Training minimizes this. For binary classification (like our examples) we use *binary cross-entropy*:

```
L = − [ y log(ŷ) + (1−y) log(1−ŷ) ]
```

**Stochastic Gradient Descent (SGD) — for mathematicians**

All variants share the same skeleton: move parameters in the direction of steepest descent of the loss.

*Full (batch) gradient descent* — compute the gradient over all N training samples per step:

```
θ_{t+1} = θ_t − η · ∇_θ L(θ_t)
         where  L(θ) = (1/N) Σᵢ ℓ(f(xᵢ; θ), yᵢ)
```

*SGD (true stochastic)* — approximate the gradient with one randomly drawn sample:

```
θ_{t+1} = θ_t − η · ∇_θ ℓ(f(xᵢ; θ_t), yᵢ)
```

*Mini-batch SGD* (what people usually mean by "SGD" in practice) — average over a batch B of size b ≪ N:

```
θ_{t+1} = θ_t − η · (1/b) Σⱼ∈Bₜ ∇_θ ℓ(f(xⱼ; θ_t), yⱼ)
```

All three are **first-order methods** — they use only ∇L, not the Hessian ∇²L.

Second-order methods (Newton, L-BFGS) use the Hessian: θ_{t+1} = θ_t − η [∇²L]⁻¹ ∇L. They converge faster in iterations but computing and inverting the Hessian for a network with millions of parameters is completely intractable. Hence first-order methods dominate deep learning.

*Adam* (used in the notebook) is a first-order adaptive method. It maintains exponential moving averages of the gradient (mₜ) and its element-wise square (vₜ), and normalizes each parameter's update by its historical gradient variance:

```
mₜ = β₁ mₜ₋₁ + (1−β₁) ∇L
vₜ = β₂ vₜ₋₁ + (1−β₂) (∇L)²
θ_{t+1} = θ_t − η · m̂ₜ / (√v̂ₜ + ε)
```

This effectively gives each parameter its own adaptive learning rate.

**Learning Rate**
Controls how large each weight update step is. Too large → training is unstable. Too small → training is very slow or gets stuck. Typical values: 0.001–0.01.

**Epoch**
One complete pass through the entire training dataset. We typically train for multiple epochs (10–100+). Each epoch gives the network another chance to adjust its weights.

**Batch Size**
How many training examples are processed before weights are updated. A batch size of 64 means: compute the loss on 64 samples, compute gradients, update weights — then repeat.

---

## Worked Example: A 2-Neuron Network

The simplest non-trivial network: one scalar input, two neurons in sequence, one scalar output.

```
x  →  Neuron 1 (ReLU)  →  h  →  Neuron 2 (sigmoid)  →  ŷ
```

**Parameters**: w₁, b₁ (Neuron 1), w₂, b₂ (Neuron 2) — 4 learnable scalars.

**Forward pass:**

```
z₁ = w₁x + b₁
h  = ReLU(z₁) = max(0, z₁)
z₂ = w₂h + b₂
ŷ  = σ(z₂)  =  1 / (1 + e^{−z₂})
```

**Loss** (binary cross-entropy, single sample):
```
L = − [ y log(ŷ) + (1−y) log(1−ŷ) ]
```

**Backward pass — the chain rule applied:**

```
∂L/∂w₂ = (ŷ − y) · h                      ← error signal × input to neuron 2
∂L/∂b₂ = (ŷ − y)

∂L/∂w₁ = (ŷ − y) · w₂ · ReLU'(z₁) · x   ← error propagated back through neuron 2
∂L/∂b₁ = (ŷ − y) · w₂ · ReLU'(z₁)
```

where ReLU'(z₁) = 1 if z₁ > 0, else 0.

The factor (ŷ − y) appears in every gradient — it is the error signal at the output, propagated backwards. This is **backpropagation**: repeated application of the chain rule, flowing error information from output to input.

**SGD update** (one step, learning rate η):
```
w₁ ← w₁ − η · ∂L/∂w₁      w₂ ← w₂ − η · ∂L/∂w₂
b₁ ← b₁ − η · ∂L/∂b₁      b₂ ← b₂ − η · ∂L/∂b₂
```

This repeats for each batch, for each epoch. Training a deep network with millions of parameters is exactly this — just with more layers and the chain rule applied more times.

---

## Network Architectures

**MLP — Multi-Layer Perceptron**
The simplest deep network: fully-connected layers stacked together. Every neuron in one layer connects to every neuron in the next. Input images must be flattened into a vector first, which discards spatial structure.

**CNN — Convolutional Neural Network**
Designed for structured data like images. Instead of connecting every pixel to every neuron, *convolutional filters* slide over the image and detect local patterns (edges, curves, textures). Key ideas:
- **Convolutional layer**: applies a small learnable filter across the image
- **Pooling layer**: reduces spatial size (e.g., 2×2 max-pooling), builds in some translation invariance
- **Spatial hierarchy**: early layers detect edges; later layers detect higher-level features

CNNs preserve spatial relationships that MLPs throw away, which is why they typically outperform MLPs on image tasks.

---

## Interpretability

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
A technique for visualizing *where* a CNN is "looking" when it makes a prediction. It computes the gradient of the predicted score with respect to the last convolutional layer's feature maps, then creates a heatmap showing which spatial regions influenced the decision most.

Useful for:
- Debugging: is the model focusing on the right region?
- Trust: does the model's reasoning align with domain knowledge?
- Communication: explaining predictions to non-experts

**Smoothed Grad-CAM**
A Gaussian blur applied to the Grad-CAM heatmap. Reduces noise and makes the highlighted regions easier to interpret visually — especially helpful for low-resolution images like the 28×28 datasets we use.

---

## Quick-Reference Table

| Term | One-line definition |
|------|---------------------|
| Neuron | Computes a = f(wᵀx + b); a parameterized nonlinear function |
| Weights & biases | The learnable parameters; define the linear part of each neuron |
| Activation function | Nonlinearity f applied after the linear combination (ReLU, sigmoid) |
| Pre-activation | z = wᵀx + b; the linear combination before f is applied |
| Train set | Data the model learns from (weights updated on this) |
| Validation set | Data used during training to monitor overfitting / tune hyperparameters |
| Test set | Held-out data; evaluated only once at the very end |
| Loss function | Scalar measure of prediction error; training minimizes this |
| SGD | First-order optimization: θ ← θ − η ∇L, gradient estimated on a mini-batch |
| Learning rate η | Step size for each weight update |
| Epoch | One full pass through the training data |
| Batch size | Number of samples per gradient estimate |
| Backpropagation | Chain rule applied layer-by-layer to compute ∂L/∂θ |
| MLP | Fully-connected network; flattens input |
| CNN | Spatial network; uses convolutional filters |
| Grad-CAM | Heatmap showing which image regions most influenced the CNN's prediction |
