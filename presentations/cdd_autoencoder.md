# Concept Drift Detection Based on DNNs and Autoencoders
Lisha Hu, Yaru Lu, Yuehua Feng
Eirini Tzanaki, December 2025

---

# Methodology Overview

- Semi-supervised Concept Drift Detection in data streams
- 3-stage framework in a stable latent space:
  - Stage 1: Learn latent features of the old concept
  - Stage 2: Map new samples to the same latent space
  - Stage 3: Autoencoder and reconstruction error to detect drift

---

# Stage 1: Pre-training Model (Model 1)
## Goal: Learn latent representation of the old concept

- **Input:** drift-free historical dataset $X_{\text{old}} = \{x_i\}$  
- Train Model 1 on $X_{\text{old}}$   
- For each sample $x_i \in X_{\text{old}}$:
  - compute latent vector $a^{(L)}_i = \text{Model1}(x_i)$
- Collect latent feature set:
  - $A = \{ a^{(L)}_i \}$ 
- **Output:** Latent feature matrix $A$ describing the *old* concept

---

# Stage 2 – Streaming Latent Features (Model 2)
## Goal: Represent new stream samples in the same latent space

- Initialize Model 2 with weights from Model 1
- For each incoming stream sample $x_t$:
  - compute latent vector $o^{(L)}_t = \text{Model2}(x_t)$
- Build sequence of stream latent features:
  - $O = \{ o^{(L)}_t \}$

---

# Stage 3 – Autoencoder Training (Model 3)
## Goal: Learn structure of the old concept in latent space

- Input: latent feature set $A$
- Train an autoencoder $\text{AE}(\cdot)$ on $A$:
  - for each $a_i \in A$:
    - $\hat{a}_i = \text{AE}(a_i)$
    - reconstruction error $e_i = \|a_i - \hat{a}_i\|$
- Compute $\mu$, $\sigma$ of reconstruction error:
  - $\mu = \text{mean}(e_i)$  
  - $\sigma = \text{std}(e_i)$
- Define drift threshold (3$\sigma$ rule):

$$
T = \mu + 3\sigma
$$

