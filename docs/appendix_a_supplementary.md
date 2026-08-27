# Appendix A Supplementary Materials

This document mirrors the supplementary material submitted as Appendix A.

## 1. Preliminary Diagnostic Analysis Of Motion Attributes

Before finalizing the eight motion attributes, the diagnostic experiment examined whether the attributes are suitable for describing body-part motion patterns. The diagnostics are conducted directly on original skeleton coordinates and perturbed versions. They do not involve model training, parameter optimization, recognition loss, unseen-class samples, unseen-class labels, or unseen-class test results.

For each body part `p` at time `t`, the motion-attribute vector is:

```text
a^(p)(t) = [d_mu^(p)(t), d_sigma2^(p)(t), e^(p)(t), rho^(p)(t), theta^(p)(t), k^(p)(t), a_c^(p)(t), E_rel^(p)(t)]^T
```

The attributes cover local compactness, spatial extent, dominant orientation, motion intensity, and internal deformation.

Compared representations:

- Raw part coordinates: normalized joint coordinates of a body part over time.
- Eight motion attributes: the proposed attribute sequence computed from the same body part.

Perturbation protocol on seen-class validation samples:

- Temporal resampling: speed factors `{0.75, 0.85, 1.0, 1.15, 1.25}`, interpolated back to the original frame count.
- Motion-amplitude scaling: factors `{0.70, 0.85, 1.0, 1.15, 1.30}` relative to body center.
- Coordinate noise: Gaussian noise `sigma = 0.01` in bone-length-normalized coordinate space.
- Random frame dropping: 10% of frames removed and linearly interpolated back.
- Local joint perturbation: 30% of joints perturbed with Gaussian noise `sigma = 0.02`.

Diagnostic metrics:

Perturbation consistency:

```text
Cons = (1 / N) * sum_i cos(Z(X_i), Z(X_i_tilde))
```

Normalized perturbation distance:

```text
Dist = (1 / N) * sum_i ||Z(X_i) - Z(X_i_tilde)||_F / (||Z(X_i)||_F + epsilon)
```

Class-specific part centroid:

```text
mu_(c,p) = (1 / N_c) * sum_{i:y_i=c} Z_p(X_i)
```

Intra-class compactness, inter-class separation, and separation ratio:

```text
D_intra = (1 / (P|Y_s|)) * sum_p sum_{c in Y_s} (1 / N_c) * sum_{i:y_i=c} ||Z_p(X_i) - mu_(c,p)||_2
D_inter = (1 / P) * sum_p (1 / (|Y_s|(|Y_s|-1))) * sum_{c != c'} ||mu_(c,p) - mu_(c',p)||_2
Sep = D_inter / (D_intra + epsilon)
```

Diagnostic results:

Table 1. Stability comparison on seen-class validation samples, NTU RGB+D 60, 55/5 split.

| Representation | Temporal resampling Cons. | Amplitude scaling Cons. | Coord. noise Cons. | Frame drop Cons. | Joint perturb. Cons. | Avg. Cons. | Avg. Dist. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw part coordinates | 0.873 | 0.901 | 0.856 | 0.831 | 0.842 | 0.861 | 0.094 |
| Eight motion attributes | 0.924 | 0.937 | 0.912 | 0.874 | 0.899 | 0.909 | 0.053 |

Table 2. Separability comparison on seen-class validation samples, NTU RGB+D 60, 55/5 split.

| Representation | D_intra | D_inter | Sep |
|---|---:|---:|---:|
| Raw part coordinates | 0.412 | 0.651 | 1.58 |
| Eight motion attributes | 0.298 | 0.814 | 2.73 |

## 2. Body-Part-Level Stability Analysis

Table 3. Body-part-level perturbation consistency.

| Representation | Head | Torso | Left Arm | Right Arm | Left Leg | Right Leg |
|---|---:|---:|---:|---:|---:|---:|
| Raw part coordinates | 0.847 | 0.859 | 0.839 | 0.834 | 0.856 | 0.851 |
| Eight motion attributes | 0.903 | 0.911 | 0.919 | 0.916 | 0.908 | 0.905 |
| Improvement | +6.6% | +6.1% | +9.5% | +9.8% | +6.1% | +6.3% |

## 3. Recognition Ablation With Attribute Subsets

Table 4. Attribute group ablation on NTU RGB+D 60, 55/5 split.

| Attribute setting | Clean HM | Perturbed HM | HM Drop |
|---|---:|---:|---:|
| Static structural attributes only | 76.3 | 68.7 | 7.6 |
| Dynamic/deformation attributes only | 74.1 | 67.2 | 6.9 |
| Full eight attributes | 79.8 | 73.9 | 5.9 |

## 4. Body-Part Partition Settings

- 2-part partition: upper body `{head, torso, left arm, right arm}`, lower body `{left leg, right leg}`.
- 4-part partition: `{head, torso, arms (left arm, right arm), legs (left leg, right leg)}`.
- 6-part partition, default: `{head, torso, left arm, right arm, left leg, right leg}`.
- 8-part partition: `{head, torso, left upper arm, left lower arm (left forearm + hand), right upper arm, right lower arm (right forearm + hand), left leg, right leg}`.

The 6-part partition is used as the default because it provides the best balance between fine-grained motion decomposition and stable cross-part aggregation.

## 5. Hyperparameter Ablation

Table 5. Hyperparameter sensitivity on NTU RGB+D 60, 55/5 split. Values are clean test-set HM.

| lambda_i / lambda_c | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---:|---:|---:|---:|---:|
| 0.01 | 77.0 | 78.1 | 78.8 | 78.0 | 76.8 |
| 0.05 | 77.8 | 79.3 | 79.5 | 79.0 | 77.6 |
| 0.10 | 78.4 | 79.6 | 79.8 | 79.3 | 78.1 |
| 0.20 | 78.0 | 79.1 | 79.4 | 78.9 | 77.7 |
| 0.50 | 76.8 | 78.3 | 78.7 | 78.0 | 76.4 |

Default values are `lambda_c = 0.5` and `lambda_i = 0.1`.

## 6. LLM Prompt Template

The exact prompt template is released in `prompts/prompt_template.md`.

Representative examples from Appendix A Table 6 are released in `data/prompts/appendix_a_examples.json`.
