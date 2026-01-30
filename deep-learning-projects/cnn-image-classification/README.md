# CNN Image Classification (University Project)

This project group focuses on **Convolutional Neural Networks (CNNs) for image-based prediction tasks**, developed as part of a university deep learning course.

The work is organized as a **progressive learning pipeline**:
1. Designing and evaluating CNNs **from scratch**
2. Extending the analysis using **pretrained backbones and fine-tuning**

The two subprojects are strongly related and should be read together.

## Project Overview

The goal of this project line is to understand how different CNN design choices impact performance on real-world image data, and how **transfer learning** compares to custom architectures trained from scratch.

Key themes:
- CNN architecture design
- Feature extraction vs end-to-end learning
- Regression vs classification from images
- Model evaluation and comparison
- Practical limitations of data augmentation and architectural complexity


## Subprojects

### 1. Age Estimator — CNNs from Scratch
[1_age-estimator/](1_age-estimator/)

An end-to-end CNN project where multiple architectures are **designed, trained, and evaluated from scratch** on facial images.

**Tasks**
- Age estimation (regression)
- Age group prediction (classification)

**Explored variations include**
- Network depth (3 vs 4 convolutional layers)
- Activation functions (ReLU, Leaky ReLU, ELU)
- Kernel sizes (3×3, 5×5, mixed)
- Skip connections
- Pooling vs no pooling
- RGB vs grayscale (1-channel)
- Data augmentation strategies

All models are evaluated on a held-out test set, and the **best-performing architectures are selected and compared against baselines**.

### 2. Backbone Models and Fine-Tuning
[2_backbone_model-and-fine-tuning/](2_backbone_model-and-fine-tuning/)

A continuation of the previous project, focusing on **transfer learning** using pretrained CNN backbones.

This subproject investigates:
- Using pretrained networks as feature extractors
- Frozen vs fine-tuned backbones
- Performance and convergence comparisons with custom CNNs
- Trade-offs between model complexity and generalization

The experiments reuse the same problem setting to allow **fair comparison with models trained from scratch**.

## ## Key Takeaways

- Architectural choices (depth, skip connections, activations) have a strong impact on regression performance.
- Classification tasks are more sensitive to dataset properties such as class imbalance.
- Data augmentation does not always improve performance, especially when domain constraints are violated.
- Transfer learning provides a strong alternative to training CNNs from scratch, especially with limited data.

