# SiCoVa: Self-Supervised Learning with Cross-Correlation Regularization

This repository implements **SiCoVa** (Self-Supervised Learning with Variance-Invariance-Covariance and Cross-Correlation Regularization), extending the existing VICReg framework with an additional **cross-correlation regularization** term. SiCoVa supports self-supervised pretraining, linear evaluation, and fine-tuning on downstream classification tasks. The implementation uses **PyTorch** and **Torchvision**, featuring a ResNet50 backbone, MLP projector, and linear classifier.

## Features

- **SiCoVa Implementation**:
  - ResNet50 encoder with an MLP expander for self-supervised learning.
  - Additional **cross-correlation regularization** term for improved representation learning.
- **Linear Evaluation**:
  - Freezes the pretrained encoder and trains a linear classification layer.
- **Fine-Tuning**:
  - Option to unfreeze parts of the encoder for end-to-end optimization.
- **Training and Validation**:
  - Configurable training loop with support for checkpointing and detailed metrics.
- **Top-k Accuracy**:
  - Evaluate classification performance using top-1 or top-k metrics.
![](image.png)
---

## File Structure

### 1. `pretraining/`
This directory is dedicated to the pretraining stage using the SiCoVa method.

- **`pretraining.py`**: The main script for the pretraining process, implementing SiCoVa with ResNet-50.

---

### 2. `linear_evaluation/`
Contains scripts and resources for evaluating the pretrained model via linear evaluation.

- **`linear_eval.py`**: The script for performing linear evaluation on the pretrained model to assess its learned representations.

---

### 3. `fine_tuning/`
Includes resources for fine-tuning the pretrained model on a downstream task.

- **`fine_tune.py`**: The script for fine-tuning the pretrained model on a specific dataset.

---

## How to Use

### Pretraining
1. Navigate to the `pretraining/` directory.
2. Run the `pretraining.py` script:
   ```bash
   python pretraining.py
   ```
3. Checkpoints will be saved in the same directory for further use.

### Linear Evaluation

1. Navigate to the `linear_evaluation/` directory.
2. Run the `linear_eval.py` script:
   ```bash
   python linear_eval.py
   ```

### Fine-Tuning

1. Navigate to the `fine_tuning/` directory.
2. Run the `fine_tune.py` script:
   ```bash
   python fine_tune.py
   ```

## Prerequisites
- Python >= 3.8
- PyTorch >= 1.10
- Install additional dependencies:
 ```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy training! 🚀