# SiCoVa: Self-Informed Cross-Correlation for Variance and Invariance in medical representation learning

This repository contains code and resources for a deep learning workflow involving pretraining, linear evaluation, and fine-tuning using SiCoVa with a ResNet-50 backbone. The pretraining is done on EyePacs dataset. The target datasets used are Messidor-2, APTOS-19 and Ocular Toxoplasmosis Fundus Images Dataset

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

