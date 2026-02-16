# SIRTGP: Sparse Interaction Regularized Time-varying Gaussian Process

## Overview

This repository contains the implementation of the Sparse Interaction Regularized Time-varying Gaussian Process (SIRTGP) model for EEG-based BCI classification.

The example script demonstrates how to train and evaluate the model on one subject (S01).

---

## Workflow

### Run Calculations

Execute:

```bash
python run_s01.py
```

The results will be saved in the `results` and `evals` folders.

---

## Repository Structure

### Code

All core model implementations are located in the `model` folder.

- `SIRTGP_probit.py` – Probit version of the model  
- `SIRTGP_logit.py` – Logit version of the model  
- `helper.py` / `utils.py` – Supporting functions  

### Example Script

- `run_s01.py` – Example script for running SIRTGP on subject S01  

### Data

The repository expects EEG data in:

```bash
data/real/raw_data/s01.mat
```

For the logit version, precomputed basis files are required in the `data` folder.

---

## Python Environment

The code was tested using Python 3.9+.

Install dependencies with:

```bash
pip install -r requirements.txt
```

