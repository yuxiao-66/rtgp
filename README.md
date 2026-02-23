`# SIRTGP: Sparse Interaction Regularized Time-varying Gaussian Process
`
## Overview

This repository implements the **Sparse Interaction Regularized
Time-varying Gaussian Process (SIRTGP)** model for EEG-based
Brain--Computer Interface (BCI) classification.

The package includes: - Probit and logit model variants - Interaction
and non-interaction versions - Example script for reproducing results on
subject S01

The example script is designed for demonstration and uses reduced MCMC
iterations for faster execution.

------------------------------------------------------------------------

# Quick Start

## 1. Install Dependencies

Python 3.9+ is recommended.

``` bash
pip install -r requirements.txt
```

## 2. Prepare Data

The example expects EEG data in:

    data/real/s01.mat

The `.mat` file must be MATLAB v7.3 format and contain:

-   `train`
-   `test`

If your file is not v7.3, convert it or replace `mat73.loadmat` with
`scipy.io.loadmat`.

## 3. Run Example

To run the model using the default configuration 
(subject 1 and default hyperparameters), use:
``` bash
python run_subject.py
```
Default settings:

- subject_id = 1  
- poly = 60  
- a = 0.01  
- b = 100  
- first_burnin = 100  
- second_burnin = 0  
- mcmc_sample = 100  
- thin = 1  
- seed = 0 

You can override the default settings using command-line arguments.

For example:
```bash
python run_subject.py --subject_id 2 --poly 80 --mcmc_sample 200 --seed 1
```

Outputs will be saved to:

    results/real/
    evals/

------------------------------------------------------------------------

# What run_s01.py Does

The script performs the following steps:

1.  Load subject S01 data.
2.  Preprocess using `processRun2()` from `strange_funcs.py`.
3.  Construct interaction features:
    -   Computes pairwise channel correlations
    -   Applies Fisher z-transform
4.  Fit four model variants:
    -   SIRTGP_probit (with interaction)
    -   SRTGP_probit (without interaction)
    -   SIRTGP_logit (with interaction)
    -   SRTGP_logit (without interaction)
5.  Predict on test data.
6.  Evaluate character-level accuracy.
7.  Save results as pickle files and accuracy text files.

------------------------------------------------------------------------

# Data Format

After preprocessing, the following shapes are expected:

-   `X_train`: (N_train, K, T)
-   `y_train`: (N_train,)
-   `X_test`: (N_test, K, T)

Interaction features:

-   `X0_train`: (N_train, K(K−1)/2)
-   `X0_test`: (N_test, K(K−1)/2)

------------------------------------------------------------------------

# Model Variants

-   **SIRTGP**: Includes signal interaction terms (`fit_si()`)
-   **SRTGP**: No interaction (`fit_s()`)
-   Probit and logit likelihood versions are available.

------------------------------------------------------------------------

# Saved Results

Each method saves a pickle file containing:

    {
        'post_est': posterior samples and estimates,
        'loglik': log-likelihood trace,
        'method': model name,
        'accu': character-level accuracy
    }

------------------------------------------------------------------------

# Using Your Own Data

To apply SIRTGP to new EEG data:

1.  Format EEG trials as `(N, K, T)`.
2.  Provide binary labels `(N,)`.
3.  Compute interaction features (or modify model to disable
    interactions).
4.  Initialize and fit:

``` python
model = SIRTGP_probit(y, X, X0, ...)
model.fit_si()   # with interaction
# or
model.fit_s()    # without interaction
```

------------------------------------------------------------------------

# Notes

-   The example uses small MCMC settings for speed.
-   Full paper results require larger burn-in and sampling iterations.
-   Interaction feature computation is O(NK²T); consider caching for
    large datasets.

------------------------------------------------------------------------

# Troubleshooting

If you encounter errors:

-   Verify `.mat` file format (must be v7.3 for mat73).
-   Ensure `strange_funcs.py` is present in the repository.
-   Check that all dependencies are installed.

------------------------------------------------------------------------

For questions or issues, please open a GitHub issue.
