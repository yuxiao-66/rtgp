import os
import numpy as np
import torch
import random
import mat73
import sys


import strange_funcs
import model.SIRTGP_probit as sirtgp_probit
import model.SIRTGP_logit as sirtgp_logit
from model.utils import save_pickle


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fisher_z(x):
    x = np.clip(x, -0.999999, 0.999999)
    return 0.5 * np.log((1 + x) / (1 - x))

set_seed(0)

poly_degree = 60
a = 0.01
b = 100
first_burnin = 100
second_burnin = 0
mcmc_sample = 100
thin = 1

subject_id = 1

EEG = mat73.loadmat('data/real/s01.mat')
X_train, y_train = strange_funcs.processRun2(EEG['train'])
X_test, y_test = strange_funcs.processRun2(EEG['test'])

K, T = X_train.shape[-2:]
V0 = K * (K - 1) // 2

X_train = X_train.reshape([-1, K, T])
X_test = X_test.reshape([-1, K, T])
y_train = y_train.reshape(-1)

N_train = X_train.shape[0]
N_test = X_test.shape[0]

X0_train = np.zeros((N_train, V0))
X0_test = np.zeros((N_test, V0))

for i in range(N_train):
    idx = 0
    for u in range(K - 1):
        for v in range(u + 1, K):
            X0_train[i, idx] = np.corrcoef(X_train[i, u], X_train[i, v])[0, 1]
            idx += 1

for i in range(N_test):
    idx = 0
    for u in range(K - 1):
        for v in range(u + 1, K):
            X0_test[i, idx] = np.corrcoef(X_test[i, u], X_test[i, v])[0, 1]
            idx += 1

X0_train = fisher_z(X0_train)
X0_test = fisher_z(X0_test)

X_train = torch.from_numpy(X_train).float()
X0_train = torch.from_numpy(X0_train).float()
X_test = torch.from_numpy(X_test).float()
X0_test = torch.from_numpy(X0_test).float()
y_train = torch.from_numpy(y_train).float()

methods = [
    'SIRTGP_probit',
    'SRTGP_probit',
    'SIRTGP_logit',
    'SRTGP_logit'
]

os.makedirs('results/real', exist_ok=True)
os.makedirs('evals', exist_ok=True)

accu_file = f'evals/s{subject_id:02}_example_accu.txt'
with open(accu_file, "w") as f:
    f.write("")

for method in methods:

    print(f'Running {method}')

    if "probit" in method:
        ModelClass = sirtgp_probit.SIRTGP_probit
    else:
        ModelClass = sirtgp_logit.SIRTGP_logit

    model = ModelClass(
        y_train,
        X_train,
        X0_train,
        poly_degree=poly_degree,
        a=a,
        b=b,
        first_burnin=first_burnin,
        second_burnin=second_burnin,
        thin=thin,
        mcmc_sample=mcmc_sample
    )

    if "SIRTGP" in method:
        model.fit_si(mute=False)
    else:
        model.fit_s(mute=False)

    post_res = model.post_means()
    post_pred = model.predict_score(X_test, X0_test)
    ypred = post_pred.reshape(-1, 15, 2, 6)
    my_accu = strange_funcs.evaluate_chr_accu(ypred.numpy(), y_test)

    save_pickle(
        {
            'post_est': post_res,
            'loglik': model.loglik_y,
            'method': method,
            'accu': my_accu
        },
        f'results/real/s{subject_id:02}_{method}_example.pickle'
    )

    with open(accu_file, "a") as f:
        f.write(method + ": " + ', '.join(str(x.round(3)) for x in my_accu) + "\n")

print('\nFinished successfully')