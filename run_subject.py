import os
import numpy as np
import torch
import random
import mat73
import sys

import argparse
import strange_funcs
import model.SIRTGP_probit as sirtgp_probit
import model.SIRTGP_logit as sirtgp_logit
from model.utils import save_pickle


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description="Run SIRTGP/SRTGP on EEG data")

    parser.add_argument("--subject_id", type=int, default=20,
                        help="Subject index (e.g., 1 for s01)")
    parser.add_argument("--poly", type=int, default=60,
                        help="Polynomial degree")
    parser.add_argument("--a", type=float, default=0.01,
                        help="Hyperparameter a")
    parser.add_argument("--b", type=float, default=100,
                        help="Hyperparameter b")
    parser.add_argument("--first_burnin", type=int, default=100)
    parser.add_argument("--second_burnin", type=int, default=0)
    parser.add_argument("--mcmc_sample", type=int, default=100)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        type=str,
        default="SIRTGP_probit",
        choices=[
            "SIRTGP_probit",
            "SRTGP_probit",
            "SIRTGP_logit",
            "SRTGP_logit",
        ],
        help="Model variant to run"
    )

    return parser.parse_args()

# Fisher Z transform (equivalent to R DescTools::FisherZ)
def fisher_z(x):
    x = np.clip(x, -0.999999, 0.999999)
    return 0.5 * np.log((1 + x) / (1 - x))


def main():

    args = get_args()
    set_seed(args.seed)
    print(args)

    # -------------------------------
    # Basic settings
    # -------------------------------
    subject_id = args.subject_id
    poly_degree = args.poly
    a, b = args.a, args.b

    os.makedirs("results/real", exist_ok=True)
    os.makedirs("evals", exist_ok=True)

    accu_file = f"evals/s{subject_id:02}_poly{poly_degree}_a{a}_b{b}_accu.txt"
    with open(accu_file, "w") as file:
        file.write("\n")

    # -------------------------------
    # Load data
    # -------------------------------
    file_path = f"data/real/s{subject_id:02d}.mat"
    EEG = mat73.loadmat(file_path)

    X_train, y_train = strange_funcs.processRun2(EEG["train"])
    X_test, y_test = strange_funcs.processRun2(EEG["test"])

    K, T = X_train.shape[-2:]
    V0 = K * (K - 1) // 2

    X_train = X_train.reshape([-1, K, T])
    X_test = X_test.reshape([-1, K, T])
    y_train = y_train.reshape(-1)

    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    # -------------------------------
    # Construct pairwise correlations
    # -------------------------------
    X0_train = np.zeros((N_train, V0))
    X0_test = np.zeros((N_test, V0))

    for i in range(N_train):
        idx = 0
        for u in range(K - 1):
            for v in range(u + 1, K):
                X0_train[i, idx] = np.corrcoef(
                    X_train[i, u], X_train[i, v]
                )[0, 1]
                idx += 1

    for i in range(N_test):
        idx = 0
        for u in range(K - 1):
            for v in range(u + 1, K):
                X0_test[i, idx] = np.corrcoef(
                    X_test[i, u], X_test[i, v]
                )[0, 1]
                idx += 1

    # Fisher Z transform
    X0_train = fisher_z(X0_train)
    X0_test = fisher_z(X0_test)

    # Convert to torch
    X_train = torch.from_numpy(X_train).float()
    X0_train = torch.from_numpy(X0_train).float()
    X_test = torch.from_numpy(X_test).float()
    X0_test = torch.from_numpy(X0_test).float()
    y_train = torch.from_numpy(y_train).float()

    # -------------------------------
    # Model variants
    # -------------------------------
    methods = [args.method]

    for method in methods:

        print(f"\nRunning {method}")

        if method == "SIRTGP_probit":
            model = sirtgp_probit.SIRTGP_probit(
                y_train, X_train, X0_train,
                poly_degree=poly_degree, a=a, b=b,
                first_burnin=args.first_burnin,
                second_burnin=args.second_burnin,
                thin=args.thin,
                mcmc_sample=args.mcmc_sample
            )
            model.fit_si(mute=False)

        elif method == "SRTGP_probit":
            model = sirtgp_probit.SIRTGP_probit(
                y_train, X_train, X0_train,
                poly_degree=poly_degree, a=a, b=b,
                first_burnin=args.first_burnin,
                second_burnin=args.second_burnin,
                thin=args.thin,
                mcmc_sample=args.mcmc_sample
            )
            model.fit_s(mute=False)

        elif method == "SIRTGP_logit":
            model = sirtgp_logit.SIRTGP_logit(
                y_train, X_train, X0_train,
                poly_degree=poly_degree, a=a, b=b,
                first_burnin=args.first_burnin,
                second_burnin=args.second_burnin,
                thin=args.thin,
                mcmc_sample=args.mcmc_sample
            )
            model.fit_si(mute=False)

        elif method == "SRTGP_logit":
            model = sirtgp_logit.SIRTGP_logit(
                y_train, X_train, X0_train,
                poly_degree=poly_degree, a=a, b=b,
                first_burnin=args.first_burnin,
                second_burnin=args.second_burnin,
                thin=args.thin,
                mcmc_sample=args.mcmc_sample
            )
            model.fit_s(mute=False)

        else:
            raise ValueError("Model name wrong")

        # ---------------------------
        # Prediction & evaluation
        # ---------------------------
        post_res = model.post_means()
        post_pred = model.predict_score(X_test, X0_test)

        ypred = post_pred.reshape(-1, 15, 2, 6)
        my_accu = strange_funcs.evaluate_chr_accu(
            ypred.numpy(), y_test
        )

        # Save results
        save_pickle(
            {
                "post_est": post_res,
                "loglik": model.loglik_y,
                "method": method,
                "accu": my_accu,
            },
            f"results/real/s{subject_id:02}_{method}_poly{poly_degree}_a{a}_b{b}_est.pickle",
        )

        with open(accu_file, "a") as file:
            file.write(
                method + ": " +
                ", ".join(str(x.round(3)) for x in my_accu) + "\n"
            )

    print("\nFinished successfully.")


if __name__ == "__main__":
    main()
