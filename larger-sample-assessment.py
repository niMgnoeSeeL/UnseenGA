import os
import argparse
import numpy as np
import sys
import pandas as pd

sys.path.append("/Better-Turing") # root directory of the project

import helpf1var
import matplotlib.pyplot as plt
import math
import multiprocessing as mp

formula_path_temp = "/Better-Turing/result/iclr/S{S}-n{n}-k0-{dist}-nGT-max20/evolutions/seed{seed}/evo2/best_ind.txt"
covmat_path_noseed_temp = "/Better-Turing/result/iclr/S{S}-n{n}-k0-{dist}-nGT-max20/Phi_matrix/covmat.pkl"
covmat_path_seed_temp = "/Better-Turing/result/iclr/S{S}-n{n}-k0-{dist}-nGT-max20/Phi_matrix/covmat-seed{seed}.pkl"
covmatsmp_path_temp = "/Better-Turing/result/iclr/S{S}-n{n}-k0-{dist}-nGT-max20/Phi_matrix/covmatsmp-seed{seed}.pkl"
result_path_temp = "/Better-Turing/result/iclr/S{S}-n{n}-k0-{dist}-nGT-max20/evolutions/seed{seed}/evo2/assess-incn.csv"


def get_objects(S, n, dist, seed):
    formula = helpf1var.load_formula(
        formula_path_temp.format(S=S, n=n, dist=dist, seed=seed)
    )
    if "diri" in dist:
        covmat_path = covmat_path_seed_temp.format(
            S=S, n=n, dist=dist, seed=seed
        )
    else:
        covmat_path = covmat_path_noseed_temp.format(S=S, n=n, dist=dist)
    covmat = helpf1var.CovMat.load_covmat(covmat_path)
    covmatsmp = helpf1var.CovMat.load_covmat(
        covmatsmp_path_temp.format(S=S, n=n, dist=dist, seed=seed)
    )
    return formula, covmat, covmatsmp


def change_n(formula, n_before, n_after):
    diff = n_after - n_before
    formula_new = []
    for row in formula:
        coef, n, k = row
        n, k = int(n), int(k)
        n_new = n + diff
        coef_new = coef * math.comb(n, k) / math.comb(n_new, k)
        formula_new.append([coef_new, n_new, k])
    return np.array(formula_new)


def compute_formula(matrix_dict, formula_mat):
    v = 0
    for coef, n, k in formula_mat:
        n, k = int(n), int(k)
        v += coef * matrix_dict[(n, k)]
    return v


def initial_check(S, n, dist, seed):
    # check if the assessment has already been done
    result_path = result_path_temp.format(S=S, n=n, dist=dist, seed=seed)
    if os.path.exists(result_path):
        print(f"assessment is already done: {result_path}")
        raise RuntimeError
    # check if the evolution is finished
    formula_path = formula_path_temp.format(S=S, n=n, dist=dist, seed=seed)
    if not os.path.exists(formula_path):
        print(f"evolution is not finished: {formula_path}")
        raise RuntimeError
    return


def assessment(S, n, dist, seed):
    global cs, num_rep
    formula_evo, covmat, covmatsmp = get_objects(S, n, dist, seed)

    c2formula_evo_m = {}
    needed_n_k = set()
    for c in cs:
        m = c * n
        needed_n_k.add((m, 1))
        formula_evo_m = helpf1var.mut_avail(change_n(formula_evo, n, m), m)
        c2formula_evo_m[c] = formula_evo_m
        for coef, n_formula, k_formula in formula_evo_m:
            needed_n_k.add((int(n_formula), int(k_formula)))
        needed_n_k_dict = {}
        for needed_n, needed_k in needed_n_k:
            if needed_n not in needed_n_k_dict:
                needed_n_k_dict[needed_n] = []
            needed_n_k_dict[needed_n].append(needed_k)

    data = []
    for _ in range(num_rep):
        sample_ext = np.concatenate(
            (
                covmatsmp.sample,
                np.random.choice(
                    range(1, S + 1),
                    size=(np.max(cs) - 1) * n,
                    p=covmat.p,
                    replace=True,
                ),
            )
        )
        phi_matrix_dict = {}
        for needed_n, needed_ks in needed_n_k_dict.items():
            sample_sub = sample_ext[:needed_n]
            freq, fof = np.unique(
                np.unique(sample_sub, return_counts=True)[1], return_counts=True
            )
            freq2fof = dict(zip(freq, fof))
            for needed_k in needed_ks:
                if needed_k not in freq2fof:
                    phi_matrix_dict[(needed_n, needed_k)] = 0
                else:
                    phi_matrix_dict[(needed_n, needed_k)] = freq2fof[needed_k]
        for c in cs:
            m = c * n
            sample_c = sample_ext[:m]
            unseens = np.array(list(set(range(1, S + 1)) - set(sample_c)))
            mm = np.sum(covmat.p[unseens - 1]) if len(unseens) > 0 else 0
            formula_gt = [[1 / m, m, 1]]
            mm_gt = compute_formula(phi_matrix_dict, formula_gt)
            formula_evo_m = c2formula_evo_m[c]
            mm_evo = compute_formula(phi_matrix_dict, formula_evo_m)
            data.append((c, mm, mm_gt, mm_evo))
    df = pd.DataFrame(data, columns=["c", "mm", "mm_gt", "mm_evo"])
    df["se_gt"] = (df["mm_gt"] - df["mm"]) ** 2
    df["se_evo"] = (df["mm_evo"] - df["mm"]) ** 2
    # mean squared error per c
    df_mse = df.groupby("c").mean()
    # rename se_gt -> mse_gt, se_evo -> mse_evo
    df_mse = df_mse.rename(columns={"se_gt": "mse_gt", "se_evo": "mse_evo"})
    df_mse["msre_gt"] = df_mse["mse_gt"] / df_mse["mm"]
    df_mse["msre_evo"] = df_mse["mse_evo"] / df_mse["mm"]
    df_mse["evo/gt"] = df_mse["msre_evo"] / df_mse["msre_gt"]
    # save
    result_path = result_path_temp.format(S=S, n=n, dist=dist, seed=seed)
    df_mse.to_csv(result_path, index=True)
    print(f"saved: {result_path}")


if __name__ == "__main__":
    # setting
    global cs, num_rep
    cs = [2, 5, 10]
    num_rep = 10000
    params = []
    for S in [100, 200]:
        for n in [S / 2, S, S * 2]:
            n = int(n)
            for dist in [
                "uniform",
                "zipf",
                "half",
                "zipfhalf",
                "diri",
                "dirihalf",
            ]:
                for seed in range(100):
                    print(f"S={S}, n={n}, dist={dist}, seed={seed}")
                    try:
                        initial_check(S, n, dist, seed)
                        # assessment(S, n, dist, seed)
                        params.append((S, n, dist, seed))
                    except RuntimeError:
                        pass
    with mp.Pool(100) as p:
        p.starmap(assessment, params)
    print("done")
