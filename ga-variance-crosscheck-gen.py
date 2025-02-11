import helpf1var
import numpy as np
import os
import pandas as pd
import itertools
import multiprocessing as mp
import sys
import pickle

p_est_method = "nGT"
maxterm = 20

dir_path_temp = os.path.join(
    "/Better-Turing/result/iclr/",  # set the result folder
    "S{S}-n{n}-k0-{dist}-" + f"{p_est_method}-max{maxterm}",
    "evolutions",
)
data_path_seeded_temp = os.path.join(dir_path_temp, "seed{seed}", "evo{evo}")
best_formula_path_temp = os.path.join("{data_path}", "best_ind.txt")
covmat_path_temp = os.path.join(dir_path_temp, "../Phi_matrix/covmat.pkl")


def get_covmat(S, n, dist):
    if "diri" in dist:
        raise NotImplementedError
    else:
        covmat_path = covmat_path_temp.format(S=S, n=n, dist=dist)
    if os.path.exists(covmat_path):
        print(f"Loading covmat from {covmat_path}", flush=True)
        covmat = helpf1var.CovMat.load_covmat(covmat_path)
    else:
        raise ValueError(f"{covmat_path} does not exist")
    return covmat


def get_mse_diff_dist(param):
    formula, covmat, gt, seed, evo, dist1, dist2 = param
    exp_mass = covmat.exp_mass / covmat.n
    var_mass = covmat.var_mass / covmat.n**2
    F_matrix = covmat.F_matrix
    exp_formula = helpf1var.compute_formula(F_matrix, formula)
    var_formula = helpf1var.compute_formula_var(formula, covmat)
    cov_formula = (
        helpf1var.compute_formula_cov_mass_evo(formula, covmat) / covmat.n
    )
    mse = (
        (exp_formula - exp_mass) ** 2 + var_formula + var_mass - 2 * cov_formula
    )
    return f"{dist1}-{dist2}", mse


if __name__ == "__main__":
    # read S and n
    S, n = sys.argv[1], sys.argv[2]
    S, n = int(S), int(n)
    dists = ["uniform", "half", "zipfhalf", "zipf"]
    num_formula = 100

    formula_dict = {dist: [] for dist in dists}
    for dist in formula_dict:
        # Ses_cands = [(v // 10, v % 10) for v in range(100)]
        Ses_cands = list(range(100))
        while len(formula_dict[dist]) < num_formula and len(Ses_cands) > 0:
            Se = Ses_cands.pop(np.random.randint(len(Ses_cands)))
            seed, evo = Se, 2
            data_path = data_path_seeded_temp.format(
                S=S,
                n=n,
                dist=dist,
                seed=seed,
                evo=evo,
            )
            best_formula_path = best_formula_path_temp.format(
                data_path=data_path
            )
            if os.path.exists(best_formula_path):
                formula = np.fromstring(
                    open(best_formula_path)
                    .read()
                    .replace("[", "")
                    .replace("]", ""),
                    sep=" ",
                ).reshape((-1, 3))
                formula_dict[dist].append((formula, seed, evo))

    mse_dict = {
        f"{dist1}-{dist2}": []
        for dist1, dist2 in itertools.product(dists, dists)
    }

    covmat_dict = {dist: get_covmat(S, n, dist) for dist in dists}

    for dist1, dist2 in itertools.product(dists, dists):
        print(f"dist1: {dist1}, dist2: {dist2}", flush=True)
        covmat = covmat_dict[dist2]
        v_gt = covmat.F_matrix[n + 1, 1] / (n + 1)
        params = []
        for formula, seed, evo in formula_dict[dist1]:
            feasible_formula = helpf1var.mut_avail(formula, n)
            params.append(
                (
                    feasible_formula,
                    covmat,
                    v_gt,
                    seed,
                    evo,
                    dist1,
                    dist2,
                )
            )
        with mp.Pool(processes=100) as pool:
            results = pool.map(get_mse_diff_dist, params)
        for key, value in results:
            mse_dict[key].append(value)

    # save mse_dict to pickle
    mse_dict_path = os.path.join(
        "/Better-Turing/result/iclr/",  # set the result folder
        f"mse_dict_S{S}_n{n}.pkl",
    )
    with open(mse_dict_path, "wb") as f:
        pickle.dump(mse_dict, f)
