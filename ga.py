import numpy as np
import pandas as pd
from typing import List, Tuple
from deap import base, tools
import time
import math
import argparse
import os
import logging
import itertools

import helpf1var

factorial = math.factorial
result_folder = "/Better-Turing/result/"  # set the result folder
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# EA parameters
NGEN = 100
MAXPOP = 40
RMUT = 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("S", type=int, help="number of species")
    parser.add_argument(
        "disttype",
        type=str,
        help="distribution type",
        choices=["uniform", "zipf", "half", "zipfhalf", "diri", "dirihalf"],
    )
    parser.add_argument("n_total", type=int, help="number of samples")
    parser.add_argument("m_target", type=int, help="target m")
    parser.add_argument("k_target", type=int, help="target k", default=0)
    parser.add_argument(
        "setting",
        type=str,
        help="setting",
        choices=["knowing", "sampling", "onlycovmat"],
        default="knowing",
    )
    parser.add_argument(
        "--p_est_method",
        type=str,
        help="estimation method for p for evolution",
        choices=["none", "emp", "nGT"],
        default="emp",
    )
    parser.add_argument(
        "--max_term",
        type=int,
        help="maximum number of terms in the formula",
        default=-1,
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="maximum number of iterations",
        default=2000,
    )
    parser.add_argument("--seed", type=int, help="random seed", default=-1)
    parser.add_argument(
        "--rep_smp", type=int, help="sampling repetitions", default=1
    )
    parser.add_argument(
        "--rep_evo", type=int, help="evoluation repititions", default=1
    )
    args = parser.parse_args()
    if args.setting == "knowing":
        args.p_est_method = "none"

    global result_folder
    result_folder = os.path.join(
        result_folder,
        f"S{args.S}-n{args.n_total}-m{args.m_target}-k{args.k_target}-{args.disttype}-{args.p_est_method}",
    )
    if args.max_term != -1:
        result_folder += f"-max{args.max_term}"
    if args.seed == -1:
        logging.warning("seed is not set. seed is set to 0.")
        args.seed = 0

    return args


def get_p(S: int, disttype: str) -> np.ndarray:
    logging.info(f"get_p: S={S}, disttype={disttype}")
    if disttype == "uniform":
        p = np.ones(S) / S
    elif disttype == "zipf":
        p = 1 / np.array(range(1, S + 1))
        p = p / sum(p)
    elif disttype == "half":
        half = int(S / 2)
        p = np.array([3] * half + [1] * (S - half))
        p = p / sum(p)
    elif disttype == "zipfhalf":
        p = 1 / np.array(range(1, S + 1))
        p = p ** (1 / 2)
        p = p / sum(p)
    elif disttype == "diri":
        p = np.sort(np.random.dirichlet(np.ones(S)))[::-1]
        p = p / np.sum(p)
    elif disttype == "dirihalf":
        p = np.sort(np.random.dirichlet(np.ones(S) * 0.5))[::-1]
        p = p / np.sum(p)
    else:
        raise ValueError("disttype not recognized")
    return p


def get_cov_matrix(
    S: int, disttype: str, n: int, m: int, k: int, postfix: str = ""
) -> Tuple[np.ndarray, np.ndarray, helpf1var.CovMat]:
    # 2. 'postfix' is used for the disttype 'diri' and 'dirihalf' where the
    #    covmat is generated multiple times.
    logging.info(
        f"get_cov_matrix: S={S}, disttype={disttype}, n={n}, m={m}, k={k}, postfix={postfix}"
    )
    global result_folder
    cov_matrix_path = os.path.join(
        result_folder, "Phi_matrix", f"covmat{postfix}.pkl"
    )
    if os.path.exists(cov_matrix_path):
        logging.info(f"CovMat exists. Load CovMat from {cov_matrix_path}.")
        cov_matrix = helpf1var.CovMat.load_covmat(cov_matrix_path)
        p = cov_matrix.p
        F_matrix = cov_matrix.F_matrix
    else:
        logging.info(f"Generate CovMat.")
        os.makedirs(os.path.dirname(cov_matrix_path), exist_ok=True)
        p = get_p(S, disttype)
        F_matrix = helpf1var.gen_F(p)
        cov_matrix = helpf1var.CovMat(p, n, m, k, F_matrix)
    return p, F_matrix, cov_matrix


def get_cov_matrix_emp(
    p: np.ndarray, n: int, m: int, k: int, seed: int, p_est_method: str
) -> helpf1var.CovMat:
    logging.info(f"get_cov_matrix_emp: n={n}, seed={seed}")
    global result_folder
    Phi_matrix_path = os.path.join(
        result_folder, "Phi_matrix", f"seed{seed}.csv"
    )
    cov_matrix_emp_path = os.path.join(
        result_folder, "Phi_matrix", f"covmatsmp-seed{seed}.pkl"
    )
    if os.path.exists(cov_matrix_emp_path):
        logging.info(
            f"Phi_matrix exists. Load CovMat from {cov_matrix_emp_path}."
        )
        cov_matrix_emp = helpf1var.CovMat.load_covmat(cov_matrix_emp_path)
    else:
        logging.info(f"Generate Phi_matrix and save to {Phi_matrix_path}.")
        os.makedirs(os.path.dirname(Phi_matrix_path), exist_ok=True)
        Phi_matrix, sample = helpf1var.gen_Phi_fromsample(p, n, seed)
        pd.DataFrame(Phi_matrix).to_csv(Phi_matrix_path)
        logging.info(f"Generate CovMat from Phi_matrix.")
        p_esti = get_p_esti(Phi_matrix, n, p_est_method)
        F_matrix_emp = helpf1var.gen_F(p_esti)
        cov_matrix_emp = helpf1var.CovMat(p_esti, n, m, k, F_matrix_emp, sample)
    return cov_matrix_emp


def get_p_esti(Phi_matrix: np.ndarray, n: int, p_est_method: str) -> np.ndarray:
    # estimate probability with sample vector with n samples in Phi_matrix,
    # i.e., the (n + 1)-th row of Phi_matrix = Phi_matrix[n, :]
    if p_est_method == "emp":
        return get_p_emp(Phi_matrix, n)
    elif p_est_method == "nGT":
        return get_p_natural_GT(Phi_matrix, n)


def get_p_emp(Phi_matrix: np.ndarray, n: int) -> np.ndarray:
    logging.info(f"get_p_emp: n={n}")
    p_emp = []
    for freq, ff in enumerate(Phi_matrix[n, :]):
        if freq != 0:
            p_emp += [freq] * ff
    return np.array(sorted(p_emp, reverse=True)) / np.sum(p_emp)


def is_empirical(vec_phi, k):
    return k > vec_phi[k + 1]


def get_p_natural_GT(Phi_matrix, n) -> np.ndarray:
    logging.info(f"get_p_natural_GT: n={n}")
    vec_phi = Phi_matrix[n, :]
    f0 = np.ceil(
        helpf1var.num_undetected_lower_bound(n, vec_phi[1], vec_phi[2])
    )
    if f0 == 0:
        return get_p_emp(Phi_matrix, k)
    p = []
    for k in range(0, n + 1):
        if k == 0 or vec_phi[k] > 0:
            if is_empirical(vec_phi, k):
                p_esti = k / n
                p += [p_esti] * vec_phi[k]
            else:
                num_s = int(vec_phi[k] if k > 0 else f0)
                p_esti = (vec_phi[k + 1] + 1) / num_s * (k + 1) / n
                p += [p_esti] * num_s
    return np.array(sorted(p, reverse=True)) / sum(p)


def get_toolbox(
    formulas: List[np.ndarray],
    m_target: int,
    k_target: int,
    cov_matrix_emp: np.ndarray,
) -> base.Toolbox:
    logging.info("get_toolbox")
    toolbox = base.Toolbox()
    toolbox.register(
        "individual_guess",
        helpf1var.initIndividual,
        helpf1var.creator.Individual,
    )
    toolbox.register(
        "population_guess",
        helpf1var.initPopulation,
        list,
        toolbox.individual_guess,
        formulas,
    )
    toolbox.register("mutate", helpf1var.mut, n_available=m_target)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register(
        "evaluate",
        helpf1var.evaluate,
        cov_matrix=cov_matrix_emp,
        m_target=m_target,
        k_target=k_target,
    )
    return toolbox


def run_evolution(
    toolbox,
    m_target,
    cov_matrix_evo,
    evo_output_dir,
    evo_idx,
    max_term,
    max_iter,
):
    global NGEN, MAXPOP, RMUT
    ngen = NGEN
    output_dir = os.path.join(evo_output_dir, f"evo{evo_idx}")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    record_path = os.path.join(output_dir, "record.csv")
    # initialize
    pop = toolbox.population_guess()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    initial_pop = list(map(toolbox.clone, pop))
    prev_fitnesses = {ind.fitness.values for ind in pop}
    min_fitness = min([fit[0] for fit in fitnesses])
    min_initial_fitness = min_fitness

    # record: (NGEN, generation, num_populations, initial_fitness, prevgen_best_fitness, best_fitness, improvement, top3_fitnesses, mean_time)
    record = []
    log = ""
    prev_log_time = time.time()

    time_check = []
    g = 0
    while True:
        start_time = time.time()
        if g >= ngen:
            break
        g += 1
        txt = f"Generation {g}, "
        time_temp = time.time()
        prev_best_pop = tools.selBest(pop, k=3)
        time_prev_best_pop = time.time() - time_temp

        time_temp = time.time()
        offspring = toolbox.select(pop, MAXPOP)
        offspring = list(map(toolbox.clone, offspring))

        link = []
        for ind in list(offspring):
            if np.random.rand() < RMUT:
                mutant, op = toolbox.mutate(ind)
                if max_term != -1 and len(mutant) > max_term:
                    # don't add the mutant to the offspring
                    continue
                if len(mutant) <= 0 or mutant.ndim != 2:
                    # don't add the mutant to the offspring
                    continue
                offspring.append(mutant)
                link.append((ind, mutant, op))
        time_mutate = time.time() - time_temp

        time_temp = time.time()
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        time_invalid_cand = time.time() - time_temp

        time_temp = time.time()
        num_invalid_ind = len(invalid_ind)
        num_checking = 0
        formula_lengths = [len(ind) for ind in invalid_ind]
        nk_pairs = set()
        nks_all = set()
        for formula in invalid_ind:
            # when computing the fitness (variance), the solutions become
            # feasible by setting n = m_target for n > m_target.
            formula_feasible = helpf1var.mut_avail(formula, m_target)
            nks = []
            for row in formula_feasible:
                nks.append((row[1], row[2]))
            nks = sorted(nks)
            nk_pairs |= set(itertools.combinations(nks, 2)).union(
                set([(nk, nk) for nk in nks])
            )
            nks_all |= set(nks)
        num_checking = len(nk_pairs)
        nan_covs = set()
        for nk_pair in nk_pairs:
            nk1, nk2 = nk_pair
            if cov_matrix_evo.check_item(*nk1, *nk2):
                nan_covs.add((*nk1, *nk2))
        nan_cov_vs_masses = set()
        for nk in nks_all:
            # logging.debug(f"check_item_vs_mass: {nk}, {cov_matrix_evo.check_item_vs_mass(*nk)}, {cov_matrix_emp.cov_vs_mass}")
            if cov_matrix_evo.check_item_vs_mass(*nk):
                nan_cov_vs_masses.add(nk)
        logging.debug(
            f"num_invalid_ind: {num_invalid_ind}, num_checking: {num_checking}, \
num_nan_covs: {len(nan_covs)}, num_nan_cov_vs_masses: {len(nan_cov_vs_masses)}, \
formula_lengths: ({min(formula_lengths)}, {np.mean(formula_lengths):.1f}, {max(formula_lengths)})"
        )
        nan_covs = list(nan_covs)
        time_check_nancov = time.time() - time_temp

        time_temp = time.time()
        if len(nan_covs) > 0:
            cov_matrix_evo.compute_covs(nan_covs, parallel=True)
        if len(nan_cov_vs_masses) > 0:
            cov_matrix_evo.compute_covs_vs_mass(
                nan_cov_vs_masses, parallel=True
            )
        time_compute_covs = time.time() - time_temp

        time_temp = time.time()
        cov_matrix_evo.check_computation_flag = True
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        cov_matrix_evo.check_computation_flag = False
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        time_fitness = time.time() - time_temp

        time_temp = time.time()
        curr_fitnesses = {ind.fitness.values[0] for ind in offspring}
        txt += f"num new fitnesses: {len(curr_fitnesses - prev_fitnesses)}, "
        pop[:] = offspring + initial_pop + prev_best_pop

        fitness_check = set()
        new_pop = []
        for ind in pop:
            if ind.fitness.values not in fitness_check:
                fitness_check.add(ind.fitness.values)
                new_pop.append(ind)
        pop[:] = new_pop
        time_new_pop = time.time() - time_temp

        time_temp = time.time()
        time_check.append(time.time() - start_time)
        mean_time = np.mean(time_check[-10:])
        txt += f"avg time: {mean_time:.1f}s, "
        txt += "Top 3 fitnesses: ("
        top3 = tools.selBest(pop, k=3)
        for ind in top3:
            txt += f"{ind.fitness.values[0]:.1e}, "
        txt += ")"
        log += txt + "\n"
        curr_min_fitness = min([ind.fitness.values[0] for ind in pop])
        improvement = (min_fitness - curr_min_fitness) / min_fitness
        if g >= ngen:
            log += f"\nprev min: {min_fitness:.1e}, curr min: {curr_min_fitness:.1e}, improvement: {improvement * 100:.2f}%\n"
            was_improved = improvement > 0.05
            if ngen >= max_iter:
                log += f"ngen >= {max_iter}. Stop.\n"
            elif was_improved:
                ngen += 100
                min_fitness = curr_min_fitness
            elif (
                min_initial_fitness - curr_min_fitness
            ) / min_initial_fitness < 0.01:
                improvement_from_initial = (
                    min_initial_fitness - curr_min_fitness
                ) / min_initial_fitness
                log += f"Improvement from initial: {improvement_from_initial * 100:.2f}%\n"
                ngen += 100
        time_gen_assess = time.time() - time_temp

        time_temp = time.time()
        # record: (ngen, generation, num_populations, initial_fitness, prevgen_best_fitness, best_fitness, improvement, top3_fitnesses, mean_time)
        rec = [
            ngen,
            g,
            len(pop),
            min_initial_fitness,
            min_fitness,
            curr_min_fitness,
            improvement,
            num_invalid_ind,
            num_checking,
            len(nan_covs),
            min(formula_lengths),
            np.mean(formula_lengths),
            max(formula_lengths),
            time_check_nancov,
            time_compute_covs,
            time_fitness,
        ]
        for top3idx, ind in enumerate(top3):
            rec.append(ind.fitness.values[0])
        if len(top3) < 3:
            for i in range(3 - len(top3)):
                rec.append((-1, -1))
        rec.append(mean_time)
        record.append(rec)
        time_record = time.time() - time_temp

        time_temp = time.time()
        if time.time() - prev_log_time > 5:
            prev_log_time = time.time()
            with open(log_path, "a") as f:
                f.write(log)
            log = ""
            pd.DataFrame(record).to_csv(
                record_path,
                index=False,
                header=[
                    "ngen",
                    "generation",
                    "num_populations",
                    "initial_fitness",
                    "prevgen_best_fitness",
                    "best_fitness",
                    "improvement",
                    "num_no_fitness",
                    "num_cov_checking",
                    "num_nan_covs",
                    "min_formula_length",
                    "mean_formula_length",
                    "max_formula_length",
                    "time_check_nancov",
                    "time_compute_covs",
                    "time_fitness",
                    "top1_fitness",
                    "top2_fitness",
                    "top3_fitness",
                    "mean_time",
                ],
            )
        time_log = time.time() - time_temp

        logging.debug(
            f"Gen {g}. Times [\
PBP: {time_prev_best_pop:.1f}s, \
MUT: {time_mutate:.1f}s, \
INV: {time_invalid_cand:.1f}s, \
NAN: {time_check_nancov:.1f}s, \
COV: {time_compute_covs:.1f}s, \
FIT: {time_fitness:.1f}s, \
POP: {time_new_pop:.1f}s, \
GAS: {time_gen_assess:.1f}s, \
REC: {time_record:.1f}s, \
LOG: {time_log:.1f}s]"
        )
    with open(log_path, "a") as f:
        f.write(log)
    pd.DataFrame(record).to_csv(
        record_path,
        index=False,
        header=[
            "ngen",
            "generation",
            "num_populations",
            "initial_fitness",
            "prevgen_best_fitness",
            "best_fitness",
            "improvement",
            "num_no_fitness",
            "num_cov_checking",
            "num_nan_covs",
            "min_formula_length",
            "mean_formula_length",
            "max_formula_length",
            "time_check_nancov",
            "time_compute_covs",
            "time_fitness",
            "top1_fitness",
            "top2_fitness",
            "top3_fitness",
            "mean_time",
        ],
    )

    return pop, output_dir


def result_assess(
    output_dir, best_ind, m_target, k_target, cov_matrix, cov_matrix_emp
):
    logging.info(f"Best individual:\n{best_ind}")
    with open(os.path.join(output_dir, "best_ind.txt"), "w") as f:
        f.write(str(best_ind))
    assert k_target == 0
    # p_true = cov_matrix.p
    # sample = cov_matrix_emp.sample
    # S = len(p_true)
    # currently only for k_target = 0
    # unseens = np.array(list(set(range(1, S + 1)) - set(sample)))
    # mass_sample = np.sum(p_true[unseens - 1])
    # logging.info(f"Ground truth (smp): {mass_sample:.2e}")
    exp_mass = cov_matrix.exp_mass / (m_target + 1)
    var_mass = cov_matrix.var_mass / (m_target + 1) ** 2
    logging.info(
        f"Ground truth (  n): {exp_mass:.2e} +- {2 * np.sqrt(var_mass):.2e} (95%)"
    )
    goodturing_formula = np.array(
        [[(k_target + 1) / m_target, m_target, k_target + 1]]
    )
    # Phi_matrix = cov_matrix_emp.Phi_matrix
    # goodturing_estimate = helpf1var.compute_formula(
    #     Phi_matrix, goodturing_formula
    # )
    # goodturing_se = (goodturing_estimate - mass_sample) ** 2
    # logging.info(
    #     f"Good-Turing formula (smp): {goodturing_estimate:.2e} (se: {goodturing_se:.2e})"
    # )
    F_matrix = cov_matrix.F_matrix
    exp_goodturing = helpf1var.compute_formula(F_matrix, goodturing_formula)
    var_goodturing = helpf1var.compute_formula_var(
        goodturing_formula, cov_matrix
    )
    cov_goodturing = helpf1var.compute_formula_cov_mass_evo(
        goodturing_formula, cov_matrix
    ) / (m_target + 1)
    mse_goodturing = (
        (exp_goodturing - exp_mass) ** 2
        + var_goodturing
        + var_mass
        - 2 * cov_goodturing
    )
    logging.info(
        f"Good-Turing formula (  n): {exp_goodturing:.2e} +- {2 * np.sqrt(var_goodturing):.2e} (95%, mse: {mse_goodturing:.2e})"
    )
    best_feasible = helpf1var.mut_avail(best_ind, m_target)
    # evo_estimate = helpf1var.compute_formula(Phi_matrix, best_feasible)
    # evo_se = (evo_estimate - mass_sample) ** 2
    # logging.info(
    #     f"Evolution formula (smp): {evo_estimate:.2e} (se: {evo_se:.2e})"
    # )
    exp_evo = helpf1var.compute_formula(F_matrix, best_feasible)
    var_evo = helpf1var.compute_formula_var(best_feasible, cov_matrix)
    cov_evo = helpf1var.compute_formula_cov_mass_evo(
        best_feasible, cov_matrix
    ) / (m_target + 1)
    mse_evo = (exp_evo - exp_mass) ** 2 + var_evo + var_mass - 2 * cov_evo
    logging.info(
        f"Evolution formula (  n): {exp_evo:.2e} +- {2 * np.sqrt(var_evo):.2e} (95%, mse: {mse_evo:.2e})"
    )
    result = pd.DataFrame(
        [
            ["Ground truth", exp_mass, var_mass, 0],
            [
                "Good-Turing formula",
                # goodturing_estimate,
                # goodturing_se,
                exp_goodturing,
                var_goodturing,
                mse_goodturing,
            ],
            [
                "Evolution formula",
                # evo_estimate,
                # evo_se,
                exp_evo,
                var_evo,
                mse_evo,
            ],
        ],
        columns=["formula", "exp", "var", "mse"],
    )
    result.to_csv(os.path.join(output_dir, "assess.csv"), index=False)


if __name__ == "__main__":
    # Notice:
    # - The algorithm evolves the formula for F_{k_target + 1}(n_target + 1),
    #   instead of M_{k_target}(n_target)
    #              = (k_target+1)/(n_target+1) * F_{k_target+1}(n_target+1)
    # - After the evoluation, for final formula is multiplied by
    #   (k_target+1)/(n_target+1) to get the final formula for
    #   \hat M_{k_target}(n_target)
    # - The base formula during the evolution is [1, n_target+1, k_target+1].
    #   The base formula at the assessment is
    #   [(k_target+1)/(n_target+1), n_target+1, k_target+1)].
    # - The Good-Turing formula only appears in the assessment. Therefore,
    #   it is [(k_target+1)/n_target, n_target, k_target+1)]

    args = get_args()
    # print all arguments for logging
    logging.info(args)
    n_total = args.n_total
    m_target = args.m_target
    k_target = args.k_target
    # k_target = 0

    # The disttype 'diri' and 'dirihalf' generate p from the dirichlet
    # distribution. Therefore p is not fixed. It will be generated for each
    # seed in the inner loop.
    is_dist_fixed = args.disttype not in ["diri", "dirihalf"]
    # get p, F_matrix, cov_matrix for the fixed distribution
    if is_dist_fixed:
        p, F_matrix, cov_matrix = get_cov_matrix(
            args.S, args.disttype, n_total, m_target, k_target
        )

    # base formula
    base_formula = np.array([[1, m_target + 1, k_target + 1]])

    if args.setting == "knowing":
        logging.info("setting: knowing")
        # get toolbox
        if is_dist_fixed:
            toolbox = get_toolbox(
                [base_formula], m_target, k_target, cov_matrix
            )
        # evolution
        evo_output_dir = os.path.join(result_folder, "evolutions", "knowing")
        os.makedirs(evo_output_dir, exist_ok=True)
        next_evo_idx = len(os.listdir(evo_output_dir))
        evo_idxs = range(next_evo_idx, next_evo_idx + args.rep_evo)
        logging.info(f"next_evo_idx: {next_evo_idx}, evo_idxs: {evo_idxs}")
        for rep_evo, evo_idx in enumerate(evo_idxs):
            logging.info(
                f"Evolution rep {rep_evo + 1}/{len(evo_idxs)} ({evo_idx=})"
            )
            # get p, F_matrix, cov_matrix, and toolbox for the unfixed
            # distribution
            if not is_dist_fixed:
                postfix = f"-evo{evo_idx}"
                p, F_matrix, cov_matrix = get_cov_matrix(
                    args.S, args.disttype, n_total, m_target, k_target, postfix
                )
                toolbox = get_toolbox(
                    [base_formula], m_target, k_target, cov_matrix
                )
            pop, output_dir = run_evolution(
                toolbox,
                m_target,
                cov_matrix,
                evo_output_dir,
                evo_idx,
                args.max_term,
                args.max_iter,
            )
            logging.info("Evolution finished. Assessing results.")
            best_ind = tools.selBest(pop, k=1)[0]
            # compensate the coefficient (k_target+1)/(m_target+1) before
            # the assessment
            best_ind[:, 0] *= (k_target + 1) / (m_target + 1)
            result_assess(
                output_dir, best_ind, m_target, k_target, cov_matrix, cov_matrix
            )
            # if the distribution is not fixed, save covmat here
            if not is_dist_fixed:
                cov_matrix_path = os.path.join(
                    result_folder,
                    "Phi_matrix",
                    f"covmat{postfix}.pkl",
                )
                helpf1var.CovMat.save_covmat(cov_matrix, cov_matrix_path)
        # otherwise, save covmat here
        if is_dist_fixed:
            cov_matrix_path = os.path.join(
                result_folder, "Phi_matrix", "covmat.pkl"
            )
            helpf1var.CovMat.save_covmat(cov_matrix, cov_matrix_path)
    elif args.setting == "sampling":
        logging.info("setting: sampling")
        sample_seeds = range(args.seed, args.seed + args.rep_smp)
        logging.info(f"sample_seeds: {sample_seeds}")
        for rep_seed, seed in enumerate(sample_seeds):
            logging.info(
                f"Seed rep {rep_seed + 1}/{len(sample_seeds)} ({seed=})"
            )
            # get p, F_matrix, cov_matrix for the unfixed distribution
            if not is_dist_fixed:
                postfix = f"-seed{seed}"
                p, F_matrix, cov_matrix = get_cov_matrix(
                    args.S, args.disttype, n_total, m_target, k_target, postfix
                )

            # get/load cov_matrix_emp
            cov_matrix_emp = get_cov_matrix_emp(
                p,
                n_total,
                m_target,
                k_target,
                seed,
                args.p_est_method,
            )

            # get toolbox
            toolbox = get_toolbox(
                [base_formula], m_target, k_target, cov_matrix_emp
            )
            # evolution
            evo_output_dir = os.path.join(
                result_folder, "evolutions", f"seed{seed}"
            )
            os.makedirs(evo_output_dir, exist_ok=True)
            next_evo_idx = len(os.listdir(evo_output_dir))
            evo_idxs = range(next_evo_idx, next_evo_idx + args.rep_evo)
            logging.info(f"next_evo_idx: {next_evo_idx}, evo_idxs: {evo_idxs}")
            for rep_evo, evo_idx in enumerate(evo_idxs):
                logging.info(
                    f"Evolution rep {rep_evo + 1}/{len(evo_idxs)} ({evo_idx=})"
                )
                pop, output_dir = run_evolution(
                    toolbox,
                    m_target,
                    cov_matrix_emp,
                    evo_output_dir,
                    evo_idx,
                    args.max_term,
                    args.max_iter,
                )
                logging.info("Evolution finished. Assessing results.")
                best_ind = tools.selBest(pop, k=1)[0]
                # compensate the coefficient (k_target+1)/(m_target+1) before
                # the assessment
                best_ind[:, 0] *= (k_target + 1) / (m_target + 1)
                result_assess(
                    output_dir,
                    best_ind,
                    m_target,
                    k_target,
                    cov_matrix,
                    cov_matrix_emp,
                )

            # save covmat_emp
            cov_matrix_emp_path = os.path.join(
                result_folder, "Phi_matrix", f"covmatsmp-seed{seed}.pkl"
            )
            helpf1var.CovMat.save_covmat(cov_matrix_emp, cov_matrix_emp_path)
            # if the distribution is not fixed, save covmat here
            if not is_dist_fixed:
                cov_matrix_path = os.path.join(
                    result_folder, "Phi_matrix", f"covmat{postfix}.pkl"
                )
                helpf1var.CovMat.save_covmat(cov_matrix, cov_matrix_path)
        # otherwise, save covmat here
        if is_dist_fixed:
            cov_matrix_path = os.path.join(
                result_folder, "Phi_matrix", f"covmat.pkl"
            )
            helpf1var.CovMat.save_covmat(cov_matrix, cov_matrix_path)
