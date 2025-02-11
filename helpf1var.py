import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools
import multiprocessing
import pickle
from typing import Callable, List, Tuple
from math import exp, ceil, comb, factorial
import warnings
from collections import Counter
from scipy.optimize import fsolve

warnings.filterwarnings("ignore", "The iteration is not making good progress")

# Set the number of processes to parallelize the evolutionary algorithm.
MAXCORES = 32

# factorial = np.math.factorial

################################################################################
# Variance computation
################################################################################


def minimal_var_compute(
    n: int, k: int, snd_term: float, p: np.ndarray
) -> float:
    v_var = snd_term
    if 2 * k <= n:
        coef = factorial(n) / (factorial(k) ** 2 * factorial(n - 2 * k))
        bothk = 0
        for xidx in range(len(p) - 1):
            for yidx in range(xidx + 1, len(p)):
                bothk += (
                    p[xidx] ** k
                    * p[yidx] ** k
                    * (1 - p[xidx] - p[yidx]) ** (n - 2 * k)
                )
        v_var += 2 * coef * bothk
    return v_var


def minimal_cov_compute(
    n1: int, k1: int, n2: int, k2: int, snd_term: float, p: np.ndarray
):
    if n1 == n2 and k1 == k2:
        return minimal_var_compute(n1, k1, snd_term, p)
    if n1 < n2:
        n1, k1, n2, k2 = n2, k2, n1, k1
    fst_term = 0
    for lxidx in range(len(p)):
        for rxidx in range(len(p)):
            if n1 == n2:
                if lxidx == rxidx:
                    if k1 == k2:
                        fst_term += (
                            n2.choose(n1, k1)
                            * p[lxidx] ** k1
                            * (1 - p[lxidx]) ** (n1 - k1)
                        )
                else:
                    if k1 + k2 <= n1:
                        fst_term += (
                            factorial(n1)
                            / (
                                factorial(k1)
                                * factorial(k2)
                                * factorial(n1 - k1 - k2)
                            )
                            * p[lxidx] ** k1
                            * p[rxidx] ** k2
                            * (1 - p[lxidx] - p[rxidx]) ** (n1 - k1 - k2)
                        )
            else:
                # n1 > n2
                if lxidx == rxidx:
                    if k1 >= k2 and n1 - n2 >= k1 - k2:
                        fst_term += (
                            factorial(n2)
                            / (factorial(k2) * factorial(n2 - k2))
                            * factorial(n1 - n2)
                            / (
                                factorial(k1 - k2)
                                * factorial(n1 - n2 - k1 + k2)
                            )
                            * p[lxidx] ** k1
                            * (1 - p[lxidx]) ** (n1 - k1)
                        )
                else:
                    if k1 + k2 <= n1:
                        for i in range(
                            max(0, k1 - (n1 - n2)),
                            min(k1, n2 - k2) + 1,
                        ):
                            fst_term += (
                                factorial(n2)
                                / (
                                    factorial(k2)
                                    * factorial(i)
                                    * factorial(n2 - k2 - i)
                                )
                                * factorial(n1 - n2)
                                / (
                                    factorial(k1 - i)
                                    * factorial(n1 - n2 - k1 + i)
                                )
                                * p[lxidx] ** k1
                                * (1 - p[lxidx]) ** (n1 - n2 - k1 + i)
                                * p[rxidx] ** k2
                                * (1 - p[rxidx] - p[lxidx]) ** (n2 - k2 - i)
                            )
            if fst_term is np.nan:
                raise ValueError(
                    f"error n1: {n1}, k1: {k1}, n2: {n2}, k2: {k2}, lxidx: {lxidx}, rxidx: {rxidx}"
                )
    return fst_term - snd_term


def minimal_cov_vs_mass_compute(
    n: int, k: int, m_mass: int, k_mass: int, snd_term: float, p: np.ndarray
):
    S = len(p)
    exp_mass_phi = 0
    for i_mass in range(S):
        for i_phi in range(S):
            if i_mass == i_phi:
                exp_imm_ievo = 0 if k > 0 else (1 - p[i_mass]) ** m_mass
            else:
                exp_imm_ievo = (
                    mult_well(get_binom(n, k))
                    * p[i_phi] ** k
                    * (1 - p[i_phi] - p[i_mass]) ** (n - k)
                    * (1 - p[i_mass]) ** (m_mass - n)
                )
            exp_mass_phi += p[i_mass] * exp_imm_ievo
    # (m_mass + 1), which actually should be comb(m_mass + 1, k_mass + 1)
    # is the term to make
    # M_{k_mass}(m_mass) to
    # comb(m_mass + 1, k_mass + 1) * M_{k_mass}(m_mass)
    return exp_mass_phi * (m_mass + 1) - snd_term


class F_mat:
    def __init__(self, p):
        self.p = p
        self._F_matrix = {}

    def __getitem__(self, nk):
        n, k = nk
        if (n, k) not in self._F_matrix:
            self._F_matrix[(n, k)] = self._compute_F(n, k)
        return self._F_matrix[(n, k)]

    def _compute_F(self, n, k):
        if k == 0:
            return np.sum(self.p**n)
        else:
            return mult_well(get_binom(n, k)) * np.sum(
                self.p**k * (1 - self.p) ** (n - k)
            )


class CovMat:
    def __init__(self, p, n, m, k, F_matrix: F_mat, sample=None):
        # Seems the refactoring is needed.
        # Currently CovMat/GA is happening in phi level. And later convert to
        # probability mass level.
        # However, the random variable phi / binom and probability mass are not
        # exactly the same. (e.g., expected value is the same, but variance is
        # different.)
        # So, the refactoring is needed to make the code more readable and
        # understandable.
        self.p = p
        self.n = n
        self.m = m
        self.k = k
        # self.nks = self.gen_nks()
        self.F_matrix = F_matrix
        self.sample = sample
        # self.Phi_matrix = Phi_matrix
        self.exp_mass, self.var_mass = self.stat_mass()
        self.mass_dict = {}
        self.cov_matrix = {}
        self.cov_vs_mass = {}
        self.check_computation_flag = False

    def stat_mass(self) -> Tuple[float, float]:
        # The truth is, it computes the statistics of mass(m, k) * comb(m, k)
        # Need to be refactored.
        p, m, k = self.p, int(self.m), int(self.k)
        assert k == 0  # FIXME: currently only support k=0
        # below (m) actually should ben comb(m, k)
        exp_mass = np.sum(p ** (k + 1) * (1 - p) ** (m - k)) * (m + 1)
        var_mass_sq = np.sum(p**2 * (1 - p) ** m * (1 - (1 - p) ** m))
        var_mass_cov = 0
        S = len(p)
        for i in range(S):
            for j in range(i + 1, S):
                var_mass_cov += (
                    p[i]
                    * p[j]
                    * (
                        (1 - p[i] - p[j]) ** m
                        - (1 - p[i]) ** m * (1 - p[j]) ** m
                    )
                )
        var_mass_cov *= 2
        # below (m) actually should ben comb(m, k)
        var_mass = (var_mass_sq + var_mass_cov) * (m + 1) ** 2
        return exp_mass, var_mass

    # def gen_nks(self):
    #     nks = []
    #     for ni in range(self.n + 1):
    #         for k in range(ni + 1):
    #             nks.append([ni, k])
    #     return np.array(nks)

    # def get_covmat_idx(self, n, k):
    #     idxs = np.where((self.nks[:, 0] == n) & (self.nks[:, 1] == k))
    #     if len(idxs[0]) == 0:
    #         raise ValueError(f"({n}, {k}) not in nks")
    #     return idxs[0][0]

    def __getitem__(self, nks):
        n1, k1 = nks[0]
        n2, k2 = nks[1]
        n1, k1, n2, k2 = int(n1), int(k1), int(n2), int(k2)
        # lidx = self.get_covmat_idx(n1, k1)
        # ridx = self.get_covmat_idx(n2, k2)
        # lidx, ridx = min(lidx, ridx), max(lidx, ridx)
        if n1 < n2 or (n1 == n2 and k1 < k2):
            lidx, ridx = (n1, k1), (n2, k2)
        else:
            lidx, ridx = (n2, k2), (n1, k1)
        if (lidx, ridx) not in self.cov_matrix:
            if self.check_computation_flag:
                raise ValueError(
                    f"({n1}, {k1}), ({n2}, {k2}) not computed yet!"
                )
            # lazy evaluation
            if lidx == ridx:
                f = self.F_matrix[n1, k1]
                snd_term = f * (1 - f)
                self.cov_matrix[(lidx, lidx)] = minimal_var_compute(
                    n1, k1, snd_term, self.p
                )
            else:
                # if n1 != n2, make n1 > n2
                snd_term = self.F_matrix[n1, k1] * self.F_matrix[n2, k2]
                v_cov = minimal_cov_compute(n1, k1, n2, k2, snd_term, self.p)
                self.cov_matrix[(lidx, ridx)] = v_cov
        return self.cov_matrix[(lidx, ridx)]

    # check the item
    def check_item(self, n1: int, k1: int, n2: int, k2: int) -> bool:
        # lidx = self.get_covmat_idx(n1, k1)
        # ridx = self.get_covmat_idx(n2, k2)
        # lidx, ridx = min(lidx, ridx), max(lidx, ridx)
        if n1 < n2 or (n1 == n2 and k1 < k2):
            lidx, ridx = (n1, k1), (n2, k2)
        else:
            lidx, ridx = (n2, k2), (n1, k1)
        return (lidx, ridx) not in self.cov_matrix

    def check_item_vs_mass(self, n: int, k: int) -> bool:
        # idx = self.get_covmat_idx(n, k)
        return (n, k) not in self.cov_vs_mass

    # multi processing
    def compute_covs(
        self, nkpairs: List[Tuple[int]], parallel: bool = False
    ) -> None:
        if not parallel:
            for n1, k1, n2, k2 in nkpairs:
                self[(n1, k1), (n2, k2)]  # lazy evaluation
        else:
            params = []
            for n1, k1, n2, k2 in nkpairs:
                n1, k1, n2, k2 = int(n1), int(k1), int(n2), int(k2)
                if n1 == n2 and k1 == k2:
                    f = self.F_matrix[n1, k1]
                    params.append((n1, k1, n2, k2, f * (1 - f), self.p))
                else:
                    params.append(
                        (
                            n1,
                            k1,
                            n2,
                            k2,
                            self.F_matrix[n1, k1] * self.F_matrix[n2, k2],
                            self.p,
                        )
                    )
            with multiprocessing.Pool(min(MAXCORES, len(params))) as p:
                results = p.starmap(minimal_cov_compute, params)
            for idx, (n1, k1, n2, k2) in enumerate(nkpairs):
                # lidx = self.get_covmat_idx(n1, k1)
                # ridx = self.get_covmat_idx(n2, k2)
                # lidx, ridx = min(lidx, ridx), max(lidx, ridx)
                if n1 < n2 or (n1 == n2 and k1 < k2):
                    lidx, ridx = (n1, k1), (n2, k2)
                else:
                    lidx, ridx = (n2, k2), (n1, k1)
                self.cov_matrix[(lidx, ridx)] = results[idx]

    def compute_covs_vs_mass(
        self, nks: List[Tuple[int]], parallel: bool = False
    ) -> None:
        m_mass, k_mass = self.m, self.k
        assert k_mass == 0  # FIXME: currently only support k_mass = self.k = 0
        if not parallel:
            for n, k in nks:
                cov_vs_mass = minimal_cov_vs_mass_compute(
                    n,
                    k,
                    m_mass,
                    k_mass,
                    self.F_matrix[n, k] * self.exp_mass,
                    self.p,
                )
                self.cov_vs_mass[(n, k)] = cov_vs_mass
        else:
            params = []
            for n, k in nks:
                n, k = int(n), int(k)
                params.append(
                    (
                        n,
                        k,
                        m_mass,
                        k_mass,
                        self.F_matrix[n, k] * self.exp_mass,
                        self.p,
                    )
                )
            with multiprocessing.Pool(min(MAXCORES, len(params))) as p:
                results = p.starmap(minimal_cov_vs_mass_compute, params)
            for result_idx, (n, k) in enumerate(nks):
                # idx = self.get_covmat_idx(n, k)
                self.cov_vs_mass[(n, k)] = results[result_idx]

    def get_cov_vs_mass(self, n, k):
        m_mass, k_mass = self.m, self.k
        # idx = self.get_covmat_idx(n, k)
        if (n, k) not in self.cov_vs_mass:
            if self.check_computation_flag:
                raise ValueError(f"({n}, {k}) not computed yet!")
            cov_vs_mass = minimal_cov_vs_mass_compute(
                n,
                k,
                m_mass,
                k_mass,
                self.F_matrix[n, k] * self.exp_mass,
                self.p,
            )
            self.cov_vs_mass[(n, k)] = cov_vs_mass
        return self.cov_vs_mass[(n, k)]

    @staticmethod
    def save_covmat(covmat, filename):
        with open(filename, "wb") as f:
            pickle.dump(covmat, f)

    @staticmethod
    def load_covmat(filename):
        with open(filename, "rb") as f:
            covmat = pickle.load(f)
        return covmat


def compute_formula_var(formula_mat: np.ndarray, cov_matrix: CovMat):
    v_var = 0
    # iterate row for variance
    for ridx in range(len(formula_mat)):
        coef, n, k = formula_mat[ridx]
        v_var += coef**2 * cov_matrix[(n, k), (n, k)]
    # iterate row X row for covariance
    if len(formula_mat) > 1:
        for lidx in range(len(formula_mat)):
            for ridx in range(lidx + 1, len(formula_mat)):
                coef_l, n_l, k_l = formula_mat[lidx]
                coef_r, n_r, k_r = formula_mat[ridx]
                v_var += (
                    2 * coef_l * coef_r * cov_matrix[(n_l, k_l), (n_r, k_r)]
                )
    return v_var


def compute_formula_cov_mass_evo(formula_mat: np.ndarray, cov_matrix: CovMat):
    cov_mass_evo = 0
    for coef, n, k in formula_mat:
        cov_mass_evo += coef * cov_matrix.get_cov_vs_mass(int(n), int(k))
    return cov_mass_evo


def compute_formula_mse(
    formula_mat: np.ndarray, cov_matrix: CovMat, m_target: int, k_target: int
):
    # The evolutionary algorithm evolves for
    # M_{k_target}(m_target) * binom(m_target, k_target),
    # not M_{k_target}(m_target).
    # To make the formula feasible, change all n' (second column) to m_target
    # when n' > m_target.
    formula_mat_feasible = mut_avail(formula_mat, m_target)
    exp_mass = cov_matrix.exp_mass
    var_mass = cov_matrix.var_mass
    exp_evo = compute_formula(cov_matrix.F_matrix, formula_mat_feasible)
    var_evo = compute_formula_var(formula_mat_feasible, cov_matrix)
    cov_mass_evo = compute_formula_cov_mass_evo(
        formula_mat_feasible, cov_matrix
    )
    return (exp_evo - exp_mass) ** 2 + var_evo + var_mass - 2 * cov_mass_evo


################################################################################
# Helpers for the evoluationary algorithm
################################################################################
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)


def initIndividual(icls, content):
    return icls(content)


def initPopulation(pcls, ind_init, formulas):
    return pcls(ind_init(c) for c in formulas)


def cx(ind1, ind2):
    if np.random.rand() < 0.5:  # weighted average
        weight = np.random.rand()
    else:
        weight = -np.random.randint(2, 10)
    newind1 = np.copy(ind1)
    newind1[:, 0] = weight * ind1[:, 0]
    newind2 = np.copy(ind1)
    newind2[:, 0] = (1 - weight) * ind1[:, 0]
    for ridx in range(len(ind2)):
        # check if (n, k) exists in ind1
        coef, n, k = ind2[ridx]
        if np.sum((ind1[:, 1:] == [n, k]).all(axis=1)) == 0:
            newind1 = np.vstack((newind1, [coef * (1 - weight), n, k]))
            newind2 = np.vstack((newind2, [coef * weight, n, k]))
        else:
            idx = np.where((ind1[:, 1:] == [n, k]).all(axis=1))[0][0]
            newind1[idx, 0] += coef * (1 - weight)
            newind2[idx, 0] += coef * weight
    return initIndividual(creator.Individual, newind1), initIndividual(
        creator.Individual, newind2
    )


def mut(ind, n_available, op=None):
    newind = np.copy(ind)
    if op:
        n, k, case, ratio = op
        assert np.where((newind[:, 1:] == [n, k]).all(axis=1))[0][0]
        idx = np.where((newind[:, 1:] == [n, k]).all(axis=1))[0][0]
        coef = newind[idx, 0]
    else:
        # choose random n, k from ind
        idx = np.random.randint(len(newind))
        coef, n, k = newind[idx]

        # choose random case
        # case 0: g_n_k -> g_(n-1)_(k-1) - g_n_(k-1)
        # case 1: g_n_k -> g_(n-1)_k - g_n_(k+1)
        # case 2: g_n_k -> g_(n+1)_k + g_(n+1)_(k+1)
        cases = [0, 1, 2]
        if n > n_available:
            cases.remove(2)
        if k <= 1:
            cases.remove(0)
        if k >= n:
            cases.remove(1)
        case = np.random.choice(cases, size=1)[0]

        # choose random ratio
        # for half of the chance, ratio = 1, otherwise, ratio ~ U(0, 1)
        if np.random.rand() < 0.5:
            ratio = 1
        else:
            ratio = np.random.rand()

    coef_update = coef * (1 - ratio)
    coef_distribute = coef * ratio

    if ratio < 1:
        newind[idx, 0] = coef_update
    else:
        newind = np.delete(newind, idx, axis=0)
    gcoef_distribute = coef_distribute * comb(int(n), int(k))
    if case == 0:
        new_n1, new_k1, new_n2, new_k2 = n - 1, k - 1, n, k - 1
    elif case == 1:
        new_n1, new_k1, new_n2, new_k2 = n - 1, k, n, k + 1
    elif case == 2:
        new_n1, new_k1, new_n2, new_k2 = n, k, n + 1, k + 1
    new_coef1 = gcoef_distribute / comb(int(new_n1), int(new_k1))
    new_coef2 = (
        gcoef_distribute
        / comb(int(new_n2), int(new_k2))
        * (1 if case == 2 else -1)
    )
    # check if (new_n1, new_k1) exists in ind
    if np.sum((newind[:, 1:] == [new_n1, new_k1]).all(axis=1)) == 0:
        newind = np.vstack((newind, [new_coef1, new_n1, new_k1]))
    else:
        idx = np.where((newind[:, 1:] == [new_n1, new_k1]).all(axis=1))[0][0]
        newind[idx, 0] += new_coef1
    # check if (new_n2, new_k2) exists in ind
    if np.sum((newind[:, 1:] == [new_n2, new_k2]).all(axis=1)) == 0:
        newind = np.vstack((newind, [new_coef2, new_n2, new_k2]))
    else:
        idx = np.where((newind[:, 1:] == [new_n2, new_k2]).all(axis=1))[0][0]
        newind[idx, 0] += new_coef2

    # drop variables with small coefficients
    coef_threshold = 0.0001
    newind = newind[abs(newind[:, 0]) > coef_threshold]

    # assert all the values in the second column are smaller than or equal to
    # n_available + 1
    assert np.all(newind[:, 1] <= n_available + 1)

    return initIndividual(creator.Individual, newind), (coef, n, k, case, ratio)


def mut_avail(ind, n_available):
    newind = np.copy(ind)
    # if k > n_available, delete the row
    try:
        idxs = np.where(newind[:, 2] > n_available)[0]
    except IndexError as e:
        print(f"Index Error in mut_avail! {e=}")
        print(f"{newind=}")
        # save error msg
        with open("mut_avail_error.txt", "a") as f:
            f.write(f"{e=}\n")
            f.write(f"{newind=}\n")
        raise e
    newind = np.delete(newind, idxs, axis=0)
    # convert any Phi(n, k) where n > n_available to Phi(n_available, k)
    del_idxs = []
    for idx in range(len(newind)):
        coef, n, k = newind[idx]
        if n > n_available:
            del_idxs.append(idx)
            # check if (n_available, k) exists in ind
            if np.sum((newind[:, 1:] == [n_available, k]).all(axis=1)) == 0:
                newind = np.vstack((newind, [coef, n_available, k]))
            else:
                idxs = np.where(
                    (newind[:, 1:] == [n_available, k]).all(axis=1)
                )[0]
                newind[idxs[0], 0] += coef
    newind = np.delete(newind, del_idxs, axis=0)
    zero_coef_idxs = np.where(newind[:, 0] == 0)[0]
    newind = np.delete(newind, zero_coef_idxs, axis=0)
    return initIndividual(creator.Individual, newind)


def evaluate(individual, cov_matrix, m_target, k_target):
    return (compute_formula_mse(individual, cov_matrix, m_target, k_target),)


################################################################################
# formula generation
################################################################################
def mult_well(mult_cand: np.ndarray) -> np.float64:
    if 0 in mult_cand:
        return 0
    # check if there is nan or inf
    if np.isnan(mult_cand).any() or np.isinf(mult_cand).any():
        raise ValueError("mult_cand contains NaN or Inf!")
    sign = 1 if sum(mult_cand < 0) % 2 == 0 else -1
    mult_cand = np.abs(mult_cand)
    mult_cand = [x for x in mult_cand if x != 1]
    if len(mult_cand) == 0:
        return sign
    larger_than_1 = sorted([x for x in mult_cand if x >= 1], reverse=True)
    smaller_than_1 = sorted([x for x in mult_cand if x < 1])
    ret = np.float64(1)
    while True:
        if ret == 0:
            return 0
        elif len(larger_than_1) == 0:
            return sign * ret * np.prod(smaller_than_1)
        elif len(smaller_than_1) == 0:
            return sign * ret * np.prod(larger_than_1)
        if ret > 1:
            ret *= smaller_than_1.pop()
        else:
            ret *= larger_than_1.pop()


def get_binom(n, k) -> np.ndarray:
    n, k = int(n), int(k)
    if n <= 1 or k == 0 or n == k:
        return np.array([1])
    else:
        ret = []
        for i in range(1, k + 1):
            ret = ret + [n - i + 1, 1 / i]
        return np.array(ret, dtype=np.float64)


def gen_F(p):
    # F_matrix = np.zeros((n + 1, n + 1), dtype=np.float64)
    # for ni in range(n + 1):
    #     f_n_n = np.sum(p**ni)
    #     F_matrix[ni, ni] = f_n_n
    #     if ni > 0:
    #         for k in range(ni - 1, -1, -1):
    #             print(f"n={ni}, k={k}", end="\r", flush=True)
    #             gt_f_n_k = mult_well(get_binom(ni, k)) * np.sum(
    #                 p**k * (1 - p) ** (ni - k)
    #             )
    #             F_matrix[ni, k] = gt_f_n_k
    return F_mat(p)


# generate Phi (freq. of freq.) matrix from observations
def gen_Phi(obs, S) -> np.ndarray:
    total_n = len(obs)
    Phi_matrix = np.zeros((total_n + 1, total_n + 1))
    Phi_matrix[0, 0] = S
    for ni in range(1, total_n + 1):
        sub_obs = obs[:ni]
        freq_sub_obs = np.array([np.sum(sub_obs == i) for i in range(1, S + 1)])
        for k in range(ni + 1):
            Phi_matrix[ni, k] = sum(freq_sub_obs == k)
    return np.array(Phi_matrix, dtype=int)


def gen_Phi_fromsample(p, n, seed=None) -> Tuple[np.ndarray, np.ndarray]:
    S = len(p)
    if seed:
        np.random.seed(seed)
    obs = np.random.choice(range(1, S + 1), size=n, p=p, replace=True)
    return gen_Phi(obs, S), obs


def compute_formula(matrix, formula_mat):
    v = 0
    for coef, n, k in formula_mat:
        n, k = int(n), int(k)
        v += coef * matrix[n, k]
    return v


def load_formula(ind_path) -> np.ndarray:
    formula = []
    with open(ind_path, "r") as f:
        for line in f.readlines():
            formula.append(line.strip().strip("[]").split())
    return np.array(formula, dtype=float)


################################################################################
# Additional functions
################################################################################


def num_undetected_lower_bound(
    n: int, num_singletons: int, num_doubletons: int
) -> float:
    if num_doubletons > 0:
        return ((n - 1) / n) * ((num_singletons**2) / (2 * num_doubletons))
    else:  # todo: check and prohibit negative
        return ((n - 1) / n) * (num_singletons * (num_singletons - 1) / 2)
