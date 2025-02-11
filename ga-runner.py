import subprocess
import multiprocessing as mp


def subprocess_test(args):
    (
        S,
        dist,
        n,
        m,
        k,
        setting,
        p_est_method,
        seed,
        rep_evo,
        max_term,
        max_iter,
    ) = args
    script = f"python ga.py {S} {dist} {n} {m} {k} {setting} --p_est_method {p_est_method} --seed {seed} --rep_smp 1 --rep_evo {rep_evo} --max_term {max_term} --max_iter {max_iter}"
    print(script)
    subprocess.run(
        script,
        shell=True,
    )


if __name__ == "__main__":
    planner_path = "ga-planner.txt"
    # Set the number of processes to parallelize the suite of experiments.
    # Notice that each experiment also uses parallelization for the
    # evolutionary algorithm. (see help11var.py)
    max_procs = 24

    with open(planner_path, "r") as f:
        lines = f.readlines()[1:]
    if len(lines) == 0:
        raise ValueError("Empty planner file")
    args = []
    for line in lines:
        if line.strip() == "":
            continue
        if line.startswith("#"):
            continue
        (
            S,
            dist,
            n,
            m,
            k,
            setting,
            p_est_method,
            seed,
            rep_smp,
            rep_evo,
            max_term,
            max_iter,
        ) = line.strip().split()
        S, n, m, k, seed, rep_smp, rep_evo, max_term, max_iter = (
            int(S),
            int(n),
            int(m),
            int(k),
            int(seed),
            int(rep_smp),
            int(rep_evo),
            int(max_term),
            int(max_iter),
        )
        if setting == "knowing":
            args.append(
                (
                    S,
                    dist,
                    n,
                    m,
                    k,
                    setting,
                    p_est_method,
                    -1,
                    rep_evo,
                    max_term,
                    max_iter,
                )
            )
        elif setting == "sampling":
            for i in range(seed, seed + rep_smp):
                args.append(
                    (
                        S,
                        dist,
                        n,
                        m,
                        k,
                        setting,
                        p_est_method,
                        i,
                        rep_evo,
                        max_term,
                        max_iter,
                    )
                )

    with mp.Pool(min(max_procs, len(args))) as pool:
        pool.map(subprocess_test, args)
