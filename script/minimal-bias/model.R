library(gmp)
library(Rmpfr)
library(extraDistr)

get_model <- function(model_idx, prec, S = 10000, n = 5000) {
  if (model_idx == 1) {
    # Model 1. Unifrom
    p <- rep(mpfr(1, prec), times = S)
    p <- p / sum(p)
  } else if (model_idx == 2) {
    # Model 2. 1/2 * 1 + 1/2 * 3
    p <- c(
      rep(mpfr(1, prec), times = 0.5 * S),
      rep(mpfr(3, prec), times = 0.5 * S)
    )
    p <- p / sum(p)
  } else if (model_idx == 3) {
    # Model 3. Zipf distribution with parameter 1
    p <- mpfr(seq(1, S, 1), prec)
    p <- p**(-1)
    p <- p / sum(p)
  } else if (model_idx == 4) {
    # Model 4. Zipf distribution with parameter 1 / 2
    p <- mpfr(seq(1, S, 1), prec)
    p <- p**(-0.5)
    p <- p / sum(p)
  } else if (model_idx == 5) {
    # Model 5. Dirichlet-1 prior
    p <- rdirichlet(1, rep(1, S))
    p <- mpfr(c(p), prec)
    p <- p / sum(p)
  } else if (model_idx == 6) {
    # Model 6. Dirichlet-1/2 prior
    p <- rdirichlet(1, rep(1 / 2, S))
    p <- mpfr(c(p), prec)
    p <- p / sum(p)
  }
  return(list(S = S, p = p, n = n))
}
