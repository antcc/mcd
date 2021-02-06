set.seed(42)

# Parameters
R <- 1000
n <- 10
alpha <- 0.05

# Original sample
sample_orig <- c(1,2,3.5,4,7,7.3,8.6,12.4,13.8,18.1)
var_orig <- var(sample_orig)

# Bootstrap resamples (n x R matrix, one resample for each column)
resample_bootstrap <- sample(sample_orig, n * R, rep = T)
resample_bootstrap <- matrix(resample_bootstrap, nrow = n)

# Variance of the bootstrap resamples
var_bootstrap <- apply(resample_bootstrap, 2, var)

# Bootstrap estimator T*
T_bootstrap <- sqrt(n) * (var_bootstrap - var_orig)

# Get limits of the confidence interval
ci_left <- var_orig - quantile(T_bootstrap, 1 - alpha/2)/sqrt(n)
ci_right <- var_orig - quantile(T_bootstrap, alpha/2)/sqrt(n)

cat("Confidence interval for σ²: (", ci_left, ",", ci_right, ")\n")
