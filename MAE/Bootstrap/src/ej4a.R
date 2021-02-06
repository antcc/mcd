library(ggplot2)
set.seed(42)

# Parameters
R <- 1000
n <- 10

# Original sample
sample_orig <- c(1,2,3.5,4,7,7.3,8.6,12.4,13.8,18.1)
var_orig <- var(sample_orig)

# Bootstrap resamples (n x R matrix, one resample for each column)
resample_bootstrap <- sample(sample_orig, n * R, rep = T)
resample_bootstrap <- matrix(resample_bootstrap, nrow = n)

# Variance of the bootstrap resamples
var_bootstrap <- apply(resample_bootstrap, 2, var)

# Histogram of bootstrap variances
df <- data.frame(var_bootstrap = var_bootstrap)
ggplot(df, aes(x = var_bootstrap)) +
  geom_histogram(aes(y = ..density..),
                 bins = 20, fill = "#69b3a2", col = 'black') +
  geom_vline(xintercept = var_orig, size = 1.1, col ='red') +
  geom_density(aes(y=..density..))

# Bootstrap estimate of the sd of the variance
sd_var <- sd(var_bootstrap)
cat("Bootstrap estimate:", sd_var, "\n")
