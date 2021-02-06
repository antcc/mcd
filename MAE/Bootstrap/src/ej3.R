library(ggplot2)
set.seed(42)

# Parameters
Rs <- c(10, 100, 1000, 2000)
nrep <- 10
sample_orig <- c(1, 2, 3.5, 4, 7, 7.3, 8.6, 12.4, 13.8, 18.1)
n <- length(sample_orig)
q <- 0.4

for (R in Rs) {
  sd_trimmed_mean <- vector(, nrep)

  for (i in 1:nrep) {
    # Generate R bootstrap resamples of size n (n x R matrix)
    resample_bootstrap <- sample(sample_orig, n * R, rep = T)
    resample_bootstrap <- matrix(resample_bootstrap, nrow = n)

    # Sort the columns in ascending order
    resample_bootstrap <- apply(resample_bootstrap, 2, sort, decreasing = F)

    # Trimmed mean of each bootstrap resample
    trimmed_mean_bootstrap <- apply(resample_bootstrap, 2, mean, trim = q/2)

    # Bootstrap estimator of the standard deviation of the trimmed means
    sd_trimmed_mean[i] <- sd(trimmed_mean_bootstrap)
  }

  # Histogram and estimated density of the bootstrap estimates
  df <- data.frame(sd_trimmed_mean = sd_trimmed_mean)
  p <- ggplot(df, aes(x=sd_trimmed_mean)) +
    geom_histogram(aes(y =..density..),
                   bins = 30, fill = "#69b3a2", col = 'black') +
    geom_density(aes(y=..density..)) +
    ggtitle(paste("R =", R))
  print(p)

  # Mean and standard error of the nrep estimators
  cat("Mean of", nrep, "independent simulations ( R =", R, "):",
      mean(sd_trimmed_mean), "\n")
  cat("Standard error:", sd(sd_trimmed_mean)/sqrt(nrep), "\n")
}
