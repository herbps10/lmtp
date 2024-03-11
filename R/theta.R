theta_sub <- function(eta) {
  if (is.null(eta$weights)) {
    theta <- mean(eta$m[, 1])
  }

  if (!is.null(eta$weights)) {
    theta <- weighted.mean(eta$m[, 1], eta$weights)
  }

  if (eta$outcome_type == "continuous") {
    theta <- rescale_y_continuous(theta, eta$bounds)
  }

  out <- list(
    estimator = "substitution",
    theta = theta,
    standard_error = NA_real_,
    low = NA_real_,
    high = NA_real_,
    shift = eta$shift,
    outcome_reg = switch(
      eta$outcome_type,
      continuous = rescale_y_continuous(eta$m, eta$bounds),
      binomial = eta$m
    ),
    fits_m = eta$fits_m,
    outcome_type = eta$outcome_type
  )

  class(out) <- "lmtp"
  out
}

theta_ipw <- function(eta) {
  if (is.null(eta$weights)) {
    theta <- mean(eta$r[, eta$tau]*missing_outcome(eta$y))
  }

  if (!is.null(eta$weights)) {
    theta <- weighted.mean(
      eta$r[, eta$tau]*missing_outcome(eta$y),
      eta$weights
    )
  }

  out <- list(
    estimator = "IPW",
    theta = theta,
    standard_error = NA_real_,
    low = NA_real_,
    high = NA_real_,
    shift = eta$shift,
    density_ratios = eta$r,
    fits_r = eta$fits_r
  )

  class(out) <- "lmtp"
  out
}

eif <- function(r, tau, shifted, natural, cumulated, conditional_indicator) {
  cumulative_indicator <- as.logical(apply(conditional_indicator, 1, prod))

  indicator <- t(apply(conditional_indicator, 1, \(x) as.logical(rev(cumprod(x)))))

  natural[is.na(natural)] <- -999
  shifted[is.na(shifted)] <- -999
  m <- shifted[, 2:(tau + 1), drop = FALSE] - natural[, 1:tau, drop = FALSE]
  if(cumulated) {
    w <- r
  }
  else {
    w <- compute_weights(r, 1, tau)
  }

  # Need to include weights?
  theta <- mean(shifted[cumulative_indicator, 1])
  1 / mean(cumulative_indicator) * (rowSums(w * m , na.rm = TRUE) + cumulative_indicator * (shifted[, 1] - theta))
  #1 / mean(conditional_indicator) * conditional_indicator * (rowSums(w * m, na.rm = TRUE) + shifted[, 1] - theta)
}

theta_dr <- function(eta, augmented = FALSE) {
  cumulative_indicator <- as.logical(apply(eta$conditional_indicator, 1, prod))

  #eta$r[, 1] <- case_when(
  #  dat$A_1 == 0 ~ 0,
  #  dat$A_1 == 1 ~ 1,
  #  dat$A_1 == 2 ~ 2,
  #)
  #eta$r[, 2] <- eta$r[, 1] * case_when(
  #  eta$data$A_2 == 0 ~ 0,
  #  eta$data$A_2 == 1 ~ 1,
  #  eta$data$A_2 == 2 ~ 1,
  #)
  #eta$m$shifted[, 1] <- scale_y(case_when(
  #  eta$data$A_1 == 0 ~ 1,
  #  eta$data$A_1 == 1 ~ 2,
  #  eta$data$A_1 == 2 ~ 3,
  #  eta$data$A_1 == 3 ~ 3,
  #), eta$bounds)
  #eta$m$natural[, 1] <- scale_y(case_when(
  #  eta$data$A_1 == 0 ~ 0,
  #  eta$data$A_1 == 1 ~ 1,
  #  eta$data$A_1 == 2 ~ 2,
  #  eta$data$A_1 == 3 ~ 3,
  #), eta$bounds)
  inflnce <- eif(r = eta$r,
                 tau = eta$tau,
                 shifted = eta$m$shifted,
                 natural = eta$m$natural,
                 cumulated = eta$cumulated,
                 conditional_indicator = eta$conditional_indicator)

  theta <- {
    if (augmented)
      if (is.null(eta$weights))
        mean(inflnce)
      else
        weighted.mean(inflnce, eta$weights)
    else
      if (is.null(eta$weights))
        mean(eta$m$shifted[cumulative_indicator, 1])
      else
        weighted.mean(eta$m$shifted[cumulative_indicator, 1], eta$weights[cumulative_indicator])
  }

  if (eta$outcome_type == "continuous") {
    inflnce <- rescale_y_continuous(inflnce, eta$bounds)
    theta <- rescale_y_continuous(theta, eta$bounds)
  }

  clusters <- split(inflnce, eta$id)
  j <- length(clusters)
  se <- sqrt(var(vapply(clusters, function(x) mean(x), 1)) / j)
  ci_low  <- theta - (qnorm(0.975) * se)
  ci_high <- theta + (qnorm(0.975) * se)

  out <- list(
    estimator = eta$estimator,
    theta = theta,
    standard_error = se,
    low = ci_low,
    high = ci_high,
    eif = inflnce,
    id = eta$id,
    shift = eta$shift,
    outcome_reg_natural = switch(
      eta$outcome_type,
      continuous = rescale_y_continuous(eta$m$natural, eta$bounds),
      binomial = eta$m$natural
    ),
    outcome_reg = switch(
      eta$outcome_type,
      continuous = rescale_y_continuous(eta$m$shifted, eta$bounds),
      binomial = eta$m$shifted
    ),
    density_ratios = eta$r,
    fits_m = eta$fits_m,
    fits_r = eta$fits_r,
    outcome_type = eta$outcome_type
  )

  class(out) <- "lmtp"
  out
}
