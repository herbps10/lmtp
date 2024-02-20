cf_rr <- function(Task, lr, epochs, hidden, dropout, pb) {
  fopts <- options("lmtp.bound", "lmtp.trt.length")
  out <- list()

  for (fold in seq_along(Task$folds)) {
    out[[fold]] <- future::future({
      options(fopts)

      estimate_rr(
        get_folded_data(Task$natural, Task$folds, fold),
        get_folded_data(Task$shifted, Task$folds, fold),
        Task$trt, Task$cens, Task$risk, Task$tau, Task$node_list$trt,
        lr, epochs, hidden, dropout, pb
      )
    },
    seed = TRUE)
  }

  recombine_ratios(future::value(out), Task$folds)
  #recombine_ratios(out, Task$folds)
}

estimate_representer <- function(natural, shifted, natural_valid, shifted_valid, lr, epochs, hidden, dropout) {
  d_in <- ncol(natural)
  d_out <- 1

  natural <- torch::torch_tensor(as.matrix(natural), dtype = torch::torch_float())
  shifted <- torch::torch_tensor(as.matrix(shifted), dtype = torch::torch_float())

  natural_valid <- torch::torch_tensor(as.matrix(natural_valid), dtype = torch::torch_float())
  shifted_valid <- torch::torch_tensor(as.matrix(shifted_valid), dtype = torch::torch_float())

  riesz <- torch::nn_sequential(
    torch::nn_linear(d_in, hidden),
    torch::nn_elu(),
    torch::nn_linear(hidden, hidden),
    torch::nn_elu(),
    torch::nn_dropout(dropout),
    torch::nn_linear(hidden, d_out),
    torch::nn_softplus()
  )

  Map(\(x) torch::nn_init_normal_(x, 0, 0.1), riesz$parameters)

  learner <- function(x) {
    riesz(x)[,1]
  }

  optimizer <- torch::optim_adam(params = c(
    riesz$parameters
  ), lr = lr, weight_decay = 0)

  scheduler <- torch::lr_one_cycle(optimizer, max_lr = lr, total_steps = epochs)
  for(epoch in 1:epochs) {
    rr <- learner(natural)
    rr_shifted <- learner(shifted)

    # Regression loss
    loss <- (rr$pow(2) - 2 * rr_shifted)$sum()

    if(epoch %% 20 == 0) {
      cat("Epoch: ", epoch, " Loss: ", loss$item(), "\n")
    }

    optimizer$zero_grad()
    loss$backward()

    optimizer$step()
    scheduler$step()
  }

  riesz$eval()

  list(
    rr = torch::as_array(learner(natural_valid)),
    rr_shifted = torch::as_array(learner(shifted_valid))
  )
}

estimate_rr <- function(natural, shifted, trt, cens, risk, tau, node_list, lr, epochs, hidden, dropout, pb) {
  representers <- matrix(nrow = nrow(natural$valid), ncol = tau)
  fits <- list()

  for (t in 1:tau) {
    jrt <- censored(natural$train, cens, t)$j
    drt <- at_risk(natural$train, risk, t)
    irv <- censored(natural$valid, cens, t)$i
    jrv <- censored(natural$valid, cens, t)$j
    drv <- at_risk(natural$valid, risk, t)

    trt_t <- ifelse(length(trt) > 1, trt[t], trt)

    vars <- c(node_list[[t]], cens[[t]])

    cat("t = ", t, "\n")
    rr <- estimate_representer(
      natural$train[jrt & drt, vars],
      shifted$train[jrt & drt, vars],
      natural$valid[jrv & drv, vars],
      shifted$valid[jrv & drv, vars],
      lr,
      epochs,
      hidden,
      dropout
    )

    representers[jrv & drv, t] <- rr$rr

    pb()
  }

  list(ratios = representers)
}
