run_ensemble <- function(data, y, learners, outcome_type, id, folds) {
  if(sd(data[[y]]) < 1e-5) {
    warning("Outcome has zero variance")
    learners <- c("mean")
  }
  fit <- mlr3superlearner(data = data,
                          target = y,
                          library = learners,
                          outcome_type = outcome_type,
                          folds = folds,
                          group = id)
  fit
}

SL_predict <- function(fit, newdata) {
  if (inherits(fit, "glm")) {
    return(as.vector(predict(fit, newdata, type = "response")))
  }
  predict(fit, newdata)
}
