library(caret)

rm(list = ls())
setwd("~/Dropbox/Informatikk/STK4030/Statistical-learning/project")

set.seed(4030)

AllColumnsExcept = function(df, col.names) {
  return(df[, !(names(df) %in% col.names)])
}

load("bostonhousing.rdata")

training.data = data[data$train == TRUE, ]
training.data = AllColumnsExcept(training.data, 'train')

test.data = data[data$train == FALSE, ]
test.data = AllColumnsExcept(test.data, 'train')

# 1.1
# Estimate a linear Gaussian regression model including all 14 independent variables by (ordinary) least squares (OLS) on the training set.
lgr.model = lm(y ~ ., training.data)

# Report the estimated coefficients.
print(lgr.model$coefficients)

# Which covariates have the strongest association with y?
# In particular, the study focused on the effect of air pollution, measured through the concentrations of nitrogen oxide pollutants (nox) and particulate (part).

# Correlations
cor(training.data$y, training.data, method = "pearson")
# 1. lstat
# 2. rm
# 3. tax
# 4. indus
# 5. nox
# 6. crim

# Coefficient on standardized regression model
scaled.training.data = lapply(training.data, scale)
scaled.model = lm(y ~ ., scaled.training.data)
summary(scaled.model)
# 1. lstat
# 2. rad
# 3. dis
# 4. tax
# 5. crim
# 6. nox

# Incremental/partial R2.
# Gain in R2 when adding variable as the last one
# TODO

# Do they have any effect on the house price? If yes, which kind of effect?
# - part has a (statistically insignificant) negative effect on housing prices
# - nox has a (statistically significant) negative effect on housing prices

# 1.2
# The model above can be also used to predict the price for the other tracts (test set).

# RMSE
RMSE = function(model, data) {
  predictions = predict(model, data)
  prediction.error = sqrt(mean((predictions - data$y)^2))
  return(prediction.error)
}

# Compute the prediction error on the test data. 
lgr.prediction.error = RMSE(lgr.model, test.data)
print(lgr.prediction.error)
# 0.2147

# Moreover, derive two reduced models by applying a
# backward elimination procedure with
# AIC and α = 0.05 as stopping criteria, respectively.
# For both models, report the estimated coefficients and
# the prediction error estimated on the test data.
# Comment the results.

# Stopping criteria: AIC
# NOTE: Use CV?
aic.model = lm(y ~ ., training.data)
aic.model = step(aic.model, direction = "backward")
print(aic.model$coefficients)
aic.prediction.error = RMSE(aic.model, test.data)
print(aic.prediction.error)
# 0.2143025

# Stopping criteria: α = 0.05
PValues = function(model) {
  return(summary(model)$coefficients[-1, 4])
}

alpha.data = training.data
repeat {
  alpha.model = lm(y ~ ., alpha.data)

  if (all(PValues(alpha.model) <= 0.05)) {
    break;
  }
  
  p.values = PValues(alpha.model)
  highest.p.value = p.values[which.max(p.values)]
  predictor.with.highest.p.value = names(highest.p.value)
  alpha.data = AllColumnsExcept(alpha.data, predictor.with.highest.p.value) 
}
print(alpha.model$coefficients)
alpha.prediction.error = RMSE(alpha.model, test.data)
print(alpha.prediction.error)
# 0.2144203

# 1.3
# Estimate a principal component regression model,
# selecting the number of components by 10-fold cross-validation.

# oneSE is a rule in the spirit of the "one standard error" rule of Breiman et al. (1984),
# who suggest that the tuning parameter associated with the best performance may over fit.
# They suggest that the simplest model within one standard error of the empirically optimal model is the better choice.

RotateData = function(pca, data, response.variable = 'y') {
  rotated.data = as.data.frame(predict(pca, data))
  rotated.data[response.variable] = data[response.variable]
  return(rotated.data)
}

pca = prcomp(AllColumnsExcept(training.data, 'y'), scale = TRUE, center = TRUE)

rotated.training.data = RotateData(pca, training.data)
rotated.test.data = RotateData(pca, test.data)

SelectBestPCRModel = function(pca, rotated.data, method, number) {
  best.model = NULL
  principal.components = as.data.frame(pca$x)

  for (i in 1:ncol(principal.components)) {
    predictors = names(principal.components)[1:i]
    formula = as.formula(paste("y ~ ", paste(predictors, collapse = "+")))
    
    candidate.model = train(
      formula,
      data = rotated.data,
      method = "lm",
      trControl = trainControl(
        method = method,
        number = number
      ))
    
    print(paste(i, 'components:', candidate.model$results$RMSE))
    
    # TODO: Use 1 STD rule
    candidate.model.is.better =
      is.null(best.model) ||
      candidate.model$results$RMSE < best.model$results$RMSE
    
    if (candidate.model.is.better) {
      best.model = candidate.model
    }
  }
  
  best.model$num.of.components = length(best.model$coefnames)

  return(best.model)
}mac

pcr.cv.model = SelectBestPCRModel(pca, rotated.training.data, 'cv', 10)
# [1] "1 components: 0.276600019113419"
# [1] "2 components: 0.270699193508081"
# [1] "3 components: 0.243359850423786"
# [1] "4 components: 0.240114174863371"
# [1] "5 components: 0.20910206446859"
# [1] "6 components: 0.212961471976449"
# [1] "7 components: 0.211454445813939"
# [1] "8 components: 0.203513057434645"
# [1] "9 components: 0.192610688513614"
# [1] "10 components: 0.191086539345536"
# [1] "11 components: 0.18157022262457"
# [1] "12 components: 0.182831708817368"
# [1] "13 components: 0.174977003404454"
# [1] "14 components: 0.175404382517462"

pcr.cv.prediction.error = RMSE(pcr.cv.model, rotated.test.data)
print(pcr.cv.prediction.error)

# How many components have been selected?
print(pcr.cv.model$num.of.components)
# 13

# What does it mean?
# TODO

# 1.4
# Repeat the procedure to choose the number of components by using the .632 bootstrap procedure.

pcr.bootstrap.model = SelectBestPCRModel(pca, rotated.training.data, 'boot632', 100)
# [1] "1 components: 0.286262598775084"
# [1] "2 components: 0.275033522526662"
# [1] "3 components: 0.251944753705648"
# [1] "4 components: 0.241849708105148"
# [1] "5 components: 0.217159239616464"
# [1] "6 components: 0.217080107073819"
# [1] "7 components: 0.216854459061916"
# [1] "8 components: 0.209204225935823"
# [1] "9 components: 0.195589879656002"
# [1] "10 components: 0.198151792331573"
# [1] "11 components: 0.186556514210195"
# [1] "12 components: 0.18751841723099"
# [1] "13 components: 0.181274575543526"
# [1] "14 components: 0.178034097303653"

pcr.bootstrap.prediction.error = RMSE(pcr.bootstrap.model, rotated.test.data)
print(pcr.bootstrap.model$num.of.components)
print(pcr.bootstrap.prediction.error)

# Does the number of selected components change?
# Yes (but depending on seed)

# Report the estimate of the prediction error for each possible number of components.
# See above

# 1.5
# Estimate the regression model by ridge regression,
# where the optimal tuning parameter λ is chosen by 10-fold cross-validation.

# TODO: What are reasonable lambda values?
ridge.model = train(
  y ~ .,
  data = training.data,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = 0,
    lambda = 10^seq(-5, 5, by = .1)
  ),
  trControl = trainControl(
    method = 'cv',
    number = 10
  )
)

# Report the estimated coefficients
print(
  coef(ridge.model$finalModel, ridge.model$bestTune$lambda)
)
# (Intercept)  3.8771124229
# crim        -0.0080244317
# zn           0.0008041282
# indus       -0.0002841339
# chas         0.1359930995
# nox         -0.3888605767
# rm           0.0965529950
# age         -0.0004318776
# dis         -0.0441903838
# rad          0.0061303460
# tax         -0.0002416576
# ptratio     -0.0298469164
# bk          -0.3716559594
# lstat       -0.0309399065
# part        -0.0080482342

# the obtained value of lambda and
print(ridge.model$bestTune$lambda)
# 0.03162278

# the prediction error computed on the test data.
ridge.prediction.error = RMSE(ridge.model, test.data)
print(ridge.prediction.error)
# 0.2095294

# 1.6
# Repeat the same procedure by using lasso and component-wise L2Boost.
# Use 10-fold cross-validation to find the optimal value for λ (lasso) and mstop (L2Boost),
# while set the boosting step size ν equal to 0.1.

# Lasso
lasso.model = train(
  y ~ .,
  data = training.data,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = 10^seq(-5, 5, by = .1)
  ),
  trControl = trainControl(
    method = 'cv',
    number = 10
  )
)

# Coefficients
print(coef(lasso.model$finalModel, lasso.model$bestTune$lambda))
# (Intercept)  4.3451841588
# crim        -0.0089773685
# zn           0.0012940837
# indus        0.0023592275
# chas         0.1256043914
# nox         -0.6013660851
# rm           0.0713155344
# age          .           
# dis         -0.0595664564
# rad          0.0125352071
# tax         -0.0005012206
# ptratio     -0.0343277653
# bk          -0.3905806133
# lstat       -0.0363025232
# part        -0.0080993993

print(lasso.model$bestTune$lambda)
# 0.0007943282

lasso.prediction.error = RMSE(lasso.model, test.data)
print(lasso.prediction.error)
# 0.2144327

# L2boost
# TODO

# 1.7
# It has been argued that the predictors rm and dis do not have a linear effect on the outcome.
# Substitute the former with its cube and the latter with its inverse (dis-1) in the first model (OLS) and refit the model.
TransformData = function(data) {
  transformed.data = data
  transformed.data$rm.squared = data$rm^2
  transformed.data$dis.inverse = data$dis^(-1)
  transformed.data = AllColumnsExcept(transformed.data, c('rm', 'dis'))
}

transformed.training.data = TransformData(training.data)
transformed.test.data = TransformData(test.data)

transformed.model = lm(y ~ ., transformed.training.data)

# Report the estimated coefficients.
print(transformed.model$coefficients)

# Compute the prediction error on the test set and compare the result with that obtained at point 1.
transformed.prediction.error = RMSE(transformed.model, transformed.test.data)
print(transformed.prediction.error)
# 0.2120228
print(lgr.prediction.error)
# 0.2147611

# The fit improved!
# TODO: Comment/test significance of difference

