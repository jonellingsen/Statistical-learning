rm(list = ls())

# Load libraries ----------------------------------------------------------
library(ElemStatLearn)

# Load and prepare data set -----------------------------------------------

data("spam")

spam.column.index <- which(colnames(spam) == "spam")
spam[, -spam.column.index] <- scale(spam[, -spam.column.index])
spam$spam <- ifelse(spam$spam == "spam", 1, 0)
x <- as.matrix(spam[, -spam.column.index])
y <- spam$spam



# Split into training and test set ----------------------------------------

share.training <- 0.75

set.seed(123)
train.index <- sample(1:nrow(x), round(nrow(x)*share.training))
test.index <- -train.index



# Least squares regression ------------------------------------------------

train.ls <- lm(spam ~ ., data = spam[train.index, ])
ls.prediction <- predict.lm(train.ls, spam[test.index, ])
ls.coefficients <- train.ls$coefficients
ls.misclassification.rate <- mean(y[test.index] != round(ls.prediction))



# Ridge regression --------------------------------------------------------

library(glmnet)

lambda.vector <- 10^seq(10, -2, length = 100)

train.ridge <- glmnet(x[train.index, ], y[train.index], alpha = 0, lambda = lambda.vector)
train.ridge.cv <- cv.glmnet(x[train.index, ], y[train.index], alpha = 0)
plot(train.ridge.cv)
cv.ridge.lambda <- train.ridge.cv$lambda.1se
ridge.prediction <- predict(train.ridge, s = cv.ridge.lambda, newx = x[test.index, ])
ridge.coefficients <- as.vector(coef(train.ridge.cv, s = "lambda.1se"))
ridge.misclassification.rate <- mean(y[test.index] != round(ridge.prediction))



# Lasso regression --------------------------------------------------------

lambda.vector <- 10^seq(10, -2, length = 100)

train.lasso <- glmnet(x[train.index, ], y[train.index], alpha = 1, lambda = lambda.vector)
train.lasso.cv <- cv.glmnet(x[train.index, ], y[train.index], alpha = 1)
plot(train.lasso.cv)
cv.lasso.lambda <- train.lasso.cv$lambda.1se
lasso.prediction <- predict(train.lasso, s = cv.lasso.lambda, newx = x[test.index, ])
lasso.coefficients <- as.vector(coef(train.lasso.cv, s = "lambda.1se"))
lasso.misclassification.rate <- mean(y[test.index] != round(lasso.prediction))


# Principal components regression -----------------------------------------

library(pls)
spam.with.intercept <- cbind(rep(1, nrow(spam)), spam)

train.pcr <- pcr(spam ~ ., data = spam.with.intercept[train.index, ], validation = "CV")
cv.pcr <- selectNcomp(train.pcr, method = "onesigma", plot = TRUE)
pcr.prediction <- predict(train.pcr, newdata = spam.with.intercept[test.index, ], ncomp = cv.pcr)
pcr.coefficients <- as.vector(train.pcr$coefficients[, , cv.pcr])
pcr.misclassification.rate <- mean(y[test.index] != round(pcr.prediction))

# Partial least squares regression ----------------------------------------

spam.with.intercept <- cbind(rep(1, nrow(spam)), spam)
train.plsr <- plsr(spam ~ ., data = spam.with.intercept[train.index, ], validation = "CV")
cv.plsr <- selectNcomp(train.plsr, method = "onesigma", plot = TRUE)
plsr.prediction <- predict(train.plsr, newdata = spam.with.intercept[test.index, ], ncomp = cv.plsr)
plsr.coefficients <- as.vector(train.plsr$coefficients[, , cv.plsr])
plsr.misclassification.rate <- mean(y[test.index] != round(plsr.prediction))


# Create summary table ----------------------------------------------------

merged.coefficients <- cbind(ls.coefficients,
                             ridge.coefficients,
                             lasso.coefficients,
                             pcr.coefficients,
                             plsr.coefficients)

merged.test.errors <- cbind(ls.misclassification.rate,
                            ridge.misclassification.rate,
                            lasso.misclassification.rate,
                            pcr.misclassification.rate,
                            plsr.misclassification.rate)

table <- rbind(merged.coefficients, merged.test.errors)
rownames(table)[nrow(table)] <- "Test error"
