rm(list = ls())

library(ElemStatLearn)
library(class)
library(AUC)

data(zip.train)
data(zip.test)

PrepareData = function(data) {
  df = as.data.frame(data)
  colnames(df) = c("digit", paste("pixel", as.character(1:256), sep = ""))
  # We are only concerned with 2s and 3s
  df = df[df$digit %in% c(2, 3), ]
  return(df)
}

ExceptY = function(df, Y.name = "digit") {
  return(df[, names(df) != Y.name])
}

train.data = PrepareData(zip.train)
train.X = ExceptY(train.data)
test.data = PrepareData(zip.test)
test.X = ExceptY(test.data)

results = data.frame(
  method = character(0),
  misclassification.rate.train = numeric(0),
  misclassification.rate.test = numeric(0)
)

threshold = function(prediction) {
  ifelse(prediction <= 2.5, 2, 3)
}

# Linear Regression
ols.model = lm(digit ~ ., train.data)
ols.predictions.train = threshold(predict(ols.model, train.data))
ols.predictions.test = threshold(predict(ols.model, test.data))
ols.results = data.frame(
  method = "OLS",
  misclassification.rate.train = mean(ols.predictions.train != train.data$digit),
  misclassification.rate.test = mean(ols.predictions.test != test.data$digit)
)
results = rbind(results, ols.results)

# kNN
Ks = c(1, 3, 5, 7, 15)
knn.results = lapply(Ks, function(k) {
  predictions.train = knn(train.X, train.X, factor(train.data$digit), k = k)
  misclassification.rate.train = mean(predictions.train != train.data$digit)

  predictions.test = knn(train.X, test.X, factor(train.data$digit), k = k)
  misclassification.rate.test = mean(predictions.test != test.data$digit)

  return(
    data.frame(
      method = paste(k, "NN", sep = ""),
      misclassification.rate.train = misclassification.rate.train,
      misclassification.rate.test = misclassification.rate.test
    )
  )
})

results = do.call(rbind, c(list(results), knn.results))
results = results[order(-results$misclassification.rate.test), ]
print(results)
