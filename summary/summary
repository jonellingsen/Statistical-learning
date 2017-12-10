## Chapter 2: Overview of Supervised Learning

#### 2.2: Variable Types and Terminology
**Regression**: predicting quantitative outputs (Y)

**Classification**: predicting qualitative outputs (G)

#### 2.3: Two Simple Approaches to Prediction: Least Squares and Nearest Neighbors
##### Least squares
Low variance and potenitally high bias.
$$ RSS(\beta) = \sum_{i = 1}^N (y_i - x_i^T \beta)^2 $$
$$ RSS(\beta) = (\mathbf{y} - \mathbf{X}\beta)^T (\mathbf{y} - \mathbf{X}\beta) $$

Differentiate w.r.t. $\beta$ and get:
$$ \mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\beta}) = 0 $$
$$ \hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

##### Nearest-neighbor
High variance and low bias.

Idea: use the observations in the training set closest in input space to $x$ to form $\hat{Y}$. The k-nearest neighbor fit for $\hat{Y}$ is:
$$ \hat{Y}(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i $$
Closeness is measures in euclidean distance. Hence, we find the nearest neighbors and average their responses. The error of the traning data increases in k, and will always be 0 for k = 1. The number of parameters are N/k. Why? There will be N/k neighborhoods, and we fit one parameter (the mean) to each of them.

#### 2.4: Statistical Decision Theory
Most used loss function is the squared error loss, $L_2$:
$$ L(Y, f(X)) = (Y - f(X))^2 $$
According to this we get the following criterion for choosing the prediction function, f:
$$ EPE(f) = E(Y - f(X))^2 = E_X E_{Y|X} ([Y - f(X)]^2 | X) $$
It is sufficient to minimize EPE pointwise:
$$ f(x) = argmin_c E_{Y|X} ([Y - c]^2 | X = x) $$
The solution is:
$$ f(x) = E((Y|X = x) $$
This is the best solution given a squared error loss function. The solution is simply the regression function, i.e. the conditional mean, conditioning on a specific point, x. The nearest neighbor method is trying to achieve this, but with two approximations. First, expectation is approximated by sample average. Second, the conditioning on one specific point is relaxed to conditioning on some region "close" to the target point.

Difference in model assumptions of least squares and k-nearest neighbors:

* Least squares assumes $f(x)$ is well approximated by a **globally linear** function.
* k-nearest neighbors assumes $f(x)$ is well approximated by a **locally constant** function. Near x_i the approximating function does not change, so we average over the neighbors with the same function.

#### 2.5: Local Methods in High Dimensions
##### Curse of dimensionality
When the dimensions increase, the neighborhood is no longer local, since we need to collect a larger range of the inputs in order to capture the same fraction of data.

##### Bias-variance decomposition
Say we want to predict $f(x_0) = y_0$ at the test point $x_0$. Denote the traning set by $\mathcal{T}$. 
$$ MSE(x_0) = E_{\mathcal{T}} \big(\hat{y}_0 - f(x_0)\big)^2$$
$$ = E_{\mathcal{T}} \big((\hat{y}_0 - E_{\mathcal{T}}(\hat{y}_0)) + (E_{\mathcal{T}}(\hat{y}_0) - f(x_0))\big)^2 $$
$$ = E_{\mathcal{T}} \big((\hat{y}_0 - E_{\mathcal{T}}(\hat{y}_0))^2\big) + 2\big(E_{\mathcal{T}}(\hat{y}_0) - E_{\mathcal{T}}(\hat{y}_0)\big)\big(E_{\mathcal{T}}(\hat{y}_0) - f(x_0)\big) + E_{\mathcal{T}}\big(E_{\mathcal{T}}(\hat{y}_0) - f(x_0)\big)^2 $$
$$ = E_{\mathcal{T}} \big((\hat{y}_0 - E_{\mathcal{T}}(\hat{y}_0))^2\big) + E_{\mathcal{T}}\big(E_{\mathcal{T}}(\hat{y}_0) - f(x_0)\big)^2 $$
$$ = Var_{\mathcal{T}} \big(\hat{y}_0\big) + Bias^2\big(\hat{y}_0\big)$$


#### 2.6: Statistical Models, Supervised Learning and Function Approximation

#### 2.7: Structured Regression Models
The RSS criterion for an arbitrary function, f:
$$ RSS(f) = \sum_{i = 1}^N (y_i - f(x_i))^2 $$
The minimization problem of RSS has infinitely many solutions: any function that passes through the training points $(x_i, y_i)$ is a solution. In order to obtain useful results for finite N, we restrict the solutions of the minimization problem to a smaller set of functions. However, this is a choice we make, and it does not remove the ambiguity caused by the multiplicity of solutions.

#### 2.8: Classes of Restricted Estimators

#### 2.9: Model Selection and the Bias–Variance Tradeoff
**Overfitting**: When we increase the complexity of the model it adapts more closely to the training data and may generalize poorly out-of-sample. Hence, the reduction in bias is met by an increase in the variance.


## Chapter 3: Linear Methods for Regression
#### 3.1: Introduction

#### 3.2: Linear Regression Models and Least Squares
**t-test**:
$$z_j = \frac{\hat{\beta}_j}{\hat{\sigma} \sqrt{v_j}}$$ where the denominator represents the standard errors of $y$ and $x_j$ is distributed as $t_{N - p - 1}$ under the null that $\beta_j = 0$.

**F-test**:
$$F = \frac{(RSS_0 - RSS_1) / (p_1 - p_0)}{RSS_1 / (N - p_1 - 1)}$$ where $RSS_1$ is the residual sum-of-squares of the bigger model with $p_1 + 1$ parameters, and $RSS_0$ the same for the nested smaller model with $p_0 + 1$ parameters, having $p_1 - p_0$ parameters constrained to be zero. The F-statistic measures the change in residual sum-of-squares per additional parameter in the bigger model, and is normalized by an estimate of $\sigma^2$. Under the Gaussian assumptions, and the null hypothesis that the smaller model is correct, the F-statistic wll have a $F_{p_1 - p_0, \ N - p_1 - 1}$ distribution.

**Gauss-Markov theorem**: The least squares estimates of the parameters $\beta$ have the smallest possible variance among all linear unbiased estimates (BLUE).

#### 3.3: Subset Selection
*Subset selection*: we retain only a subset of the variables, and eliminate the rest from the model.
Two reasons why least squares are often not satisfying:

* *Predction accuracy*: Least squares often have low bias but large variance. By shrinkage we can sometimes improve prediction by sacrificing some bias to reduce the variance of the predicted values, and hence improve the prediction accuracy.
* *Interpretation*: Willing to sacrifice some of the small details (many coefficients) in order to get the "big picture". Hence we would like a smaller subset that exhibits the strongest effects.

##### 3.3.1 Best-subset selection
Best-subset regression finds for each $k \in \{0,1,2,\ldots,p\}$ the subset of size $k$ that gives the smallest residual sum of squares. An efficient algorithm is leaps and bounds. The subsets need not include the same variables. Infeasible for $p > 40$ or $p > N$.

##### 3.3.2 Forward- and backward-stepwise selection
Seek a good path through the subsets. Forward-stepwise selection starts with intercept and sequentially adds into the predictor that most improves the fit. Updates coefficients when a new variable enters. Greedy algorithm since step $k$ includes models up to $k-1$. Feasible for $p > N$.
Backward-stepwise selection starts with the full model, and sequetially deletes the predictor that has the least impact on the fit (the variable with lowest z-score or AIC). Infeasible for $p > N$.

##### 3.3.2 Forward-stagewise regression
Finds the variable most correlated with residual, and computes the least square coefficient. Then, goes on to the next and do the same. Stops when no variable have correlation with the residuals. Does not update coefficients when a new variable enters. Approaches the least squares fit when $N > p$.

#### 3.4: Shrinkage Methods
Shrinkage methods are more continous than subset selection and hence may reduce the variance.

##### 3.4.1 Ridge regression
Shrinks the regression coefficients by imposing a L2 penalty on their size. Important to standarize the variables.
$$ \hat{\beta}^{ridge} =  \underset{\beta}{\operatorname{argmin}} \Bigg\{\sum_{i = 1}^N \Big(y_i - \beta_0 - \sum_{j = 1}^p x_{i,\ j}\beta_j\Big)^2 + \lambda \sum_{j = 1}^p \beta_j^2\Bigg\} $$
or:
$$ \hat{\beta}^{lasso} =  \underset{\beta}{\operatorname{argmin}} \Bigg\{\sum_{i = 1}^N \Big(y_i - \beta_0 - \sum_{j = 1}^p x_{i,\ j}\beta_j\Big)^2\Bigg\} \quad \text{s.t. }\sum_{j = 1}^p \beta_j^2 \leq t $$
Often, highly correlated variables offset each other, but ridge regression alleviates this. Coefficients are shrunk towards 0 and each other. In matrix form (when all variables are centered so that no intercept is included):
$$ \text{RSS}(\lambda) = (\mathbf{y} - \mathbf{X}\beta)^T (\mathbf{y} - \mathbf{X}\beta) + \lambda \beta^T \beta $$
$$\hat{\beta}^{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda I)^{-1} \mathbf{X}^T\mathbf{y} $$
Observe that the solutions is a linear function of y.

Ridge regression shrinks the directions in the column space of $\mathbf{X}$ having small variance.

##### 3.4.2 Lasso
Shrinks the regression coefficients by imposing a L1 penalty on their size.
$$ \hat{\beta}^{lasso} =  \underset{\beta}{\operatorname{argmin}} \Bigg\{\sum_{i = 1}^N \Big(y_i - \beta_0 - \sum_{j = 1}^p x_{i,\ j}\beta_j\Big)^2 + \lambda \sum_{j = 1}^p |\beta_j|\Bigg\} $$
or:
$$ \hat{\beta}^{lasso} =  \underset{\beta}{\operatorname{argmin}} \Bigg\{\sum_{i = 1}^N \Big(y_i - \beta_0 - \sum_{j = 1}^p x_{i,\ j}\beta_j\Big)^2\Bigg\} \quad \text{s.t. }\sum_{j = 1}^p |\beta_j| \leq t $$

The lasso solutions are nonlinear in the $y_i$, and there is no closed form expression as in ridge regression. The lasso does a kind of continuous subset selection by setting some coefficients equal to zero for a suffieciently high $\lambda$.

We can denote the shrinkage factor by:
$$s = \frac{t}{\sum_{j = 1}^p |\beta_j|} $$
Hence, we see that $s = 1$ means no shrinkage an thus gives the least squares solution.

##### 3.4.3 Elastic-net
Combination of penalty from ridge regression and lasso:
$$\lambda \sum_{i = 1}^p \big(\alpha \beta_j^2 + (1 - \alpha) |\beta_j|\big)$$
The elastic-net selects variables like the lasso (shrinks some coefficients to exactly zero), and shrinks together the coefficients of correlated predictors like ridge.


#### 3.5: Methods Using Derived Input Directions
In many situations we have a large number of inputs, often very correlated. The methods in this section produce a small number of linear combinations $Z_m, \ m = 1,\ldots,M$ of the original inputs $X_j$, and the $Z_m$ are then used in place of the $X_j$ as inputs in the regression.

##### 3.5.1 Principal components regression
First, a little note on how to derive principal components. The SVD of the $N \times p$ matrix $\mathbf{X}$ has the form:
$$\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^T$$
Here $\mathbf{U}$ and $\mathbf{V}$ are $N \times p$ and $p \times p$ orthogonal matrices, with the columns of $\mathbf{U}$ spanning the column space of $\mathbf{X}$, and the columns of $\mathbf{V}$ spanning the row space. $\mathbf{D}$ is a $p \times p$ diagonal matrix, with diagonal entries $d_1 \geq d_2 \geq \cdots \geq d_p \geq 0$ called the singular values of $\mathbf{X}$.

Further, we can write the eigen decomposition of $\mathbf{X}^T\mathbf{X}$ as:
$$\mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{D}^2\mathbf{V}^T$$
The eigenvectors $v_j$ (the columns of $\mathbf{V}$) are called the principal component directions of $\mathbf{X}$. The first principal component has the property that $\mathbf{z_1} = \mathbf{X}v_1$ has the largest sample variance amongst all linear combinations of the columns of $\mathbf{X}$. All the $\mathbf{z_i}$ are orthogonal.

Principal components regression simply estimates a least squares model on the derived inputs such that:
$$\hat{y}_{M}^{pcr} = \bar{y}\mathbf{1} + \sum_{m = 1}^M \hat{\theta}_m \mathbf{z_m}$$

$M = p$ gives the least squares solution.

##### 3.5.1 Partial least squares
![](pls.png)


#### 3.6: Discussion: A Comparison of the Selection and Shrinkage Methods
Ridge regression, PCR and PLS tend to behave similarly, but ridge regression may be preferred because it shrinks the coefficients smoothly rather than in discrete steps. Lasso falls somewhere between ridge regression and best subset regression, and enjoys some of the properties of each.

#### 3.8: More on the Lasso and Related Path Algorithms
#### 3.9: Computational Considerations


## Chapter 4: Linear Methods for Classification
#### 4.1: Introduction
In classification, the predictor, $G(x)$ takes values in a discrete set, $\mathcal{G}$. Hence, we can always divide our input space into a collection of regions labeled according to the classification. The boundaries between these regions can be rough or smooth, and in some cases linear. Linear methods in the context of classification refers to the decision boundaries being linear in the input space.

#### 4.2: Linear Regression of an Indicator Matrix
![](masking.png)

#### 4.3: Linear Discriminant Analysis (MISSING)
Decision theory for classification (Section 2.4) tells us that we need to know the class posteriors $Pr(G|X)$ for optimal classification. Suppose $f_k(x) = Pr(X = x|G = k)$, and let $\pi_k = Pr(G = k)$, with $\sum_{k = 1}^K \pi_k = 1$. A simple application of Bayes theorem gives us:
$$ Pr(G = k|X = x) = \frac{f_k(x) \pi_k}{\sum_{l = 1}^K f_l(x) \pi_l} $$
Discrimininant analysis model each class density, $f_k(x)$, as multivariate Gaussian:
$$ f_k(x) = \frac{1}{(2 \pi)^{p/2} |\Sigma_k|^{1/2}}e^{- \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)} $$
Linear discriminant analysis (LDA) arises in the special case when we assume that the classes have a common covariance matrix $\sum_k = \sum \forall k$. In comparing two classes $k$ and $l$, it is sufficient to look at the log-ratio, and we see that:
$$ log \frac{Pr(G = k|X = x)}{Pr(G = l|X = x)} = log \frac{f_k(x)}{f_l(x)} + log \frac{\pi_k}{\pi_l} $$
$$ = log \frac{\pi_k}{pi_l} - \frac{1}{2} (\mu_k + \mu_l)^T \Sigma^{-1} (\mu_k - \mu_l) + x^T \Sigma^{-1} (\mu_k - \mu_l) $$
The discriminant function for class $k$:
$$ \delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + log \pi_k $$
The parameters are estimated from the training set:

* $\pi_k = N_k / N$, where $N_k$ is the number of class-k observations
* $\hat{\mu}_k = \sum_{g_i = k} x_i / N_k$
* $\hat{\Sigma} = \sum_{k = 1}^K \sum_{g_i = k} (x_i - \hat{\mu}_k) (x_i - \hat{\mu}_k)^T / (N - K)$

Rule: classify an observation to the class that maximizes the discriminant function.


#### 4.4: Logistic Regression
The logistic regression model arises from the desire to model the posterior probabilities of the K classes via linear functions in x, while at the same time ensuring that they sum to one and remain in [0,1]. Logistic regression models are used mostly as a data analysis and infer- ence tool, where the goal is to understand the role of the input variables in explaining the outcome. The model has the form
![](logistic_reg.png)
Since the probabilities sum to one, we have
![](logistic_reg2.png)

Interpretation of output from logistic regression:
![](logistic_regression_output.png)
One unit increase in tobacco consumption is associated with an increase in the odds of coronary heart disease of $exp(0.081) -1 = 8.4 \%$. I.e. $\frac{p}{1-p}$ increases with $8.4 \%$.



## Chapter 7: Model Assessment and Selection
#### 7.1: Introduction

#### 7.2: Bias, Variance and Model Complexity
![](bias_variance.png)
![](model_selection_assessment.png)
If we are in a data-rich situation, the best approach for both problems is to randomly divide the dataset into three parts: a training set, a validation set, and a test set. The training set is used to fit the models; the validation set is used to estimate prediction error for model selection; the test set is used for assessment of the generalization error of the final chosen model. Ideally, the test set should be kept in a “vault,” and be brought out only at the end of the data analysis.

#### 7.3: The Bias–Variance Decomposition
Let $Y = f(X) + \varepsilon$ where $E(\varepsilon) = 0$ and $Var(\epsilon) = \sigma_{\varepsilon}^2$. The expected prediction error of a regression fit $\hat{f}(X)$ at an input point $X = x_0$, using squared-error loss is:
$$ Err(x_0) = E \Big[ \big(Y - \hat{f}(x_0)\big)^2 \Big] = \sigma^2 + Bias^2(\hat{f}(x_0)) + Var(\hat{f}(x_0)) $$
$$ = \text{Irreducible error} + Bias^2 + Variance $$
![](bias_variance_proof.png)

#### 7.4: Optimism of the Training Error Rate
Given a training set $\mathcal{T} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_N , y_N )\}$ the *generalization error* of a model $\hat{f}$ is
$$\text{ERR}_{\mathcal{T}} = E_{X^0, Y^0}\big[L(Y^0, \hat{f}(X^0))| \mathcal{T}\big]$$
Note that the training set $\mathcal{T}$ is fixed above. The point $(X^0, Y^0)$ is a new test data point, drawn from $F$, the joint distribution of the data. Averaging over training sets $\mathcal{T}$ yields the expected error
$$\text{ERR} = E_{\mathcal{T}}E_{X^0, Y^0}\big[L(Y^0, \hat{f}(X^0))| \mathcal{T}\big]$$
which is more amenable to statistical analysis.

Now typically, the training error
$$\overline{\text{err}} = \frac{1}{N} \sum_{i = 1}^N L(y_i, \hat{f}(x_i))$$
will be less than the true error $\text{Err}_{\mathcal{T}}$ , because the same data is being used to fit the method and assess its error. A fitting method typically adapts to the training data, and hence the training error $\overline{\text{err}}$ will be an overly optimistic estimate of the generalization error $\text{Err}_{\mathcal{T}}$.

Part of the discrepancy is due to where the evaluation points occur. The quantity $\text{Err}_{\mathcal{T}}$ can be thought of as extra-sample error, since the test input vectors don’t need to coincide with the training input vectors (we can observe different $x$'s). The nature of the optimism in $\overline{\text{err}}$ is easiest to understand when we focus instead on the in-sample error
$$\text{ERR}_{\text{in}} = \frac{1}{N} \sum_{i = 1}^N E_{Y^0}\big[L(Y_i^0, \hat{f}(x_i))| \mathcal{T}\big]$$
The $Y^0$ notation indicates that we observe $N$ new response values at each of the training points $x_i, i = 1,2,\ldots,N$. We define the optimism as the difference between $\text{ERR}_{\text{in}}$ and the training error $\overline{\text{err}}$:
$$\text{op} \equiv \text{Err}_{\text{in}} - \overline{\text{err}}$$
This is typically positive since $\overline{\text{err}}$ is usually biased downward as an estimate of prediction error. Finally, the average optimism is the expectation of the optimism over training sets
$$\omega \equiv E_y(\text{op})$$
Here the predictors in the training set are fixed, and the expectation is over the training set outcome values; hence we have used the notation $E_y$ instead of $E_{\mathcal{T}}$ . We can usually estimate only the expected error $\omega$ rather than $\text{op}$, in the same way that we can estimate the expected error $\text{Err}$ rather than the conditional error $\text{Err}_{\mathcal{T}}$.
For squared error, 0–1, and other loss functions, one can show quite generally that
$$\omega = \frac{2}{N} \sum_{i = 1}^N Cov(\hat{y}_i, y_i)$$
Thus the amount by which $\overline{\text{err}}$ underestimates the true error depends on how strongly $y_i$ affects its own prediction. The harder we fit the data, the greater $Cov(\hat{y}_i, y_i)$ will be, thereby increasing the optimism.
In summary, we have the important relation
$$E_Y(\text{Err}_{\text{in}}) = E_y(\overline{\text{err}}) + \frac{2}{N} \sum_{i = 1}^{N} Cov(\hat{y}_i, y_i)$$
This expression simplifies if $\hat{y}_i$ is obtained by a linear fit with $d$ inputs or basis functions. For example,
$$\sum_{i = 1}^{N} Cov(\hat{y}_i, y_i) = d \sigma_{\varepsilon}^2$$
for the additive error model $Y = f(X) + \varepsilon$, and so
$$E_Y(\text{Err}_{\text{in}}) = E_y(\overline{\text{err}}) + 2\frac{d}{N}\sigma_{\varepsilon}^2$$
The optimism increases linearly with the number $d$ of inputs or basis functions we use, but decreases as the training sample size $N$ increases.

An obvious way to estimate prediction error is to estimate the optimism and then add it to the training error $\overline{\text{err}}$. $C_p$, AIC, BIC and others work in this way, for a special class of estimates that are linear in their parameters.
In contrast, cross-validation and bootstrap methods are direct estimates of the extra-sample error $\text{Err}$. These gen- eral tools can be used with any loss function, and with nonlinear, adaptive fitting techniques.

Why do we want a good estimate of the conditional error? It's the average error that the model you fitted on a particular training set $\mathcal{T}$ would incur when applied to examples drawn from the distribution of $(X,Y)$ pairs. If you lose money each time the fitted model makes an error (or proportional to the error if you're talking about regression), it's the average amount of money you lose each time you use the classifier. The expected error is an average over all training sets you will never see.

#### 7.5: Estimates of In-Sample Prediction Error (MISSING)
#### 7.6: The Effective Number of Parameters (MISSING)
#### 7.7: The Bayesian Approach and BIC (MISSING)

#### 7.10: Cross-Validation
Cross-validation is the simplest and most widely used method for estimating prediction error. However, be aware that it only measures the expected prediction error, not the conditional error.

##### 7.10.1: K-Fold Cross-Validation
Motivation: do not want to set aside a whole sample only for evaluation because we want as much data as possible. Uses part of the data to fit the model, and part to test it.

We split the data into K roughly equal-sized parts; for example, when K = 5, the scenario looks like this:
![](CV.png)

If the learning curve has a considerable slope at the given training set size, five- or tenfold cross-validation will overestimate the true prediction error.
![](learning_curve.png)

##### 7.10.2: The wrong and right way to do cross-validation
We are not allowed to do supervised learning before cross-validation, but we can do unsupervised learning. E.g. we can choose the predictors with highest variance across all the potential predictors. In this case, the predictors will not be given an unfair advantage of having seen the responses.

##### 7.10.3: Does cross-validation really work?
Yes, one average it works. But we should always report the estimated standard error of the CV estimate due to its variability.

#### 7.11: Bootstrap Methods
Bootstrap is a general tool for assessing statistical accuracy. As CV, it typically estimates the expected prediction error.

Suppose we have a model fit to a set of training data. We denote the training set by $Z = (z_1, z_2,\ldots, z_N)$ where $z_i = (x_i, y_i)$. The basic idea is to randomly draw datasets with replacement from the training data, each sample the same size as the original training set. This is done $B$ times ($B = 100$ say), producing $B$ bootstrap datasets.

There are three different estimates of error from bootstrapping:

* Fit the model by bootstrap and see how well the $B$ models perform in predicting the original training set. Let $\hat{f}^{*b}(x_i)$ denote the predicted value at $x_i$, from the model fitted to the $b$th bootstrap dataset.
$$ \widehat{\text{Err}}_{\text{boot}} = \frac{1}{B}\frac{1}{N}\sum_{b = 1}^B \sum_{i = 1}^N L(y_i, \hat{f}^{*b}(x_i)) $$
However, this estimate is not a good one. The reason is that the bootstrap datasets are acting as the training samples, while the original training set is acting as the test sample, and these two samples have observations in common. This overlap can make overfit predictions look unrealistically good, and is the reason that cross- validation explicitly uses non-overlapping data for the training and test samples.
$$Pr\{ \text{observation} \ i \in \text{bootstrap samle} \ b \}
= 1 - \Big(1 - \frac{1}{N} \Big)^N$$
$$\approx 1 - e^{-1} = 0.632$$
* By mimicking cross-validation, a better bootstrap estimate can be obtained. For each observation, we only keep track of predictions from bootstrap samples not containing that observation. The leave-one-out bootstrap estimate of prediction error is defined by
$$\widehat{\text{Err}}^{(1)} = \frac{1}{N}\sum_{i = 1}^N \frac{1}{|C^{-i}|}\sum_{b \in C^{-1}} L(y_i, \hat{f}^{*b}(x_i))$$
Here $C^{-i}$ is the set of indices of the bootstrap samples $b$ that do not contain observation $i$, and $|C^{-1}|$ is the number of such samples. In computing $\widehat{\text{Err}}^{(1)}$, we either have to choose $B$ large enough to ensure that all of the $|C^{-1}|$ are greater than zero, or we can just leave out the terms corresponding to $|C^{-1}|$’s that are zero.
* The leave-one out bootstrap solves the overfitting problem suffered by $\widehat{\text{Err}}_{\text{boot}}$, but has the training-set-size bias mentioned in the discussion of cross-validation. The average number of distinct observations in each bootstrap sample is about $0.632N$, so its bias will roughly behave like that of twofold cross-validation. Thus if the learning curve has considerable slope at sample size $N/2$, the leave-one out bootstrap will be biased upward as an estimate of the true error. The “.632 estimator” is designed to alleviate this bias. It is defined by
$$\widehat{\text{Err}}^{(.632)} = .368 \overline{\text{err}} + .632\widehat{\text{Err}}^{(1)}$$
The derivation of the .632 estimator is complex; intuitively it pulls the leave-one out bootstrap estimate down toward the training error rate, and hence reduces its upward bias.
The .632 estimator works well in “light fitting” situations, but can break down in overfit ones.

## Chapter 9: Additive Models, Trees, and Related Methods
#### 9.1: Generalized Additive Models
A general additive model is a non-parametric regression model with the form
$$E(Y | X_1, X_2, \ldots, X_p) = \alpha + f_1(X_1) + f_2(X_2) + \cdots + f_p(X_p)$$
where the $f_j$'s are unspecified smooth ("non-parametric") functions.

##### 9.1.1: Fitting Additive Models
Here, we use the cubic smoothing spline as our scatterplot smoother.
The additive model has the form
$$ Y = \alpha + \sum_{j = 1}^p f_j(X_j) + \varepsilon $$
In order to penalize complex functions, we specify the penalized sum of squares as
$$ \text{PRSS}(\alpha, f_1, f_2, \ldots,f_p) = \sum_{i = 1}^N \Big(y_i - \alpha - \sum_{j = 1}^p f_j(x_{ij}) \Big)^2 + \sum_{j = 1}^p \lambda_j \int f_j^{''}(t_j)^2 dt_j$$
![](backfitting.png)

#### 9.2: Tree-Based Methods
The algorithm finds, at each node, which variable to use a the splitting variable, and the split value. E.g. in a prediction model of the price of a car, the first node might ask whether the age of the car is above or below 2 years. How large should we grow the regression tree? This is the classical trade-off in prediction. A large tree might overfit the data (large complexity), while a small tree might not capture the important structure (low complexity).



## Chapter 10: Boosting and Additive Trees
#### 10.1: Boosting Methods
Idea: a procedure that combines the outputs of many “weak” classifiers to produce a powerful “committee”. The most popular boosting algorithm is called “AdaBoost.M1". Consider a two-class problem, with the output variable coded as $Y \in \{−1, 1\}$. Given a vector of predictor variables $X$, a classifier $G(X)$ produces a prediction taking one of the two values $\{-1, 1\}$. The error rate on the training sample is
$$ \overline{\text{err}} = \frac{1}{N} \sum_{i = 1}^N I(y_i \neq G(x_i))$$
and the expected error rate on future predictions is $E_{XY} I(Y \neq G(X))$.

A weak classifier is one whose error rate is only slightly better than random guessing. The purpose of boosting is to sequentially apply the weak classification algorithm to repeatedly modified versions of the data, thereby producing a sequence of weak classifiers $G_m(x), \ m = 1, 2, \ldots, M$.
![](adaBoost.png)
The predictions from all of them are then combined through a weighted majority vote to produce the final prediction:
$$G(x) = \text{sign} \Bigg( \sum_{i = 1}^M \alpha_m G_m(x) \Bigg)$$
Here $\alpha_1, \alpha_2, \ldots , \alpha_M$ are computed by the boosting algorithm, and weight the contribution of each respective $G_m(x)$. Their effect is to give higher influence to the more accurate classifiers in the sequence.
![](adaBoostAlgorithm.png)

#### 10.2: Boosting Fits an Additive Model
Boosting is a way of fitting an additive expansion in a set of elementary “basis” functions. Here the basis functions are the individual classifiers $G_m(x) \in \{−1, 1\}$. More generally, basis function expansions take the form
$$f(x) = \sum_{m = 1}^M \beta_m b(x; \gamma_m)$$
where $\beta_m, \ m = 1,2,\ldots,M$ are the expansion coefficients, and $b(x;\gamma) \in \mathbb{R}$ are usually simple functions of the multivariate argument $x$, characterized by a set of parameters $\gamma$.

#### 10.3: Forward Stagewise Additive Modeling
![](fsam.png)
At each step, $m$, we produce $f_{m}(x)$ by adding the optimal basis function, $b(x; \gamma_m)$, and its coefficient, $\beta_m$, to the current expansion, $f_{m - 1}(x)$.

#### 10.4: Exponential Loss and AdaBoost
The AdaBoost.M1 is equivalent to forward stagewise additive modeling using the loss function $L(y, \ f(x)) = exp(−y \ f(x))$.

#### 10.5: Why Exponential Loss?

#### 10.6: Loss Functions and Robustness


#### 10.9: Boosting Trees
#### 10.10: Numerical Optimization via Gradient Boosting
