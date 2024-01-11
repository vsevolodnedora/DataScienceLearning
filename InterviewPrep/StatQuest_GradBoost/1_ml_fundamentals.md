# Machine Learning Fundamentals: Bias and Variance

[Video](https://youtu.be/EuBBz3bI-aA?si=QzMujnmqWC_8vLgI)

- Important: Split data into _train_ and _test_ sets (to assess _bias-variance_ tradeoff)
- Try simple ML model. Start with e.g., __linear regression__. 
- `Bias` - inability of a model to capture true relationship in data `Underfitting` - high bias
- `Variance` - inability of a model to generalize to test set. `Overfitting` - high variance
- __Sum of squares__ allow to assess the model performance.  

#### Methods to achive good bias/variance balance
- Regularization 
- Boosting
- Bagging


### 1. [Gradient Boost Part 1/4: Regression main ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc)

In __AdaBoost__ the algorithm builds _stumps_ such as to correct the errors the previos _stumps_ with the weight assigned based on the error it has. 
__AdaBoost__ builds many stumps untill the req. critroeion is met. 

__Gradient Boost__ starts by building a single _leaf_. The _leaf_ is the initial guess, which for continous variable is generally a _mean value_. 
- It buiilds _fixed size_ trees based om the previous tree's errors (larger than a _stump_). 
- Gradient boost _Scales_ the trees by _the same amount_. 
- Next trees are built using the previos tree errors. Then, the tree is scaled based on its error. 

__Algorithm__:  
- Compute the mean of the continous variable
- Compute _pseudo-residual_, via $y - \hat{y}$
- Build a tree to _predict the pseudo-residual_ (note the number of leafs is restructred, so usually N leafs < N resuduals). Multiple values/entire range may end up in the same leaf -> _take average_. 
- Combine original leaf with the tree, which, as the tree predicts errors, may lead to a perfect fit. In fact the model _overfits_ the data having _high variance_. This is compensated by using `learning rate` $\nu \in [0,1]$ as $\hat{y} = \mu + \nu \hat{y}_0$, where $\hat{y}_0$ is the prediction from the first tree. With the learning rate the prediction is not perfect, but the variance is not large. 
- Compute again the pseudo-residuals using the aforementioned model. The new residuals are smaller than the once computed before. 
- Build another tree (branches can generally be different). And again, sice N values may end up in the same leaf, compute averages. 
- Combine the trees and initial leaf, scaling the trees by _learning rate_. 
- Repeat... 

Each time a tree is added _residual decrease_.  
The result is a chain of trees.  


__SUMMARY__: When Gradient Boost is used for regression we start with a _leaf_ that is the _average_ value of the variable we want to predict. Than, add a tree based on _Residuals_ (Observed - Predicted values). Scale the tree contribution by _learning rate_. 


### 2. [Gradient Boost Part 2/4: Regression Details](https://www.youtube.com/watch?v=2xudPOBz-vs)

Consider the original algorithm:

__Input__: 
    - _Data_ a is given as $\{ (x_i, y_i) \}^n_{i=1}$ 
    - Differentiable __Loss Function__ $L(y_i, F(x))$ (Common: $0.5 * (y_i-\hat{y}_i)^2$, similar to _linear regresion_ e.g., _squred residuals_, with $0.5$ is usefull in _chain rule_). 

__Step 1__: 
    - Init the model with $F_{0}(x) = \argmin \sum_{i=1}^n L(y_i,\gamma)$, where $\gamma$ is the predicted value, and $\argmin$ means get the _minimum_ $\gamma$ that _minimizes_ this $\sum(\cdot)$. 
        - At first the $\argmin$ can be found anaoytically and it gives the _average_, i.e., the mean value
        - This is a single value, so it is _a leaf_. 
        - So the model is initialized with a _constant value_. 

__Step 2__ (aloop with trees): 
    - Compute _derivative/gradient of the loss function_, i.e., the _pseudo-residual_ $r_{i,m} = - \Big[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \Big]_{F(x)=F_{m-1}(x)}$ for $i = 1,...,n$. 
    - _Fit a regression tree_ to the $r_{i,n}$ values and create a _termina regions_ (or final, averaged if needed _leafs_) $R_{j,m}$ for $j=1...J_m$ where $j$ is the index of the leaf. 
    - _Optimize leaf output values_: For $j=1...,J_m$ compute $\gamma_{j,m} = \argmin \sum_{x_i\in R_{i,j}} L(y_i, F_{m-1}(x_i) + \gamma)$
        - __Note__: here we only consider samples from the leaf; 
        - $\gamma_{j=1,m=1}$ is the prediction from the 1st leaf and 1st tree.
        - $F_{m-1}(x_i)$ is the latest predicted value for this varible
        - After simplicfication, we get $\gamma_{1,1} = \argmin 0.5 (-\text{Constant} - \gamma)^2$, where Constant is from summing up previos prediction and observed value. Take _derivative w/r to loss function_ $=0$ to solve it. Numerically, via _gradient descent_ or analytically via _chain rule_. 
        - After solving we arrive to the _value that this leaf predicts_. 
        - __NOTE__: the output value for the leaf with chosen loss function is always the averaged of values that end up in the leaf
    - _Make new predictions for each sample_: $F_{m} = f_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{j,m} I(x\in R_{j,m})$, where $I(x)$ is the tree built in this step, where the summation is added _if one sample exists in multiple leafs_. The summation is over all output values of _all the leafs_, where sample $x$ can be found. Here $\nu$ is the learning rate

__Step 3__: Output the final prediction


### 3. [Gradient Boost Part 3/4: Classification](https://www.youtube.com/watch?v=jxuNLH5dXCs)

In classification, the _logistic regression_ "analoge" of average is the __log(odds)__. So, the Gradient boost initial prediction is log(odds). Log(odds) = log(sum(+)/sum(-))$ where + amd - together are all the values. 

Log(odds) can be _converted_ to probability using the _logistic function_ as: 

`!` prob of (+) = exp(log(odds)) / (1 + exp(log(odds)))

If probability > 0.5, the prediction is + otherwise it is -

To assess the accuracy of the prediction, we can compute _pseudo-residauls_ as before. We use probabilities to compute the difference. Where the observed values are 0 or 1 for - or +.  
`!` A _residual_ is a _difference in probabilities_. 

Next. build a _tree_ with a limited number of tree. Compute output values for the tree. 

`!` In Gradient Boost for regression a leaf with a _single_ residual has an output _equal_ to that residual. Here, however, predictions are probabilities, while initial prediction is log(odds). We need to transform the prediction as sum(Residuals ) / sum(previos probability * (1 - previos probability)) where sums are over all $i$ (in case multiple predictions exist for a single leaf)

This transformation is done for each leaf prediction. 

As before, _learning rate_ is used to scale new tree prediction.  

Each time we compute the log of the odds and a probability in the end. 


### 4. [Gradient Boost Part 4/4: Classification Details](https://www.youtube.com/watch?v=StWY5QWMXCw)


__OVERALL__: In a gradient boost, a set of trees is build, where each tree gives a prediction based on the input features. The prediction is optimized through gradient descent in training. The results from all trees is combined, scaled by learning rate. 

---

## Regularization

- Used when there are _a lot of parameters_. 

### 1. [Rdige regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
__Method to improve model when there are few data points__ 

Idea is to fit training data adding an aditional bias to the regrssion model, so as to account for the limited data of the training set.  In other words, by adding _bias_ we arrive to a lower _variance_ in the trained model. 

- Rdige regression = $L_2$ norm = sum of squred residuals + $\lambda$ * slop $^2$  


`!` Ridge regression penalty ads bias that in turn reduces the overall fit variance. 

The idea is that the steep slope leads to large variance, as small change in $x$ leads to large cahnge in $y$. Reducing the slope with _ridge regression penalty_ decreases this dependency. 

The additional parameter $\lambda$ should be set _manually_ or via _cross-validation_ with multiple folds to achieve _smallest varaince_. 

- _Rdige regression_ also works with descrete data. 
- It can also be applied to _logisitc regresion_ as Pred. = y-intervept + slope * weight. There rirge regression optimises _the sum of likelihoods_ instead of _squared residuals_. __NOTE__: logistic regression is solved using _maximum likelihood_. 

- Rdige regression can be used in a multiple parameter regresssion with term being $\lambda * \sum$ of weights for different features. It includes everything except y-intercept, as these parameters are _scaled_ by meansurments. 

__NOTE__: _Ridge regression_ allows to perform regression, when there are _less_ data points than parameters. 

---

### 2. [Lasso and Ridge Regression](https://www.youtube.com/watch?v=NGf0voTMlcs)

Also called $L1$ morm

`!` Lasso Regression is _different_ from ridge regression. It includes the $\lambda$ * |the slope| term _instead_ of the squared slope. It thus introduces _less biass_. 

`!` Lasso Regression can shrink slope _all the way to 0_, while Ridge Regression shrinks it _assymptotically close to 0_

`!` Lasso Regression is usefull _when there are useless parameters_. Similar to _Dimensionality Reduction_ methods. 

---

### 3. [Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo)

Elastic Net Regression is a combination of Lasso and Ridge regression, as 
$\lambda_1 \sum$ of _modulus_ of variables __plus__ $\lambda_2 \sum$ of _squares_ of varaibles.   

The parameters can be found with _cross-validation_. 

`!` Especially usefull when there are _correlations_ between parameters. 
- It _groups_ and _schrinks_ the parameters associated with the correlated variables and leaves then in equation or removes them all at once. 

### 4. [Visualization](https://www.youtube.com/watch?v=Xm2C_gTAl8c)

Visualizing this we can image a set of parabolas in SSE + $\lambda*$solpe$^2$ vs Slope Values. As we increase $\lambda$, the _parabola moves up and left_ with its bottom, representing the __optimal slope__, moving up and left, closer _Zero Slope_. 

If we consider Lasso Regression, the parabola shape _is not preserved_. The _kink_ appears when one of the parameters becomes _discarded_. 

