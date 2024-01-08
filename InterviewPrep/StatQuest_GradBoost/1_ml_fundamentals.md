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

