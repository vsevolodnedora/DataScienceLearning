# Dedision Trees

When a _Dedision Tree_ classifies a things into _categories_ it is a __Classification Tree__.

When a _Decision Tree_ makes a predition it is a __Regresssion Tree__

### [Classification Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y&ab_channel=StatQuestwithJoshStarmer)

__NOTE__ the numeric threscholds can be _different_ for the same data between different braches of a given tree

Within a tree 
- True is left 
- False os right

`!` Top of the tree is called a __Root Node__ 
`!` Intemediate ndes are called __internal nodes__ or __bracnches__. 
`!` Outer most nodes are called __leaf nodes__ or __leafs__

`!` Impure leafs are leafs that end with a micture of positive an negatives values. In othwer words, when we construct a tree we select a featre and we ses how thsi feature _alone_ can predict the outcome. In doing this we collect the data points for each leaf of each tree with 'es' or 'no' and if we end up with a mix of those in a tree, the resulte _leaf_ is __impure__. 

We check which feature gives a tree that heas the east amount of impure leafs. This tree then is chosen as te best predictior. 

`!` __Gini Impurity__ and __Entropy__ and __information gain__ are used to _quantify_ the difference between different decision trees based on different features.

The _Gini Impurity_ of a leaf is 
$$
1 - P(yes)^2 - P(no)^2, 
$$
here each probability is a N(yes or no)/N(total)

- If a leaf splits data perfectly, both leafs are pure, the Gini Coefficnet is 0. 

`!` Total __Gini Impurity__ is a _weighted average_ of __Gini Impuriities__ for the leafs.  
The weights are caclulated as N(samples in a leaf) / N(samples in all leafs). 


So the total __Gini Impurity__ is a sum of weights multiples the associated _Gini Impuruty_.  

Overall, after taking _every_ parameter to use a tree, we compute the _weighted average of Gini coefficent_ for each and compare. 
- For _numeric columns_ it is more tricky and involes _sorting_ and computing the _mean value_ for all _adjacent_values (binary mean), and computing gini impurity for each average value (use it as a decision boundary). 

Then, the feature that gives the __lowest weighted Gini impurity coefficient__ is chosen as a root node for the decision tree. Then, we consider other parameters and compute the _weighted impurity_ for each branch of the tree where the data is now _split_ (according to the root node). 

__NOTE__: At each node of the tree, we split the data. Thus, dataset is being reduced as we build the tree. 

__NOTE__: we only split _impure_ trees, if leafs are impure. We all leafs are pure, we cannot build tree further. 

`!` The output of a _Leaf_ is whatever category that has the most _votes_. 

__NOTE__: if a leaf has very small sample of data, e.g., 1 value, that the tree _may overfit_ the data. Solutions:
- __Prunning__
- __Limit Tree Growth__

---

### [Regression Trees](https://www.youtube.com/watch?v=g9c66TUylZ4)

`!` Regression tree is a _decision tree_ where each leaf represents the numeric value.  

The regression tree is build by considering 
- first point; split dataset into 1 + the rest. Compute residuals between the first one and the mean for it (which is just this one value) and rest of the points and the mean value computed for _all of them_. Then compute the _sum of squared residuals_ for both leafs (woth one point and with the rest of the points). 

- Thus, for _each value of the threshold_ we compute the _sum of squared residuals_ using _mean values_ for splitted sides of the dataset. 

- We move the "threschold" by one datapoint value and each time we repeat the analysis. 

Than we select the _treschold_ with the __Smallest SSR__ as a root node of the _regression tree_.  

`!` The decision tree is build by finding thrscholds that give the smallest SSE.

- At each splitting we repeat the analysis untill we fild leafs (e.g., all values in the leaf are the same and cannot be split further). 

__NOTE__: it is possible that the model will _overfit_ the data having _small bias_ but _large viaraince_.  

__Solutions to overfitting__:
- Do not split the data when the N of datapoints is < than a certain value, e.g., 20  
The ouptut for the leaf then is the mean value of data points in a leaf

For multy-feature dataset, we find what feature gives the mininimum optiminal SSR (so we find the optimal threshold) and use it as a root. 
