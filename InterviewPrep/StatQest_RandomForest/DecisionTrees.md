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

`!` Total __Gini Impurity__ is a _weighted average_ of __Gini Impuriities__ for the leafs.  
The weights are caclulated as N(samples in a leaf) / N(samples in all leafs). 

So the total __Gini Impurity__ is a summ of weights multiples the associated _Gini Impuruty_.  

Overall, after taking _every_ parameter to use a tree, we compute the _weighted average of Gini coefficent_ for each and compare. For numeric columns it is more triccky and involes _sorting_ and computing the _mean value_. 

Then, the feature that gives the _lowest weighted Gini impurity coefficient_ is chosen as a root node for the decision tree. Then, we consider other parameters and compute the _weighted impurity_ for each 

