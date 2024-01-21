# [Random Forests Part 1 - Building, Using and Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

- Recommended reading: __The Elements of Statistical Learning__

_Decisin trees_ have flaws, they are __innaccurate__, they are not flexible after training. they overfit. 

- __Bootstrapped dataset__ is used when creating a _random forest_, by randomly drawing from the traiing data with repetitions. 

- Use _randomly selected featrues_ at each step and a _bootstrapped dataset_ to create a decition tree (i.e., the root node feature is randomly selected, and so on.) 

- Repeat the previos step many times (100+) 

Run the training data through all the trees, count the votes for the outcome, and use the _majority voting_ as an __aggregate__ to make the final prediction. 

`!` __Bagging__ is when you _bootstraping_ the data and use _aggregate_ (e.g., _majority voting_) to make a decision.  

#### Evaluating the algorithm

`!` __Out-of-Bag dataset__ is the data from original dataset that __did not fall__ into the _bootstrapped_ dataset. Generally 1/3 of the original data does not end up in the bootstrapped dataset, as we draw with repetition. 

- Out-of-bag (OoB) data can be used to evaluate the accurcy of the random forenst algorithm, as what proportion of OoB samples are _correctly_ labelled.  

`!` __Out-of-Bag Error__ is the _proportion_ of the OoB samples that are _incorrectly_ classified.  

#### Building a Random Forest

- Buld a random forst with _square of th number of varaibles_ 
- Evaluate accuracy of random forest
- Change the set of random features used to build each tree (go below the initial guess and above)
- Repeat untill the best pefroming random forest is found

---

# [Random Forests Part 2: Missing data and clustering](https://www.youtube.com/watch?v=sQ870aTKqiM)

- Missing data:
    - Missing data in training dataset (solved by making an initial guess and then refining it). 
    The initial guess is for numericl value is a _median_ value for this feature using the subset of data tha agrees with the guess for other features.  
    Refining the guess is done by:
        - Build a random forest
        - Rund data through all trees
        - If samples end up in the same leaf nodes, it means the _trees are similar_. 
        `!` __Proximity Matrix__ is used to keep track on the samples that end up in the similar nodes, it consits of columns and rows (both for each sample). 
        - If two simaples end up in the same leaf node, we put 1 (both, above and below the diagonal). 
        - As we run the data through all the trees we _add_ the counts if data end up in the same leaf node. 
        - Then we devide each _proximity value_ by the total number of trees. 
        - The _proximity values_ are used to make better guesses about _missing data_ by computing _weighted frequency_ of outcomes using _proximity values_ as _weights_. 
            - _Weight_ is then the value of the proximity / All proximities for this sample
            - The _frequency_ is computed as a N of this value / Total N of values for this feature. 
        - The proximity matrix can be used to compute _distance matrix_ as 1-proximity values. 
        - _Distance Matrix_ can be visualized with _heat map_ and with _MDS plot_. 
        - `!` Random forest allows to consider the relation between different samples using the _proximity or distance matrix_ visualized via heat map or MDS plot
    
    - Missing data in the data we need to predict for. 
        - Create two datasets that we need to predict for, _assuming_ different outcoms in each. 
        - Use _iterative method_ as before to make a good guess about the mising value.  
        

