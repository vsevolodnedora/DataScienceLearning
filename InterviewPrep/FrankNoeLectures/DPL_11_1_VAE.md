# Directed Generative Nets

Area of ML , where we atre trying to observe the _probaiblity distribution_ by samples and _learn it_ and have a way to generate new samples from this distribution. 

__Example__:  
- restricted boltzman machine
- directed generative nets 

In an example, we want to learn a model that takes _random noise_ as an __input__ and learns a transformation from it to the given data. This is __Generative Model__, a model that learns to generate output that resembles the output.  

Input: random sampling from a probability distribution that is easy to generate e.g., independent gaussian variables. 
Transform a meaningless randomness to meaningfull randomness.  

__IDEA__: learn to sample _intractable_ (difficult to directly generate samples) probability distribution $p(x)$ by sampling _tractable latent distribution_ (easy to sample, e.g., multivaraite gaussian) 
$$
z \sim p(z) // Sample z from p(z)
$$ 
and perform a tranformation to the desired distribution 
$$
x = G(z,\theta)\sim p(x) // Transform variable z with some function G(z,\theta) to generate samples of x, i.e., sample the distribution p(x)
$$
where $\theta$ are the parameters.

The __aim__ is that the function $G(z,\theta)$ is trained, so that its output $x$ approximately samples _from_ the target distribution $p(x)$.  

Generally, the $p(x)$ is given in terms of samples, a _dataset_. 

`!` Architecture of $G$ defines family of possible $p(x)$ distributions  
`!` Parameters $\theta$ _select_ distribution from that family


`?` How can be we measure the _difference_ of our __Generator__ $G(z,\theta)$ output, and samples that implicitely define $p(x)$, and how we can turn it into _loss function_.  

__Idea__ comes from _transformation of random varaibles_, used in statistics.  

### Simple distribution

Examples of how a _transofration_ of a random varaible can be used in order to sample from a simple distribution and transform it to approximate a more complex distribution.  

__Example 1__: Generate samples from _normal distribution_ with mean $> 0$ and covaraince matrix $\Sigma$ (multi-dimensional form of varaince, a _symmetrix, square matrix_ giving the covariance between each pair of elements of a given random vector)
- Generate samples $z ~ N(0,1)$, where $N(0,1)$ is the standard normal distribution of $0$ and _identity matrix as a covariance matrix_ 
- Transofrm $z$ via a linear transofrmation, my multiplying by matrix $L$ and adding _vector_ $\mu$ as $G(z) = \mu + L * z$, where $L$ results from __Cholesky decomposition__ of covaraince matrix, i.e., $\Sigma = L^T L$

`!` This is how a standard normal distribution can be _transformed_ into a distribution with requirem mean and covaraince matrix. 

__Example 2__: Consider generating samples from a _univariate distribution_ $p(x)$. 
- Assume $p(x)$ is a positive distribution, e.g., $P(x) > 0 $ for all $x$
- Compute cumulative distribution function $q(x) = \int_{-\infty}^x p(v) dv$ and _invert_ this function (as it is _strictly monotonous_, and therefore _invertable_)
    - Sample $z ~ Uniform(0,1)$ 
    - Define $G(z)$ and an inverted cumulative distribution evaluated at $z$ $q^{-1}(z)$
    - Find, at what $x$, the cumulative distribution $q^{-1}(z) = G(z) = x$,

`!` Thus here we transformed $z$, a unoform random varaible into _a sample from a desired_ univariate distribution $p(x)$.  

In a Neural Network we take a complex, tranable distribution $G()$ in order to take simple input $z$ in order to approxmate compelx distriubiton.

`!` Generate interesting sahpes in output distribution if we learn a _suitable_ function. 


### Complex distribution

Assume that $G$ is a neural network, a _feed-forward NN_, and we want to train $\theta$, in order to __achieve sampling from a correct distribution__, i.e., distribution that can mimic the training data. 

__Examples__:
- __VAEs__ (inference net + generator net)
- __GANs__ (generator net + discriminator net)
- Normalizing flows

The main difference is how we train them and whether they have an _explicit way_ to compute the prob. density that we are generating form or not.  

### VAE 

In a VAE we have an _encoder_ and _decoder_ network:
- __Encoder__ go from high dimensional image space to low-dimensional latent space 
- __Decoder__ go from low dimensional latent sapce to high dimensional image space

`!` Latent space enodes certain, essential properties of the data. 

`!` Autoencoder is trained by _minimizing thereconstruction error_ of decoder with respect to the input an its learn to _generalize_ as the latent space has a lower dimension. The information _bottleneck_. This is a _deterministic autoencoder_. 

`!` Variational autoencoder instead of single value for a latent space dimension uses a _probability distribution_. So each parameter, value in the latent space there is now replaced by a probability. It is a _probabilisitc description_ of the latent space varaibles. 

`!` VAE is a feed-forward NN, a directed model (can be trained with purely gradent based methods and back-propagation), where the __Encoder__ is representing the input as a _probability distribution_, it encodes __the momentes of a distribution__ (means and variances of gaussians) and a __Decoder__ samples from this distribution to get an output.

`!` The minimization of reconstruction error implies, that close points in a latent space will lead to similar output. So, there is a _continois, smooth latent space represenation_ (this is not the case in normal autoencoders)


# Variational Inference
