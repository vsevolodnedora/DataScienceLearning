# Pre-training large language models

Recall the GenAI project life cycle.  
1. Scope and defining a use case
2. Choosing an existiting model or pretraining your own

Choosing a model
- Work with exisitic 
- Training a new model 

In general it is common to start with existing models.  
There are many open source models.  

Most hubs, like, _hugging face_ have _model cards_ that describe model: 
- Training porcess
- Use cases 
- Limitations

The choice of the model depends on the task.  

Traning determines the model ability.  


### Pre-training of models

This is done using _self-supervised learning_. 
Using large amount of unstractured text (internet scapes, special datasets for LLMs).  
The model learns _patterns_ and _structures_ in the language. 

Then, model is capable to complete _training objectives_ that in turn depends on the acrutecture of the model. 
During draining, `weights` of the model are continously updated to minimise the `loss of the training objective`.  
- Encoder generates embeddings for each token

Requires a large amount of compute.  

__NOTE__: when scraping, the data must be first _curated_ (quality control), address bias etc.  
As a result (98%) ofdata may be discarded.  

There are three varaints of _transformer model_.  
- __Encoder-only models__ (`auto-encoding models`) - Trained using _Masked Language Modelling_ (MLM), tockens in the input sequence are randomly masked and the training objective is to predict the masked tocken. This is called `de-noising objective`. Models build _bi-directional representation of the input sequence_. I.e., model builds understading of the full context of a tocken (from front and back). Examples: BERT, ROBERTA.
The following tasks benifit from this _bi-directional understadning_: 
    - Sentiment analysis
    - Named entity recognition
    - Word classification
- __Decoder-only models__ (`auto-regressive models`) - Trained using _Casual Language Modelling_ (CLM), where given a sequence of tocken the goal is to predict _the next one_. Sometimes it is also called _full language modelling_. There the input sequence is masked and this masked sequence has to be predicted one tocken at a time. Here the _context is uni-directional_. Model builds a statistical representation of language. Examples: GPT, BLOOM. These models often used for 
    - Text generation
    - Other emergent behaviour (depends on the model size)
- __Encoder-decoder models__ - (`sequence-to-sequence model`). Pre-training objective is highly model-dependent. For example, model T5 pre-trains the encoder using _Span Corruption_, where a randum _sub-sequence of the input sequence_ is masked. Those masked parts are replaced with unique `sentinel tocken`. They are added to the vocabluary but they _do not correpond_ to a word of the input text. Decoder must recustruct the masked part _auto-regressively_. Examples: T5, BART. Applications:
    - Translation
    - Summarization
    - Question-answering. 

__Summary__:
- Encoder-only models: Autoencoders; Trained: MLM; Goal: setnece/tocken classification  
- Decoder-only models: Autoregressive; Trained: CLM; Goal: text generation  
- Encoder-Decoder models: Seq2Seq; trinaed: Span corruption; Goal: summarization; Q/A   

Larger models are generally more flexible and do not require model fine-tuning. Hence, Model size is a key. Training models is numerically expensive. 


### Computational challenges of training LLMs

Models requrie huge amount of memory. e.g., CUDA out of memory Error.  
Single parameter 4 bytes (32-bit float)  
1 billion paramerers -> 4GB memory (just for model weights). 

__Additional memory for training__:  
- Adam optimizer (2 states): + 8 bytes per parameter (twice the size of weights)
- Gradients: +4 bytes per parameter
- Activations: 8 bytes per parameter (high-end estimate)

So it is about 20 extra bytes of memory per parameter.  
The rune-of-thumb is to take 6 times more memory than model weights alone require. Thus, for 1 billion parameter model, ~24 GB GPU memory is required.  

__Techniques to reduce memory requirements__: 
- Quantization : 32-bit floating points -> 16-bit -> 8-bit integers. 

`Quantization` statistically _project_ the original 32-bit floating point into a lower-precision space using scaling factors computed using the range of original 32-bit floats (sometimes suported by training itself to learn the scaling factors).

Floating point representaionn (32-bit)
- 1 bit for Sign (0 or 1)
- 8 bits for exponent (1000000)
- 23 bits for fraction / matissa/significance/precision (1001001000001111111011000)

- Float32 : 4 bytes : 32 bits (1+8+23) : -3e38 to 3e38 
- Float16 : 2 bytes : 16 bits (1+8+7) : -65504 too 65504
- BFLOAT16 : 2 bytes (hybrid precision; helps in training precision; uses 8 bits for eponen but only 7 bits for fraction). It is bad for integer calculation. 
- Int8 : 1 byte : 8 bits (1+0+7) : -128 to 128 (1bit sign 7 bits for fraction (no exponent))

Thus, pi=3.1415920257568359375 in float16-bit is 3.150625 and 3 in int8


Quantization reduces memory consumption significantly. 

Modern models have 500+ billion parameters and require 10,000+ GB of memory to train.  
These models are trained using _distributed computing_. 


### Efficient multi-GPU compute strategies

Dataloader: Split dataset between several GPUs; process batches of data in parallel.  
This can be done using PyTorch: `Distributed Data Parallel (DDP)`.  
It copies the model onto each GPU and sends batches of data to each GPU in parallel. Then there 
is Forward and Backward pass; then the resut is _combined_ in _Synchronize gradients_ step.  
After; model on each GPU is updated. This allows parapllel computation.  
_NOTE_ in this case models _has to fit_ onto a given GPU as it is copied. If it is not possible, consider _model sharding_.  

PyTorch as an implementation of sharding as `Fully Sharded Data Parallel (FSDP)`. Achived via ZeRO: Zero Redundancy Optimizer. Allows to split model across GPUs when the models does not fit on one GPU.  

Model memory us mostly located in 2 optimizer steps; then in gradients; than in model parameters.  

In sharding, model parameters are _distributed_ across GPUs.
- ZeRO Stage 1: only optimizer states are sharded (reduces memory foot-print by up to 4)
- ZeRO Stage 2: shards gradients and optimizer states (reduces memory footprint by up to 8 times)
- ZeRO Stage 3: shards also model weights (reduces memory linearly with the number of GPUs)

In this approach, however, weights need to be aggregated between GPUs 2 times: 
- Before Forward Pass 
- Before backward Pass

Each GPU reqeqsts data from other GPUs on demand to materialize _sharded data_ into _unsharded data_ for the duration of operation.  

This is Performance VS Memory trade-off.  

At the end FSDP synchronises gradients as in the case of DDP.  

Thus, FSDP:
- Reduces overall GPU memory untilization
- Supports offloading to CPU if needed