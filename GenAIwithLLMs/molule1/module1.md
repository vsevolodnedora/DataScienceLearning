# Generative AI & LLMs

Setting the sceene.  
Prompt engineering
Project lifecycle.  

- Generative AI is a subset of traditional Machine Learning. Trained by finding statistical patterns in massive datasets. 
- LLMs trained on huge corpi of data exhibit emergent abilities beyond language alone. 
- Examples of LLMs (ordered by size, larger to smaller, top to bottom)
    - GPT
    - BLOOM
    - FLAN-T5
    - PaLM
    - LLaMa
    - BERT
- Size of the model equivalent to its memory; the larger the model the more diverse set of tasks it can perform.  
- GenAI is multimodal. LLMs however are primarely language-base.  
- LLMs are different from standard code; as it takes natuaral language as an intput and output the result that is also a text. 

`Prompt` - text that we pass to an LLM  
`Context window` - size of the ext that can be passed it (> 1000 wordsrrrr)  
`Completion` - the output if the model (consists of original prompt and model output)  
`Inference` - using an LLM to generate completion of a prompt  

> Next word prediction is the base of many LLM capabilities


### LLM use cases and tasks

- Write an Essay based on prompt
- Summarise text
- Translation 
- Translate natural language to machine code
- Small focused tasks:
    - Information retieval (Name-entity recognition; e.g., word classification)
- Augmenting LLMs with external sources/APIs

> The larger the size of the model the better the subjective understanding of the language of a model
> Language understanding stored in the parameters of the model allows model to process reason and solve the task


### Text generation before transformers

Before transformenrs, recurrent neural networks (RNNs) were used.  
RNNs were limited by the compute and memory.  
For a next-word prediction model requires _see_ preceeding words in the text. A lot of input must _see_ most of the input text and most of the language. 

`Homonyms` - words with the same spelling but different meaning. 

Understanding of the context is required to distingush hononyms. 

`Syntactic ambiguity` - situation when a meaning of the word is ambiguous in a sentecne (e.g., who did what in a mixed setnece).  

Transformer architecture solved many issues of RNNs in paper __Attention is All You Need__.  
- Scalable for multicore GPUs
- Parallel processing
- "Pay attention to the meaning of words"


### Transformers architecture

Transformer allows to learn the relevance and context of all words in the sentence. 
An RNN learns a relation between a word and its neighbour, while tranformer learns connections between all words and all neighbours.  
The relationships learned by tranforemr and then scaled by _attention weights_.  
Attention weights are _learned_ during an LLM training.  
`Attention map` is a diagram of attention weights between words in a text.  
`Self-attention` - weight-expressed relation between different words in text. Self-attention significantly increases model ability to encode language. 

Transformer structure: 
- Encoder
- Decoder
Inputs at the bottom; output at the top

> ML models are just big statistical calculators; they work with numbers 

Text must be first `tokenized` before it is passed to the model. 
Each word is assigned a number, a position of this word in a dictioonary of _all possible words_.  
There exists many tokenaztion methods. 
- Token ID for a word
- Token ID for parts of words

The _same tokenizer must be used for training and inference_. 

After tokenization, pass text to `embedding layer`, a trainable vector embedding space -- a high dimensional vector space where each token is represented as a vector and occupies a dinstict space.  

Each token ID in a vocablaru is mapped t oa unique vector. These vectors _learn_ to encode meaning and context of different tokens i nthe input sequence.  

In [Attention is All You Need](https://arxiv.org/abs/1706.03762) the vector space had 512 dimensions.  
By computing _angle between vectors_ in embedding vector space, a mathematical concept of similarity between words can be expressed. 

After embeddin text into token vectors, the _positional information_ is added.  
The model processes each input token _in parallel_. Thus, positional encoding is necessary to _retain information about a position of a word in a sentence_. 

After that you embedded input tokens and positional encodings and pass it into self-attention layer. 
There, model finds the relations between tokens in the input sequence, and capture _contextual dependencies. Self-attention weights reflect the importance of a given word in a sequence to all other words. 

Transformer model uses `multi-headed self-attention` - multiple sets of attention weights are _learned in parallel_, independendly. The number of heads is (12-100). Each attention head _learns a different aspect of the language. Objects, activitis, ryhms... This is not prescribed. Weights are rundomly initialized. Attention maps are hard to interpete sometimes. 

The output of this is processed by a fully-connected feed-forward NN, giving a _vector of logits_ that is proportional to probabilities score for each token in a dictionary.  

The vector of logits is passed into a soft-max layer that gives the vector of probabilities for each word in a dictionary. The highest score tocken -- is the most likely predicted token. 

Transformer Flow:
- Tokenize input (word -> number)
- Embed tokenized input (e.g., Word2Vec) 
- Add positional encoding
- Sum embedded input tokens and positional encodings
- Pass combined input into self-attention layer to lean contxtual dependencies between words. 
- Fully connected feed-forward NN for output, gives vector of logts. 
- Soft-max layer for the vector of logits, to normalize into probability score for each word


### Generating text with transformers

Transformer:

- Encoder - encodes inputs (prompts) with contextual understanding and produces one vector per token.
- Decoder - accceps input token, uses contextual info from encoder, and generates new tokens.

Consider a tranlation task - `sequence to sequencd` task (original objetive of transformer)

1. Tokenize input unsing the same okenizer that was used to train NN. 
2. Pass tokenized input to the encoder on the input side of NN
3. Pass through en embedding layer
4. Feed into multi-head attention layer
5. Output of that are fed into feed-forward NN to the _middle_ of the decoder ( at this point data represents structure and meaning of input sequnce). Insrted at the middle this data _influences_ decoder self-attention mechanism. 
6. A _start of seuence_ token is passed to the input of the decoder. This triggers the decoder to predict the next token in the sequence, for which the contextual understanding passed from the encoder is used. 
7. The output of the decoder self-attention layers are passed into decoder feed-forward NN and through final soft-max output layer. Thus, the _first output token_ is generated. 
8. This token is passed into the _input_ of the decoder again to trigger the next token generation. 
9. Generation is stoped when the model predicts `end of sequnce` token. 
10. Final sequence of tokens is de-tokenized. 

The output of soft-max layer can be used _differently_ to predict the next token (control over creativity of the model). 

Components of the network can be used separately. 
- `Encoder-only` - seq-to-seq, the input and output are the same length. 
    - Usefull for _classification_ tasks, e.g., `sentiment analythsis` (require special output layers)
        - example: BERT
- Encoder-Decoder models - seq-to-seq for where input and output sequents are different size
    - Translation, summarization, text generation
        - BART
        - T5
- `Decoder-only` - very popular and hihgly generalizable
    - Exampls: GPT, BLUOOM, LlaMa. 


### Prompting and prompt engineering

Prompts needs to be carefully crafted for the model to generate desired input. The process of desgining a prompt called `prompt engineering`. 

Usefull techniques: 
- Include example of a desired output (this is called `in-context learning`).  
- `zero-shot inference` - providing inptu data in the prompt, e.g., providing an instruction, context (text to be analyzed), and instruction to produce output (desired form/type).  
    - Large LLMs are very good at sentiment analysis and zero-shot inference.
- `one-shot inference` - providing one example of a _completed task_ (with correct output), a sample of a completed task, and one task to be ocmpleted. 
    - Improves the ability of the model to perform the task
- `few-shot inference` - include several examples of a completed task.
    - Helps smaller models to complete the task

NOTE: context window is limited. Only sso many examples can be provided. 

When even few-shot inference fails, one may try `fine-tuning the model` (additional training with new data). 

Overall, model performance and an ability to generalize depends on the model size.  
Larger models capture more understanding of the language.  
Smaller models are not as good on multi-task work.  


### Generative configuration

There are configurable parameters that control the next-token generation in the model. 
For example, on `hugging face` gives models to AWS `playground section` where several controls are available.

Example:

- __max new tokens__ (limit the number of tokes the model will generate) 
    - The model can stop before reaching this number of _end-of-sequence_ token is generated. 
- __sample top K__ (sampling technique, limit random sampling) - Use only top K tokens with highest probability for random sampling. 
- __sample top P__ (sampleing technique) - limit sampling to the set that has a _combined probability_ <= P (limit to the cimilative probability of the set of predictions). 
- __Temperature__ - controls the shape of the probability distribution the model computes to generate the next token. The higher the temperature the more the randomnes. Temeperature is a _scaling factor_ that is applined the final, soft-max layer of the model. it controls the shape of the output probability distribution. Note. This alters the prediction the model makes (cotrary to Top P and Top K parameters). 
    - Temp = 1 - strongly peaked (sharp) probability dsitribution. The model output than follows the most likely word in the sequence at each prediction timestep. 
    - Temp > 1 - broader, flatter probability distribution and prediction is more random and more variable. 


Each model exposes a set of configurable parameters. These are different from train parameters. These parameters are used at _inference time_. 

Most LLMs use `greedy decoding`, where the model uses a token with the highest probability as a next token in the sequence. In case of large output, however, it is subjected to `repetitions` of words/sequences. 

Solutions:
- `Random sampling` to avoid repetitions, - use random sampling with weights given by initial probability distribution. In some cases this needs to be seelcted manually, e.g., do_sample = true.  However, this may lead to an output being _too creative_. To improve the output consider _Sample top K_ and _Sample top P_ parameters. They _limit random sampling_ and helps output to be more sensible. 


### Generative AI project lifecycle

From cocpetion to launch of a project: infrustructre and difficulties. 

1. __Scope__. Define a sope as accurately and precisely as possible. LLMs abilities depend on the size and training. What tasks are needed -- defines what LLM (size, price) is needed. This has to be specific. 
2. __Select__. Choose an exisiting model or train from scratch. Usually we start with an existing model. 
3. __Adapt and align mode__. Once model is selected it might require ffurther:
    - prompt engineering
    - fine-tuning
    - aligning with human feedback
    to achieve required performance. Evaluation needs to be done to assess the model performance. This is an _iterative process_
4. Application integration
    - _Optimize_ and deploy model for inference (computational efficiency)
    - _Augment model_ and build LLM-powered application (mind the LLMs flaws, e.g., halucination, reasoning, math)


### Intro do Labs

Labs are done using system called _Vocareum_. Access to AWS [Sagemaker](https://aws.amazon.com/sagemaker/) for running notebooks. 
- _Start lab_. 2 hours time limit. 
- First, log out of AWS concol
- Get to SageMaker studio. This is Jupyter-based IDE - Click _Open Launcher_ -> _system terminal_
- Copy Labs from S3 bucket (object store that is in the cloud)


### LAB: Generative AI Use Case: Summarize Dialogue

Resources provided 8 CPUl 32gb RAM  
Libraries
- Tranformers (from hugging face)
- Pytorch 
- datasets (from hugging face)

Note: When a new model is evaluated as a possible candidate: zero-shot, one-shot, few-shot prompt engeneering methods are the _basis_ to understand whether a model is a good fit. 