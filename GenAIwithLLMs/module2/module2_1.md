## Week 2

### Parameter efficient fine-tuning (PEFT)

Training the full mode can be xtremely computataionally expensive and requires large amount of memory.  
Solution: `Parameter efficient fine-tuning` (PEFT), technique where most of he model parameters are frozn and only a small set of them is being updated. Or new parameters are added that are trained. Overall, most of the model weights are frozen and number of trained parameters << number of original LLM parameters. This reduces the memory requriement. it is also less prone to catastrophic forgetting. 

`Full fine-tuning` creates _copies_ of original model for each task it is trained for. Thus, if the number of needed tasks is large, storage may become an issue. 

with __FEFT__ only a small number of weights is trained and the memory foot print is therefore smaller. Then, the parameters are combined with the original LLM weights for inference. These parameters can be easily swapt out for inference, -- efficnet adaptation of original model for multiple tasks.

__PEFT trade-ffs__:
- Parameter efficiency
- Memory efficinecy
- Training speed
- Model performance
- Inference cost

__PEFT methods__:
- _Selective_ methods - use subset of _initial_ LLM parameters to fine-tune the model. 
    - Train only certain layers, components, parameter types
    - Significant trado-offs (research is on-going)
- _Reparameterization_ - uses original parameters, but uses a _low-rank transformations_ of the original network weights. 
    - LoRA technique
- _Additive methods_ - keeps original model frozen but adds additional trainable components to the model
    - Adapters methos - adds additional layers to encoder or decoder
    - Soft prompts - keep model weights frozen and manipulates input to achieve better performance (prompt embeddings/retrained prompt encoding system)


### Low-Rank Adaptation of Large Language Models (LoRA)

Reparemtrazation method.  

Recall transformer architecture. Recall Encoder and Decoder compoennts
- Self-attention (attention score cacls.)
- Feed-forward 

1. In LoRA the original model parameters are _frozen_. 
2. Inject __2 rank decomposition matrices__ alongside the original wights. dimensions of these matrxies are such that their product has the same 'shape' as the weights they are modifying. 
3. Freeze the original weights, but train the smaller matricies (typically Rank r is small, 4, 8 ... 64) using _supervised learning_. 
4. For inference, the two low-rank matrices are _multiplied and then added to the original frozen weights_ and replaced in the original model. This model has the same parameter size as the original model. 

It was shown that adding this low-rank matrices to the weights in the self-attention layer is enough to achieve good performance. 


#### Example: transformer achitecture
- Transformer weights have dimensions d * k = 512 * 64 (in original paper). So each matrxi has 32,768 parameters. 
- Using LoRa with rank 8, we add 2 small rank _decomposition matrices_, with small dimension being 8, so matrix A will have r * k = 8 * 64 with 512 parameters and matrix B will be d * r = 512 * 8 so 4,096 parameters. 
- Updating the weights in the original model with these tranable parameters, _86% reduction in trainable parameters is achived_. 

Thus, a PEFT with LoRA is possible on a singe GPU. 

Different sets of LoRA matrices can be trained for different tasks and switched at inference time. 


#### Evaluating LoRA with ROUGE method

Consider FLAN-T5 (initial full fin-tuning done)
1. Compute baseline score (rouge1, rouge2, rougeL, rougeLsum)
2. consider a model with _full fin-tuning_ on dialogue summarization and compute metrics. Note an increased rouge1 score
3. consider LoRA fine-tuned model and examine metrixs. Note that they are _slighly_ lower than the full fine-tuning, but significantly better than the original model. 

Chhosing rank of LoRA matrxies is still an ative area of research. The smaller the rank, the smaller the number of trainable parameters, better compute efficiency. 


### Soft Prompting

