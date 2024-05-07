
# Week 2

Methods to imporve prformance of existing models.  
Evaluation of a fine-tuned LLMs and quantify its performance agains base model

## Fine-tuning LLM with instruction

### Intdoduction

Instruction tuning  
In a given pre-trained model information is encoded, but there is no guadance on what to do with user queery. Model has to be tuned for a specific task.  
E.g., next word prediction is not the same as answering questions.  

Mind for _catastrophic fogetting_, when model forgets train data during fine-tuning.  

There are two types of fine-tuning: 
- Extraction fine-tuning
- Parameter-efficient fine-tuning (aciheve good performance, using less mmory)
    - freezing initial weights
    - adding extra trainable layers
    - LoRA (low range matricies for fine-tuning)

Prompting has a cieling and fine-tuning allows to achive even better performance.  

Fine-tining in an efficient way


### Instruction fine-tuning

Smaller LLMs may not be capable of identifying the task in zero-shot inference.  
One-or-more-shot inference may be enough for smaller models.  

Drawbacks
- Not always work for smaller models
- Context space is limited by examples

Solution:
- Fine-tuing the model (additional training)

`Fine-tining` is a superwised learning (contrary to pre-train, that is self-supervised), where a dataset of labeled examples is provided.  
Train data set is a Prompt - Completion pair set.  

`Instruction fine-tuning` trains the model using examples how a model should respond to a given queury. For examples, consider a set of pairs of a type:

```bash
Classify this text 
...text...
Sentiment: Positive
```

Thus model learns that it has to answer with "sentiment: positive or negative". 

For summarization it is similar:

```bash
Summarise the following text 
[example text]
[example summary]
```

Thus model learns to generate proper otuput given a certain prompt.  

`Full-fine-tuning` is an instruction fine-tuning where _all model weights_ are updated.
It requires compute and memory needed for the full model required in model training.  

__Steps__:
- Preapre training data (public datasets but not always formated, but there are prompt-template libraries, that allow to create such datasets)
    - using template that has a space for text body and score/label to be associated with
    - Devide data into _train_, _validation_ and _test_ sets
    - Pass prompt to LLM, get completion, compare it with label from the dataset. 
    - Note, LLM output is a _probability distribution_ across tockens. Thus, cross-entropy function between two token distributions can be used to compute loss and thus the difference between two distributions: predicted and expected. 
    - calulatied loss than is used to update weights via back-propagation. 
    - This is done foe many batches of prompt-completion pairs and over several epochs. 
    - Validation test can be used to compute validation accuracy 
    - Test set can be used to obtain test accucary

`Instruct model` is the output of the fine-tuning process of LLM. Instruction fine-tuning is the most common way to fine-tune model.


### Fine-tuning on a single task

If an LLM has only one task to perform, e.g., summarization or translation; than the model can be fine-tuned on this specific task.   
Notably, few 100-1000 examples sometimes are enough to achieve good results.  

Downside: - Catastrophic forgetting - when model losses traned data. This happens becase _all wieghts_ of the model are updated in the fine-tuning. This may degrade model performance on other, not-fine-tuned-for tasks.  

Solutions: 
- Decide if it is important. If only one task is needed, than lost of generalization is not an issue.
- Fine-tuning on _multiple tasks_. This requires much more examples, compute and time. 
- `Parameter-Efficient tununing (PEFT)` this preservs weights of the original LLM and updates only weights of newly added `adapter layers` and parameters. This is more robust to the problem.  



### Multi-task instruction fine-tuning

Extension of a single-task fine-uning. 
Here dataset contains examples of multiple tasks. Examples:
- Summarization
- Named-entity recognition 
- Translation
- Coding

Here< performance of all tasks must be improved simultaneously. This should help with _catastrophic forgetting_. Over many epocks of training, _calculated loss across examples_ is used to update weights of the model. Thus results in a model tuned for many different tasks.  

Drawback: 
- Requires a lot of data (50-100)*1000 examples of eah task.  

Example of a model that was trained using multi-task instruction fine-tuning:
- [FLAN](https://arxiv.org/abs/2210.11416) (fine-tune language net) (last step of the training process)
    - FLAN-T5 (473 datasets acorss large, 146 task categories) incl. 
    - FLAN-PALM (540B model finetuned on 1836 tasks incorporating Chain-of-Thought Reasoning data)

#### Fine-tuning of FLAN-T5:  
__SAMSum__ dataset used to summarise dialoge. Designed by linguests. This is a high quality dataset. Prompt templates consisted of several types of the same summarization task in a form "dialogie - summary". Note: there may still be limits in FLAN-T5 performance.  
If, for example, text is field-specific, model will not perform well.  
Additional _fine-tuning_ will be required with more data. We will consider __diaologsum__ dataset to further fine-tune FLAN-T5. This will be done for _support-chat conversation_.  
Dataset includes 13,000 support chat dialogues and summaries. 


### Model evaluation

Metrics to evaluate model performance.  
In traditional ML, there is `accuracy = correct predictions / total predictions`, where output is _already known_. There models are _determenistics_. For _non-deterministic_ models it is very difficult, as similarity between different pices of text is subjective.  

Widely used metrics:
- `ROUGE` (rcall orientated under study for gesting evaluation). Used for 
    - Text summarization
    - compares summary to one of more refernce summaries. 
- `BLEU SCORE` (bi-lingual evaluaion under-study)
    - Designed to evaluate qualtiy of macine generated translation text 


#### ROUGE scoure 

Terms and definitions:
- `unigram` in linguistics - singl word. 
- `bigram` is a pair of words (sub-sequence)
- `n-gram` is a set, sub-sequence of words in a sentence. 

`ROUGE-1 Recall = unigram matches / unigrams in reference` - The recall metric measures the number of words or unigrams that are matched between the reference and the generated output divided by the number of words or unigrams in the reference. (Max is 1)

`ROUGE-1 Precision = unigram matches / unigrams in output` 

`ROUGE-1 F1 score = 2 * precision * recall / (precision + recall)` - harmonic mean of both. 

Hhere only one word is considered at a time (hence 1 in the name); order is not considered. 

Consider now the longst common subsequnce in the generated and given text. If length is L, than consier `ROUGE-L score` with recall, precision and F1 defined as 
- ROUGE-1 Recall = LCS(Gen,Ref)/ unigrams in the text;
- ROUGE-1 Precision = LCS(Gen,Ref)/ unigrams in the output;
- ROUGE-1 F1 = 2 * Precision * Recall / (Precision + recall)

Note, same tasks here hust be compared in terms of ROUGE scores. However, bad completion may still result in a good score, e.g., the output may be non-sesical and still give a good tasks. This is possible when words are repeated or words are in a wrong order. 

Possible solutions:
- ROUGE clipping - limit the number of unigram matches to avoid repetition giving high score. 


#### BLEU score
Used to quantify the qualty of the translation.  
Computed via average _precision_ over _multiple engram sizes_.  
`bleu metric = Avg(precision across range of n-gram sizes)`  
How many engrams in the output matches in the reference text.  

To compute it, use existing libraries like hugging face.  

Note: both metrics are quite simple and cheap to compute. Just as _diagnostics_ during the model alterations/fine-tuning. 

They are not good for final evaluation. 


### Evaluation Benghmarks

Seleting the right evaluation dataset is important.  

Datasets should isolate speific model skills that can be evaluated.  

Model must not have seen the data in training.  

_Examples:_
- __GLUE__ - (general language understanding evaluation) Colection of NLP tasks. Used for analysis of model generalized performacne
- __SuperGLUE__ - successor of GLUE addresses many limitations. More tasks and challenging tasks. Tasks like multi-sentice reasoning and reading comprehention. They have leaderboards. 
- __HELM__ - Holistic evaluation of Language Models; improves transparency of models, and uses multi-metrs approach; measures 7 metrics across 16 scenarious; assess on metrics beyond basic metrics:
    - Accuracy
    - Calibration
    - Robustness
    - Fairness
    - Bias
    - Toxicity 
    - Efficiency
- __MMLU__ (massive Multitask Language Understanding) 2021; designed for modern LLM, model must have world knowledge and problem-solving ability; many subjects and reasoning
- __BIG-bench__ - 2022 with 204 task with many tasks, incl. reasonign, physics, etc. comes with three sizes

There is arm-race between emerging properties of LLMs and human performance _on benchmark tasks_. 



