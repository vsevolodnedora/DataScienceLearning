# A beginner’s guide to modern natural language processing  
By Jodie Burchell, Developer Advocate in Data Science at JetBrains

__Evolution stages:__
- Rule based
    - Binary vectorization
    - Count vecotrization
- Neural nets
    - word2vec (all usages of a word in a training data and find what words surround it)
    - LLMs (successor to LSTMs)

__Example__:
- using 
```python
from gensim.models import Word2Vec
```

### Transformer models
- Using positional embedding as weights in addition to the word embeddings themselves. 
- Using self-attention mechansism
- Encoder block is the self-attention + normalization steps
- BERT model (not a generative LLM)
    - Predict an order of sentences 
    - Predict missing word
__Example__:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```
__REPO__: [github](https://github.com/t-redactyl/text-to-vectors)

### Discussion:
- How to deal with ovefitting: 
    - Use validation set (or even a test set as well)
    - Avoid black box, where _feature leacakge_ can become a problem 
- How to avoid overfitting in fine-tuning a pre-trained LLM for new tasks?
    - Unknwon...
- How to handle out-of-vocabluary words
    - Problematic for hard word2vec models
    - Less problematic for large LLMs
    - Main way to deal with them is to use old-school
- Recomendation for entry-level job
    - Do projects and start a blog and write about them (videos, notebooks)

See t-redactyl.io

# Accelerating Python on GPUs
By Paul Graham, Senior Solutions Architect at NVIDIA

- Overview of the GPU applications in academia and industry
- Overview of the GPU achitecture (Streaming multiprocessor)
- CUDA and [cunumeric](https://developer.nvidia.com/cunumeric)

Starting with GPU programming


# Harry Potter and the Elastic Semantic Search
By Iulia Feroli, Senior Developer Advocate at Elastic



# Convert batch code into streaming with Python
Bobur Umurzokov, Developer Advocate


# Links and Sources

[video](https://www.youtube.com/watch?v=WYmyZBg2VFI)