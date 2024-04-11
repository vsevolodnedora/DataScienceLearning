# Talk 1: Potential of RAG

## Enhancing LLMs
- Prompting 
- RAG (with vector data bases)
- PEFT (paramter efficient fine-tuning) Requries high quality datasetds and compute

Fine-tuning can be replaced by using an adaptor that is more computationally easy.  
RAG requires a pipeline. 

System prompt (coaching prompt that sets model behaviour)

Most open source models have training data cut ff at june 2021 and they are trained to translate not to give the most accurate anaswer.  

### RAG 

Choices for retrieval
- model and its settings (temp.)
- prompt engineering (set the behaviour)
- use rules to process queiries (Person Identifier Information). e.g., guardrails (no private data goes to the public model)
- how responses are displayed

Consider a vector database that stores the information regarding a company (build from external materals)
- Documents are prepped as chanks of texts (via paragraphs, etc). Splitting of the data is required. Complex data preprocessing
- Convert chanks into embeddings -- tokenezation of the prepared data. Tockes are then comverted into embeddings

Consider service prompt that says:
"Use following pieces from the vector database"... 
Retriaval from DataBase (using hybrid search where the infrmation is retrieve from the database and then added to the context window of the llm, so it can processes it and augment the answer).  
The more precise information retrieval the higher the cost. Hybrid search out of the box allows to metigate the problem.  

Developer control for tuning RAG
- Choosing what document to put into vector DB
- how to prep. data for RAD
- how to embed data (word2vec etc...)
- how retrieval is done (key words...)
- which model to use, how big the ocntext window should be
- how to display the info, e.g., provide links

### Application

Take documents related to the product, create a vector DB, augment the RAD, use the LLM + RAG when questions related to the product are asked. 

Presentation by Tjerk Houweling.

NOTE: putting all data into context window cannot compensate for RAG. Errors with long context window. Advanced retiriaval may be more accurate.  
Perhaps the combination and depending on the use case a combination of adding info to the prompt and RAG is the best.  

### Problems

- Data privacy. If an augmented prompt is sent to the OpenAI this violates european data privacy law.  
    - Cloud proviers must be choisen carefully
- Stability. Using chasing API might lead to incosistent results over time. 



# Talk 2: Bulding Advanced RAG with Weaviate

by Philip Vollet

- Weaviate is an open source vector DB
- Easy to use
- GraphQL API 

MultiTenancy?

Product Quantization? 

## Optimization of RAG pipelines 

1. Pre-retrieval (indexing) - giving data to vector DB
    - Chanking overlap. Improves retrieval (language is connected; adding layers helps continouity)
    - Chanking by every two lines with one line overlap 
    - Generating metadata at chank generation for later filtering
    - self-querying retrieval (using natural language to generate filter for data (e.g., SQL quiry))
    - RAG fusion - Genate similar queries related to paranet query, add original one, fetch data for all of them, do reciprocal fusion re-ranking - geneartive output
    - Query splitting (split complex queries into sub-queiries) summarise all answers. This is done via in-context learning. 
    - QUery planning (split quieries based on the task) at the same time and combine the answers
    - Entropy at the chanking level; add more information to the context window from data base that is in the neighbourhood of the exact retived information. Not always improves the answer. 
    - Semantic Cachining (casch the answer based on its sematic relationship and return it if asked again) to improve the computational efficiently. 
    - Summarize Documents when generating vector embeddings. This is done context-aware. 
    - Include conversation as context. 
    - Reranking (model in the loop with transformer to rerank outputs from different search queeries). High computational cost. If the answer is not good, it can be good. Based on cohere re-ranker. 
        - Vombine BM25 search and Vector search (all implemented in weaviate)
    - Using Agents to do RAG (agent makes the decision which part of the pipeline to engage for a user query). 
    - ColBERT & tradititionam BM25 text search. 
        - ColBERT is very optimized and fast. It allws to perform fast reranking of the answers
    - DSPy allows to do a chain of thought querying; It is a pipeline building with LLMs (an improvement on LangChain). Generating a non-hard-coded prompts. Use a signature extracted from a harnd-written prompt for an output structure. This makes pipeline more robust. It can adapt. 
        - Initial prompt -> Compiler -> Optimized prompt. 

## Questions for RAG for equations:
- Use a model to convert PDF or image to latex and than build a RAG with latex representations.
