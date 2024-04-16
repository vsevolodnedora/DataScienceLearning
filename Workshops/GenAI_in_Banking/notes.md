# Generative AI in Banking

talk by Max Sommerfeld (Detsche Bank)

Banking does not have a physical business model.  
Product of banking is a contract.  
In finance it is easier to deploy AI as the business is data.  

### Complexities

Risk managment; high cost of error.  
Complexity of banking environemnt.  
Regulations and compliances.  
Detsche bank is partneded with google cloud.  

### Overview of use-cases of AI in banking
- Document processing
- Customer suport

! LLMs are information transformation machines

As internet allows to easily transport information from A to B, 
LLMs allow to _transform_ information; i.e., convert general information into a specific answer.   
LLMs are not good at creating information.  

### Classes of use cases:
- Information extraction (from large non-structured set of documents)
- Conversational AI (interactive information retrieval) 
- Content creation (transforming information into required from preserving the content)
- Code generation (code explanation, documentation generation; best practicies fetching)
- Unstructured to structured (converting unstractured qeury into a structured; format-specific query)
- Emerging use-cases (collaborating autonomous agents to solve a problem)
    - Context retireval can be a problem; -- a model may lack all the context

### Research use-cases

E.g. The AI outlook for 2024

#### Workflow of a research project

- Searh and prepare data
- Define content objectives
- Summarize, prodct content
- Review and edid content in a specific format 
- Manually review the outcome, adjust, add re-write
- Publish and communica
- Close the request

#### Coding with AI
- GitHub Copilot
- DuetAI (from Google)

### Challenges of LLM 
- Rapid development of tools like LangChain that requires continous update and refactoring; this makes deployment tricky
- Choosing the best LLM for a task (see chatbot arena lieaderbord - a hugging face space); No clear winnder; differnt pricing; licence compliance

### Qustin notes:  
DB approach: prompt engeneering with large LLMs; fine-tuning small LLMs; but no training LLMs from scratch.  
Multimodal models seems to be a prospect for a future when analyzing documents


# RAG

Perform a search before passing question to LLM to augment the prompt with information relevant to the question; 
RAG is realtively siple to write.  

### Problems
- Retrieval (search is not easy); One-word search yeilds too many results (use sematic; elastic ... searches). 
- Getting contextual information (when to limit context; Chunking)
    - Solution: use LLM to extract context
- The needle in a haystack problem (LLM ability to find information in large corpi of information)
- False or incomplete results: result checking especially when answer contains information that is not in the soruce
- Sequrity issues: (prompt injection; privte/dangerous information extraction)
    - Detsche Bank approaches this with the following: The key is Redundancy. All critical systems should have redundancy
    - Redundancy must be comprised of independent systems e.g., regex and LLM (with chain of thought) or language models specifically trained to find malicious information

Ovearll, ground truth is needed to evaluate an LLM or LLM + RAG

### Future research areas
- LLMs and ecosstems (stabilization of ecosystem e.g., LangChain)
- Model-agnostic prompts/chains
- RAG (document igestions/prepartion)
- RAG search and retrieval
- RAG Retrieval (needle in haystack)
- Observability (where things went wrong)
- Guardrailing LLMs

