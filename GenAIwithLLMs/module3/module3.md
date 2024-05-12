# Reinforcment learning with human feedback (RLHf)

RLHF allows to allign LLM with human values; avoid generating harmful text. 

### Aligning models with human values

Recall GenAI project lifecycle.  
W already examined fine-tuning of the LLM, which allows t oget a better performing model for a given task. 

Now we need to make sure that the model does not use
- Toxic language
- Aggressive void
- Dangerous information 

Sometines model completion is just not what is required.  
Sometimes model gives stright up wrong aswer, or harmful answer. 

Morel must have `HHH`: 
- Helpfullness
- Honesty
- Harmlessness

This _allignmant of the model wih human values_ is the last step of _Adapt and align model_ of the _GenAI project lifecycle_. 


### Reinforcement learning from human feedback (RLHF)

Consider text summarization task.  

Finetyming will improve the ability of the model to summarize.  
Research showed that _fine-tuning with human feedback_ gives __better permforming model__ than _initial fine-tuning_. 

`RLHF` is the main method to fine-tune model. There reinforment learning (RL) is used to fine-tune LLM with human feed-back data. This gives _human-aligned LLM_.  
_This gives model that_:   
- maximizes helpfulness relevance
- minimizes harm
- avoid dangerous topic
- personalization of LLM

#### RL

RL is a type of machine learning where an _agent_ learns to make decisions related to a specific _goal_ by taking _actions_ in an _environment_ with the objective of _maximze revard_ recieved for actions. 

Agent continously learns by taking actions and observing changes in the envirnment.  

This is iterative proces, that leads to agent learning an optimal strategy of taking actions or _policy_. 

Agent -> [make actions] -> Environment; 
[Reward & new state] -> Agent;

__Example__:  
train model to play Tic-Tac-Toe.  
- _Objective_: Wind the game
- _Agent_: player
- _Environment_: 3x3 game board
- _State_: card configuration of the board
- _Action space_ includes all possible positions a player can chose based on the current board state
- _RL policity_: stategy that agent follows to make decisions; take actions
- _rewards_ results of agent's actions

__GOAL__: agnet must learn optimal policy for given envirnment that optmizes the reward via an iterative process. 

`Playout/Rollout` a sereis of actions and new states whil an agent is learning.

Agent is gradually learning policy; taking initially random actions; that optimzes the _long-term_ reward. 


#### LLM finetiming with RL

__In LLM case__: 
- Agent RL policy = _LLM_
- Objetive: _generate alinged text_
- Envirnment = _Context window of the model_
- Envirnoment state = _current context_ (text at a given time in the context window)
- Action = _act of generating text_
- Action space = _tocken vocabulary_ (all tockes that a model can chose from to generate a completion)

Statistical representation of the language, learned by LLM in training, guides LLM generation of the next token in a sequence. 

At each time, generation of the next token depends on:
- prompt text in the context
- probability distribution over the vocabluary space

Reward is based on how closly the completion is aligned with human preferences. 

Determing the reward is complicated. There are many criteria, for example: txic or non-toxic (encoded as 0 or 1).  
LLM weights are then ipdated iteratively to _maximize_ the reward obained from the human classifier. 

In practice, an _additional, reward model_ is used to generate reward for an LLM. This model generats the _degree of alignment_ of the LLM output to act as a reward.

This _reward model_ is usually trained via supervised learning using human examples.

In the context of LLMs, sequence of ations is called `rollout`. 







