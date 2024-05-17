# Reinforcment learning with human feedback (RLHf)

### RLHF: Reward hacking

Recap:
RLHF - fine-tuning process to allign LLM with human values.  
Use _reward model_ to assess model completions agains human preference metric. 
Use _RL_ aka PPO to update weights of LLM.  
Use multi-iteration cycle. Untill the desired degree of alignment is achieved. 

In RL there exists `reward hacking` where an agent learns to _chear_ to facour actions that maximize the reward but _do not follow_ original objective.  
_Example_: addition of workds to completion that increase the score for the metric that is being used for alignment.  

#### Avoiding reward hacking

Consider a _reference model_ that has weights frozen and is used to evaluate the output of the RL-updated model. Then, at each iterations compare the completions of the reference and aligned model using _KL Divergence_ (shift penalty).  
`KL (Kullback-Leibler) Divergence` is a statistical measure of hau different two statistical distributions are. It allows to see how far the updated model has diverged from the reference (The entire LLM is used for computing it).  
KL-Dirvergence is often used in RL, especially with PPO, where it helps guiding optimization process to ensuire that the updated policy does not deviate too much from the original one.  
In PPO the ipdate is _iterative_ to ensure stability. Constraints, enforced by KL-Divergence are used to ensire that the iterations are small. 

KL divergence is calculated _for each tocken_ in the vocabluary of the LLM. Using the _soft-max_ the probability is smaller than the whole vocabluary size. This is still compute expensive. After, computing, it is added as an extra term to the reward function. This penalizes the model if it shifts too far from the original one. 

__NOTE__: two full copies of the LLM are required for this task.  

__NOTE__: RLHF can be combined with PEFT; then _only_ the PEFT adapter weights are changed, not of the full LLM. 

This allows us to re-use LLM for both tasks and to have just one LLM. 

To evaluiate the final model, _Summarization Dataset_ can be used to assess the model perforamce using the number, e.g., _toxicity score_. 
1. Create a _baseline_ toxicity score for an original LLM by evaluating its completions of the summaization dataset with the reward model that can assess toxic language
2. Evaluate human allinged model on the same dataset. 
3. Compare toxicity scores. 


### Scaling human feedback

Reward model requires a lot of huma-labelled data to be trained. (1000+ people an 10+ propmpts for each). This is expansive to gather. Human effort is a limited resource. 
Solutions: _Scaling through model self-supervision_.  
- `Consittutional AI` - Training a reward model using _set of rules_ in principles that guven the model behaviour. Train the model to self-critque and revise its responses to comply with these principles. 

__NOTE__: it is also usefull It is usefull with scaling data and address other limitations of alligned LLM.  
An alligned LLM may provide _harmful information_ as it tries to anaser the quations as best as it can. E.g., a user politely asked to learn to jailbreak and the model alligned for helpfulness does help. A model with _consitutional principles_ can balance competing interests and minimize the harm. 

Training Consitutional AI model has two stages
- Supervised model: prompt the model trying to generate harmfull responses: `Red Teaming`. 
- Ask the model to critique its own harmfull responses; revise them to comply with the rules. 
- Fine-tune the model provided Parir of prompts with harmfull and constitutional responses using RL. 
This process sometimes called `Reinforcment learning with AI feedback (RLAIF)`

At the end, the new, _Reward model_ is trained that can later be used to further align the original fine-tuned LLM to get Consitutional aligned LLM


### Lab 3 walkthrough

