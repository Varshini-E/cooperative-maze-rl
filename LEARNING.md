## Trying to debug why SFT/RL runs degraded performance. 

1. Learning rate was too low. Lower than had been recommended by Tinker. Experiments showed middle ground is better: tested with overfitting. 
2. Size of the log-prob changes fluctuated too much - not significant enough and often caused collapse. 
3. Length of episodes played an unfair role here: failed episodes were longer while successful episodes were smaller. We should normalize the advantage by the token count to ensure that episodes are weighted equally. Per-episode rather than per-token basis. 

## How to debug the training pipelines?
1. Always start with a small overfitting run to check the plumbing. 
2. For instance, in the case of RL here, there are uncertainties from multiple sources: sampling, the environment, interaction and evaluation. The goal with overfitting is to take repeated optimization steps on the same set of episodes to see if it picks up the training signals.
3. Latest run: RL performs well on the tasks trained on and improves but does not generalize to other settings. 
4. Tried oracle Expert and Operator: nice way to check the liabilities in the pipeline and ideal outcomes.

## RL Training Choices
1. Advantage computation: group mean in GRPO vs learned value function in PPO.
2. Off-policy correction: how off-policy is the algorithm? We consider a ratio in importance sampling; clipped ratios in PPO and CISPO; one ratio per sequence in GSPO. 
3. Current state is simple importance sampling. Need to evaluate how much differing the RL algorithm would help. 
