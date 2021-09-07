# classification augmentation by reinforcement learning



## summary

we train Proximal Policy Optimization(PPO) to write continuation for the prompts(original samples), and collect
those continuation texts as augmented sampled for text classification augmentation, in a relatively low-data regime.


## baseline

EDA, back-translation, CBERT

## base classifier

Albert 


## finetune LMs

Follow the examples from huggingface: https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation

Fine tune the original GPT2 to the supervised texts and/or external unsupervised texts


## PPO training for gpt2

This follows the language model approach proposed in paper "Fine-Tuning Language Models from Human Preferences"
Some implementations are from https://lvwerra.github.io/trl/
We train policy based on the continuations and feedbacks(rewards)


## zero-shot learning

You can refer to this blog for more info: https://joeddav.github.io/blog/2020/05/29/ZSL.html 
(This check can extremely boost the accuracy performance in next training step)

Therefore, we can train classifier based on these synthetic samples. It is surprising to find that the performance is quite satisfactory.













