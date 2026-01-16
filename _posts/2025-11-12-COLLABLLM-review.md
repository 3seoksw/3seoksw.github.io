---
layout: distill
title: COLLABLLM, From Passive Responders to Active Collaborators
date: 2025-11-12
description: Multiturn-aware LLM aiming for long-term goal
tags: LLM
categories: paper-review
giscus_comments: false
related_posts: false
bibliography: pk-yolo.bib
related_publications: true
---

The review is done with the following paper and the figures used for this article are derived from the paper:<br>
[COLLABLLM: From Passive Responders to Active Collaborators](https://arxiv.org/abs/2502.00640) <d-cite key="collabllm"></d-cite>.

## Abstract

COLLABLLM’s motivation and its idea are straightforward and intuitive.

I believe you might also have experienced this.
Imagin you’re using a large language model like Chat-GPT, and you ask some question to the model then you’ve probably gotten a passive answer at first.
Then you had to refine your instruction since the model’s initial answer wasn’t satisfactory and repeat this process multiple times, which is quite frustrating experience and consumes a lot of time.
This largely comes from how most models are trained and optimized for single-turn helpfulness, which encourages passive responses and ultimately limits long-term interaction.
COLLABLLM aims for a more interactive, multiturn-based experience.
To address the aforementioned problems, COLLABLLM introduces two ideas which are Collaborative simulation and a Multiturn-aware Reward which is called MR.
Together, these make the model more proactive and lead to higher task performance and better interactivity across multiple turns.

{% include figure.liquid path="../assets/img/COLLABLLM/comp_collabllm.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 1. Comparisons between existing LLMs and COLLABLLM.
    <d-cite key="collabllm"></d-cite>
</div>

## Overview of COLLABLLM

The following figure shows the overall process of how COLLABLLM works.

For instance, the user gives an instruction to the model asking to write about how optimism can improve our well-being.
Given the context state x, the model outputs a response y from the policy pi.
What differs from other models is that instead of giving the answer right away, it asks more contextual questions like what kind of tone are you aiming for.

The way model is not giving you passive answer, rather asking contextual questions is possible because the model is using multiturn-aware reward function to ensure model is accounting for multiturn.
Also, making the model be aware of future turns is done by using collaborative simulation to see what would the future conversation be like.

Collaborative simulation is to sample future conversations given the context state.
You can think of collaborative simulation as a conversation lookahead method between user and the model.
However, it is impossible to know what would users ask in near future, and even corresponding replies to unknown user's future input.

To make this feasible for enabling lookahead of conversations, a simulated user such as GPT 4o that imitates the actual user generates user's future input.
In collaborative simulation, forward sampling is used to retrieve unknown future dialogue between the user and the model.
Ultimately, this enables the model take future conversation into account and choose responses aligned with a long-term goal, instead of a current-focused goal.

Finally, they apply reinforcement learning fine-tuning such as SFT, PPO, and DPO using the MR function.

{% include figure.liquid path="../assets/img/COLLABLLM/framework.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 2. COLLABLLM Framework.
</div>

## Problem Formulation

- Multiple Turns

$$
\begin{align*}
    & t_j = \{u_j, m_j \}, \text{ at turn } j=1, \cdots, K \\
    & u_j: \text{ user input} \\
    & m_j: \text{ model output} \\
    & t_{1:j} = \{ t_1, \cdots, t_j \} \\
    & t_{j}^{h} = t_{1:j} \cup \{u_j\}
\end{align*}
$$

- Multiturn-aware Reward (MR)

$$
\begin{align*}
    & R^{*}(t_{1:K}|g): \text{ objective given some ground-truth goal} \\
    & MR(m_j | t_j^{h}, g) = \mathbb{E}_{t_j^f\sim P(t_j^f | t_{1:j})}[R^{*}(t_{1:j}\cup t_j^h | g)] \\
    & t_j^f = t_{j+1:K}: \text{ forward trajectory following } j\text{-th turn}
\end{align*}
$$

{% include figure.liquid path="../assets/img/COLLABLLM/MR_simulation.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 3. Multiturn-aware Rewards from collaborative simulation.
</div>

### Conversation-level Reward

$$
R^{*}(t|g) = R_{\text{ext}}(t, g) + R_{\text{int}}(t)
$$

- Extrinsic Reward
  - $$R_{\text{ext}}(t, g) = S(\text{Extract}(t), y_g)$$ evaluates how well the conversation achieves the user's goal $$g$$
  - $$S(\cdot, \cdot)$$ evaluates task-specific metrics (_e.g._, accuracy or similarity)
  - $$\text{Extract(t)}$$ extracts the final response (solution) from the conversation $$t$$
- Intrinsic Reward
  - $$R_{\text{int}}(t) = - min[\lambda \cdot \text{TokenCount}(t), 1] + R_{LLM}(t)$$, comprises penalty and helpfulness
  - $$\lambda$$ controls the penalty for the number of tokens being used
  - $$R_{LLM}$$ evaluates user-valued objectives (_e.g._, engagement or interactivity)

The above uses LLM as a judge in terms of evaluating the intrinsic variables <d-cite key="zheng2023judging"><d-cite>.

### Forward Sampling

$$
t_j^f \sim P(t_j^f | t_{1:j})
$$

- User Simulator
  - Simulator $$U$$ generates a probabilistic distribution $$P(u \mid t)$$
- Sampling Method
  - Naive approach is to use Monte Carlo Sampling $$\rightarrow$$ computationally expensive
  - Introduce a window size $$w$$ (trade objectives for huge cost savings)

$$
t_j^{f_w} = t_{j+1:j+w} \leftrightarrow t_j^f = t_{j+1:K}
$$

## Experiments

### Experimental Setup

COLLABLLM is based on Llama 3.1, and it has four variants.
First two are offline models, supervised fine-tuning model and offline DPO model.
Offline models only use the pre-determined datasets to update the model’s policy network during training.
Then from the first two models, these two are further trained to online models, PPO and online DPO model.
Online models are participating in the simulation to compute new MRs and update the policy network during training.

So the difference between offline models and online models is that offline models do not participate in the simulation and online models do.

Two baseline models will be used in the paper.
One is called based model which is vanila Llama-3.1.
The second one is called proactive base model, and it is a base model with proactive prompt engineering model.
Proactive base model is simply base model given with the prompt as such figure 4 to be more collaborative and interactive.

{% include figure.liquid path="../assets/img/COLLABLLM/MR.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 4. Generating high-quality conversation data with Multiturn-aware Rewards (MR).
</div>

And COLLABLLM is evaluated with the baseline models in three different environment datasets.

First is MediumDocEdit-Chat dataset focusing on document editing sampled from Medium articles.
It is evaluated with BLEU score measuring similarity between the extracted document and the original article.

Second is BigCodeBench-Chat dataset meant for coding assistance.
It is sampled from BigCodeBench dataset and is using pass rate as an evaluation metric.

Finally, MATH-chat dataset is used and it’s sampled from MATH dataset.
The task is evaluated with the accuracy metric.

In addition to the task-specific metrics, two task-agnostic metrics are incorporated, one is average token count and the other is interactivity.

{% include figure.liquid path="../assets/img/COLLABLLM/simulated_env.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 5. Simulated Multiturn Environment Datasets.
</div>

### Results of Simulated Experiments

{% include figure.liquid path="../assets/img/COLLABLLM/tab1.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Table 1. Evaluation results on our multiturn datasets. Green zone: Baselines; Orange zone: Variants of COLLABLLMs. Rel. Improv. indicates the relative improvements of COLLABLLMs trained with Online DPO over Proactive Base. 
</div>

### Ablation Study on Reward Mechanism

{% include figure.liquid path="../assets/img/COLLABLLM/ablation.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 6. Ablation Study of Reward Mechanisms on MediumDocEdit-Chat. This figure compares three immediate reward mechanisms with three MR variants.
</div>

### Real-world User Study

{% include figure.liquid path="../assets/img/COLLABLLM/real-world.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
    Figure 7. Real-world user study includes 201 participants interacting with Base, Proactive Base, and COLLABLLM. (a) document quality (b) overall interaction experience (c) spent time (d) additional assessments every three turns.
</div>

## Conclusion

- Most LLMs make passive and short-sighted output due to single-turn training
- Add a future lookahead
- COLLABLLM introduces collaborative simulator and multiturn-aware reward (MR)
  $$\rightarrow$$ Shows effectiveness, efficiency, engagement throughout extensive simulated and real-world evaluations.
