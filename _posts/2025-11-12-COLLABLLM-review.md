---
layout: post
title: COLLABLLM, From Passive Responders to Active Collaborators
date: 2025-11-12
description: Paper review of COLLABLLM
tags: LLM
categories: paper-review
giscus_comments: false
related_posts: false
related_publications: true
---

The review is done with the following paper and the figures used for this article are derived from the paper:<br>
[COLLABLLM: From Passive Responders to Active Collaborators](https://arxiv.org/abs/2502.00640).

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

### Conversation-level Reward

$$
R^{*}(t|g) = R_{\text{ext}}(t, g) + R_{\text{int}}(t)
$$

- Extrinsic Reward: how well the conversation achieves the user's goal $$g$$
  - $$R_{\text{ext}}(t, g) = S(\text{Extract}(t), y_g)$$
  - $$S(\cdot, \cdot)$$ evaluates task-specific metrics (_e.g._, accuracy or similarity)
  - $$\text{Extract(t)}$$ extracts the final response (solution) from the conversation $$t$$
- Intrinsic Reward:
  - $$R_{\text{int}}(t) = - min[\lambda \cdot \text{TokenCount}(t), 1] + R_{LLM}(t)$$, comprises penalty and helpfulness
  - $$\lambda$$ controls the penalty for the number of tokens being used
  - $$R_{LLM}$$ evaluates user-valued objectives (_e.g._, engagement or interactivity) <d-cite key="zheng2023judging"><d-cite>
