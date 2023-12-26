---
layout: post
title: Bayes' Theorem
date: 2023-09-13 21:30:00-0400
description: Bayes Rule: probability of an event based on prior knowledge
tags: statistics bayes
categories: concepts
giscus_comments: false
related_posts: false
related_publications: 
---

# Bayes' Theorem

**Bayes' Theorem** (or Bayesian Theorem) is a statistical method to update our prior beliefs.
Bayes theorem can be used in various fields such as in Machine Learning (ML) method.

---

### Definition

$$
\begin{align*}
    \text{Posterior} \propto \text{Prior} \times \text{Likelihood} \\
	P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} \\
\end{align*}
$$

### Usage in ML

$$
\begin{align*}
    P(w|\mathcal{D}) \propto P(\mathcal{D}|w) \times P(w) \\
\end{align*}
$$

- $w$: Prior weights for a neural network
- $\mathcal{D}$: Data
- $P(w|\mathcal{D})$: Posterior distribution, a probability distribution of the neural network weights $w$ after observing the data $\mathcal{D}$
- $P(\mathcal{D}|w)$: Likelihood function, represents how well the neural network with parameters $w$ fits the observed data $\mathcal{D}$
- $P(w)$: Prior, prior beliefs about the neural network weights before observing the data $\mathcal{D}$
- $P(y^*|x^*)=\mathbb{E}_{P(w|\mathcal{D})}[P(y^*|x^*, w)]$ At prediction time, the predictive distribution over the target $y^*$ given a test input $x^*$
