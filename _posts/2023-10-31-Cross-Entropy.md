---
layout: post
title: Cross-Entropy
date: 2023-10-31 21:30:00-0400
description: A method to measure the difference between two different probability distributions
tags: statistics bayes
categories: concepts
giscus_comments: false
related_posts: false
related_publications:
---

The **Cross-Entropy** between two probability distributions $$p$$ and $$q$$ measures the difference for a given random variable or set of events.

### Definition

The **Cross-entropy** of the distribution $$q$$ relative to a distribution $$p$$ over a given set is defined as follows:

$$
H(p, q) = -\mathbb{E}_p[\log q]
$$

where $$\mathbb{E}_p[\cdot]$$ is the expected value operator with respect to the distribution $$p$$.
The definition may be formulated using the [Kullback-Leibler Divergence](https://3seoksw.github.io/blog/2023/KLD/) $$KL[p \mid\mid q]$$.

$$
H(p, q) = H(p) + KL[p \mid\mid q]
$$

For discrete probability distributions:

$$
H(P, Q) = -\sum_{x\in\mathcal{X}}p(x*i)\log{q(x_i)}
$$

For continuous probability distributions:

$$
H(p, q) = -\int*{\mathcal{X}}P(x)\log{Q(x)}dr(x)
$$
