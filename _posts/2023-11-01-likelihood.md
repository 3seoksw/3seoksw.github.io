---
layout: post
title: Likelihood Function
date: 2023-09-13 21:30:00-0400
description: likelihood function
tags: statistics bayes
categories: concepts
giscus_comments: false
related_posts: false
related_publications:
---

# Likelihood Function

The likelihood function is the joint probability of the given data (say $x$) viewed as a function.

### Definition

$$
\begin{align*}
	& L(\theta|x_1, x_2, ..., x_n) \\
	&= \text{joint pmf/pdf of random variables } x_1, x_2, ..., x_n \text{ from } \theta \\
	 &= f(x_1, x_2, ..., x_n|\theta) \\
	 &= f(x_1|\theta) \times f(x_2|\theta) \times ... \times f(x_n|\theta) \\
	 &= \prod_{i=1}^{n}f(x_i|\theta) \\
\end{align*}
$$

### Log-Likelihood Function

Plugging the **Likelihood function** into a logarithm shows as follows:

$$
\begin{align*}
	&l(\theta|x_1, x_2, ..., x_n) \\
    &= log(L(\theta|x_1, x_2, ..., x_n)) \\
    &= log(\prod_{i=1}^{n}f(x_i|\theta)) \\
    &= \sum_{i=1}^{n}{log{f(x_i|\theta)}}
\end{align*}
$$
