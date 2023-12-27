---
layout: page
title: A DRL Approach for Sensor-driven Decision Making and RUL Prediction
description: frameworks for decision making and RUL prediction are suggested
importance: 1
category: DRL
related_publications: drl-sensor
---

The review is done with the following paper:<br>
[Erotokritos Skordilis, et al., "A deep reinforcement learning approach for real-tim sensor-driven decision making and predictive analytics", Jul 2020](https://par.nsf.gov/servlets/purl/10184005)

In this paper, the authors propose two decision making methods that use reinforcement learning (RL) and particle filtering for:

1. Deriving real-time maintenance policies
2. Estimating remaining useful life for sensor-monitored degrading systems

# Before getting started

Prerequisites for the paper:

- Bayesian Filter
- Particle Filter

### [Bayesian Filtering](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation)

Bayesian filter is a general approach for estimating unknown [probability density function (PDF)](https://en.wikipedia.org/wiki/Probability_density_function) recursively.
It is designed for the online state estimation problems.
For more detail, it will be thoroughly discussed in [2.2. Bayesian filtering](2.2.-Bayesian-filtering).

### Particle Filter

TBD

# Table of Contents

- [Introduction](#1.-introduction)
- [System characteristics and monitoring tools](#2.-System-characteristics-and-monitoring-tools)
  - [State-space modeling](#2.1.-state-space-modeling)
  - [Bayesian filtering](#2.2.-bayesian-filtering)
  - DRL?
  - NN?
- [Bayesian filtering-based DRL for maintenance decision making](#3.-bayesian-filtering-based-DRL-for-maintenance-decision-making)
- [Numerical Experiments](#4.-numerical-experiments)

# 1. Introduction

There are multiple advantages using RL frameworks with Bayesian filtering. <br>
Bayesian filtering

1. allows an explicit quantification of uncertainties
   - Standard DRL does not incorporate uncertainty
   - Standard DRL instead, uses point values from the sensor observation
2. can infer the current state based on the history of sensor measurments and the latent degradation process
3. is computationally more appealing
   - Use of its output as input for training a RL framework
     - Since Bayesian filtering can combine complex multi-dimensional sensor data

---

# 2. System characteristics and monitoring tools

Before we get to know more about the proposed approch, we first need to know the followings:

- [State-space modeling](#2.1.-State-space-modeling)
- [Bayesian filtering](#2.2.-bayesian-filtering)
- [Neural Network](#2.3.-neural-network)

### 2.1. State-space modeling

In the first place, what is [state-space model (SSM)](https://en.wikipedia.org/wiki/State-space_representation)?
Since we're dealing with a dynamic system that is degrading latently,
degradation variables (latent variables) can be inferred indirectly from sensor we're trying to use.
That is, sensor observations are the key to find the hidden degradation process.
SSM is a mathematical model that helps us to find the dependencies between the latent variables and the observations. <br>
Most likely, if dealing with healt-monitoring system,
SSM follows a generic stochastic filtering structure in a dynamic state-space form of

$$
\begin{align}
        &x_t = f_x(x_{t-1}, {u_t}, {\theta_x}, \epsilon) &\text{degradtion value} \\
        &{y_t} = f_y(x_t, {u_t}, {\theta_y}, \delta) &\text{observation process}
\end{align}
$$

where $x_t$ denotes one-dimensional continuous and degradation process value at time $t$
and ${y_t}$ denotes multi-dimensional observation process.
Stochastic functions $f$ display the evolution of model variables.
${\Theta} =\{ {\theta_x}, {\theta_y} \}$ are model variables which characterize the behaviour of the corresponding function $f$.
${u}_t$ is a multi-dimensional vector of operational inputs for each problem specified.
During the operations, processes are perturbed by statistical noise $\epsilon$ and $\delta$.
The noises $\epsilon$ and $\delta$ follow gaussian distribution and have 0 means and have convariance matrices of $Q$ and ${R}$ as follows:

$$
\begin{aligned}
    \epsilon &\sim \mathcal{N}(0, Q) \\
    \delta &\sim \mathcal{N}(0, {R})
\end{aligned}
$$

Despite the above, SSM cannot capture multimodal hidden dynamics which considers discrete-valued processes.
Hence hybrid state-space model (HSSM) has been developed in order to monitor remaining useful lifetime (RUL).
Just like SSM, HSSM includes continuous degradation state process $x_t$
and observation processes ${y_t}$.
Along with $x_t$ and ${y_t}$ as presented in $Eq. (1)-(2)$, there also exist,
an operating condition of the system $c_t$ (i.e., normal or faulty),
a latent hazard process $\lambda_t$ representing the system's probability of failure,
and the working status process $o_t$ representing the overall system's working conditions.

$$
\begin{align}
    &x_t = f_x(x_{t-1}, c_t, {u}_t, {\theta_x}) \\
    &{y}_t = f_y(x_t, c_t, {u}_t, {\theta_y}) \\
    &c_t = f_c(c_{t-1}, {u}_t, {\theta_{c}}) &\text{latent process stating normal or faulty}\\
    &\lambda_t = f_\lambda(x_t, c_t, {u}_t, {\theta_{\lambda}}) &\text{hazard process showing system's probability of failure}\\
    &o_t = f_o(\lambda_t, {\theta_o}) &\text{overall system's working conditions}
\end{align}
$$

As mentioned earlier, finding the relationship between latent variable and observation is no easy work.
Since latent variable can be inferred indirectly from the observation, modeling a function showing the dependencies would be promising.
Here, we can use a single-layered feedforward neural network namely Extreme Learning Machine (ELM).
ELM can be used to model high dimensional and correlated sensor measurments. <br>
From $Eq. (1)-(7)$, each equations need proper function to describe the dependencies between the input variables.
ELM's job is to find the proper function for each equation and every ELM will have its own weight vector $\theta$.
For instance, when finding a function $f_x$, ELM for $x$ is to find appropriate weights $\theta_x$
in order to show the dependencies between the input variables $x_{t-1}, c_t$, and ${u_t}$.

Latent hazard process $\lambda_t$ shows the probability of system's failure with the input of $x_t, c_t$, and ${u_t}$.
Hazard process is in the form of logistic regression function under the binary assumption that is, normal or faulty as follows:

$$
\begin{aligned}
    \lambda_t &= [1 + \text{exp}[-(\alpha x_t + \beta c_t + {\gamma u_t} + \beta_0)]]^{-1} \\
    &= \frac{1}{1 + e^{-(\alpha x_t + \beta c_t + {\gamma u_t} + \beta_0)}}
\end{aligned}
$$

<img src="../public/struct-SSM-sensor.png">

**Fig. 1.** Structure of the proposed SSM. <br>

- Circles: latent states <br>
- Rectangles: Observations <br>
- Diamonds: Operating inputs

### 2.2. Bayesian filtering

Finding the dependencies between the given latent states, $(x_t, c_t, \lambda_t)$, is no easy work.
Now for the sake of convenience of notation, let's say the latents states as ${z_t} = \{x_t, c_t, \lambda_t \}$.
Here, we are going to use Bayesian filter to approximate the posterior distribution.
But you might wonder, what is Bayesian filter anyway?
You can reference the following and help yourself more.

Bayesian Filter

> The purpose of Bayesian filter is to calculate the belief of the hypothesis given the evidence
> and update
>
> $$
> P(H | E) = \frac{P(H) P(E|H)}{P(E)}
> $$
>
> $H$: Hypothesis <br> > $E$: Evidence <br> > $P(H|E)$: Posterior <br> > $P(H)$: Prior <br> > $P(E|H)$: Likelihood <br>

### 2.3. Neural network

---

# 3. Bayesian filtering-based DRL for maintenance decision making

### 3.1. From Bayesian filtering to DRL

## **Algorithm 1.** Bayesian Filtering-based Deep Reinforcement Learning

Input: Full

$$
T = \sum_{j=1}^{N}{T_j}, j \in \{1, 2, ..., N \} \text{Sum of all lifetimes in $N$}
$$

$$
\sum_{i=1}^{N_s}{1}_{z_t^{i}\in(b_d, b_{d-1})}
$$

# 4. Numerical Experiments

CMAPSS (Commercial Modular Aero-Propulsion System Simulator) turbofan engine degradation dataset from the NASA diagnostic repository
([Saxena & Goebel, 2008](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf))
