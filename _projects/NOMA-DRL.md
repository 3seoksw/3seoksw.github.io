---
layout: page
title: Downlink NOMA System with Deep Reinforcement Learning
date: 2025-01-05
description: Application of custom REINFORCE algorithm to downlink NOMA system
img: assets/img/NOMA/NOMA_paper_preview.png
importance: 1
category: Personal
related_publications: false
---

[This research](https://github.com/3seoksw/Downlink-NOMA-with-RL) is published in [The Journal of Korean Institute of Communication and Information Scienes (JKICS)](https://journal.kics.or.kr/) in March 2025.
If you're interested, please check out the [paper](https://journal.kics.or.kr/digital-library/102248).

---

## Brief Introduction to The Project

As the growth of the expansion of the internet of things (IoT), the issue with a scarcity of network resources arised.
In order for wireless network (communication) to be successfully accomplished and fully utilized, optimizing the use of the resources is a key to it.
There are numerous aspects to consider when managing networking resources, such as user fairness, quality of service (QoS), and throughput efficiency.

In recent years, the [Non-Orthogonal Multiple Access (NOMA)](https://3seoksw.github.io/blog/2024/NOMA-background/) system has been considered as a promising candidate for a multiple access framework due to its performance allowing multiple users to access to channels at the same time.
The NOMA system involves two fundamental problems: channel assignment and power allocation.
While the NOMA system is considered to be a promising technique for communication, it has a few limitations as the resource allocation problem, which involves channel assignment and power allocation, is considered to be [NP-hard](https://en.wikipedia.org/wiki/NP-hardness).

In this project, in order to overcome the NP-hardness of the problem, **Deep Reinforcement Learning (DRL)** is used when finding an optimal channel assignment.
By applying DRL techniques to the project, the agent successfully managed finite and scarce networking resources efficiently, achieving near-optimal throughput.

## Background

In this page, background knowledge of wireless networking systems will not be covered.
Instead, please refer to the following page:

- [Non-Orthogonal Multiple Access (NOMA)](https://3seoksw.github.io/blog/2025/NOMA-background/)

## System Model

In this project, we assume downlink NOMA system where the base station (BS) sends data to multiple users in a wireless channels.
As the eseential of the project is using NOMA technique, we first need to justify the problem.
Simply put, NOMA can be divided into two key subproblems: channel assignment and power allocation.
While serveral works have proposed to tackle these problems, there exists an approach of the optimization of power allocation known as joint resource allocation (JRA) method[[1]].

<div align="center">
    {% include figure.liquid path="../assets/img/NOMA/system_model.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
</div>

<p align="center">
    <i>Figure 1.</i>
    Transmission of BS and reception of users of the downlink NOMA system
</p>

## Reference

<a id="1" href="https://ieeexplore.ieee.org/document/8790780">[1]</a>
C. He, Y. Hu, Y. Chen and B. Zeng, "Joint Power Allocation and Channel Assignment for NOMA With Deep Reinforcement Learning," in
<i>IEEE Journal on Selected Areas in Communications</i>,
vol. 37, no. 10, pp. 2200-2210, Oct. 2019, doi: 10.1109/JSAC.2019.2933762.
