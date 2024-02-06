---
layout: page
title: Dynaformer, A Deep Learning Model for Ageing-aware Battery Discharge Prediction
description: Transformer-based Battery EoD Prediction Model
importance: 3
category: Transformer
related_publications: biggio2022dynaformer
---

The review is done with the following paper:<br>
[Luca Biggio, Tommaso Bendinelli, Cheta Kulkarni, and Olga Fink. "Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction," _Applied Energy_, 2023.](https://www.sciencedirect.com/science/article/pii/S0306261923005937)

## Table of Contents

- [Abstract](#abstract)
- [Background](#background)
  - [Battery](#battery)
  - [Transformer](#transformer)
- [Model Architecture](#model-architecture)
  - [Embedding](#embedding)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
- [Results](#results)
  - [Experimental Setup](#experimental-setup)
  - [Performance Evaluation](#performance-evaluation)
- [Conclusion](#conclusion)

## Abstract

The main purpose of the paper is to propose a novel deep learning architecture which can be applied to batteries, namely **Dynaformer**.
The model, Dynaformer, is a Transformer-based [[2]](#2) model and outputs an EoD (end of discharge) prediction, given some context current and voltage curves.

Dynaformer is able to infer the ageing state from a limited samples and predict the full voltage discharge curve, EoD, simultaneously.

## Background

### Battery

The following _Figure 1._ represents how batteries' voltage curves look like based on the given input current profiles.
When given constant current, it forms a continuous curve as the graph on the far left.
On the other hand, when given some variable current profiles with multiple transitions, the graphs differ from the other corresponding to the altering current.

{% include figure.html path="../assets/img/dynaformer/fig1.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 1.</i>
    Varying voltage curves corresponding to altering current profiles
    <a href="#1">[1]</a>
</p>

There are two main categories of solving EoD prediction and ageing inference:
model-based method and data-driven method.
The first method, model-based method, mainly focuses on representation of battery's physical internal mechanism.
By constructing an ideal model of the battery, it has a good performance.
However, due to it's complex representation of the model, it is computationally expensive and is not an easy work to design the model precisely.<br>
The second method, data-driven method, is to predict EoD and ageing inference using a huge amount of battery data.
It is easier to model the system compared to the model-based method since it requires minimum prior knowledge on the battery's discharge and degradation processes.
Nevertheless, it also has some disadvantages.
It requires large labeled training dataset and it is inefficient to train with such long time series data.

### Transformer

It is essential to understand how **Transformer** [[2]](#2) works in order to understand the Dynaformer.
Let's consider the following sentence.

<p align="center">
    <b>The dog</b>
    is playing and
    <b>she</b>
    likes playing with
    <b>me</b>.
</p>

We know that <b>she</b> is indicating <b>the dog</b>, not <b>me</b>.
Transformer is all about understanding the context and the hidden meaning behind the given data.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="../assets/img/dynaformer/fig2-1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="../assets/img/dynaformer/fig2-2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="../assets/img/dynaformer/fig2-3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<p align="center">
    <i>Figure 2.</i>
    Transformer model architecture
    <a href="#2">[2]</a>.
    (Left) Positional embedding
    (Centre) Encoder
    (Right) Decoder
</p>

## Model Architecture

### Embedding

### Encoder

### Decoder

## Results

### Experimental Setup

### Performance Evaluation

## Conclusion

## Reference

<a id="1" href="https://www.sciencedirect.com/science/article/pii/S0306261923005937">[1]</a>
Luca Biggio, Tommaso Bendinelli, Cheta Kulkarni, and Olga Fink. "Dynaformer: A Deep Learning Model for Ageing-aware Battery Discharge Prediction,"
<i> _Applied Energy_</i> 2023.

<a id="2" href="https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html">[2]</a>
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention is All you Need,"
<i>Advances in Neural Information Processing Systems 30 (NIPS 2017)</i>, 2017.

<a id="3" href="https://nasa.github.io/progpy/index.html">[3]</a>
Chris Teubert, Katelyn Jarvis, Matteo Corbetta, Chetan Kulkarni, Portia Banerjee, Jason Watkins, and Matthew Daigle, “ProgPy Python Prognostics Packages,” v1.6, Oct 2023. URL
[https://nasa.github.io/progpy/index.html](https://nasa.github.io/progpy/index.html)

<a id="4" href="https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/">[4]</a>
B. Saha and K. Goebel (2007). “Battery Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

<a id="5" href="https://link.springer.com/chapter/10.1007/978-3-030-32381-3_16">[5]</a>
Sun, Chi, Xipeng Qiu, Yige Xu, and Xuanjing Huang. "How to fine-tune bert for text classification?." In
<i>Chinese Computational Linguistics: 18th China National Conference, CCL 2019, Kunming, China, October 18–20, 2019, Proceedings 18</i>, pp. 194-206. Springer International Publishing, 2019.

<a id="6" href="https://arxiv.org/abs/2010.11929">[6]</a>
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "An image is worth 16x16 words: Transformers for image recognition at scale."
<i>arXiv preprint arXiv:2010.11929</i> (2020).
