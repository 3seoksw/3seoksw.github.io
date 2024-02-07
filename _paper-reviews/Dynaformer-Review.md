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

Transformer is consists of three major parts: positional embedding, encoder and decoder. The above _Figure 2._ represents the mentioned three parts.<br>
Positional embedding is to transfer the given data into numerical vectors. By doing so, positional information can be embedded within the vectors.<br>

Encoder's job is to obtain **query**, **key** and **value** -- $$(Q, K, V)$$ -- given some positional embedding.
The **query** is a vector which contains given specific data such as a word itself when a sentence is given.
The **key** is a value which can specify the **query**.
And lastly, the **value** represents the **query**'s hidden meaning.
It can contain context or positional information.
Here, **self-attention** comes in. Simply saying, self-attention cells find the correlations among the data by using $$(Q, K, V)$$.<br>

Decoder's job is very similiar with the encoder but it differs with the main purpose.
While the encoder's main purpose is to find the correlations among input data,
the decoder's main purpose is to find the correlations between the input data and ouput data.<br>
To simplify, let's bring the classic translation example.
Say I want to translate some English sentences into Korean sentences.
Then the input of the encoder is the English sentences and the input of the decoder is the the Korean sentences, in other words, data that we target.
So the encoder mainly interprets and finds the meaning, hidden information, and correlations from the English sentences,
and the decoder focuses on finding the correlations between the English sentences and the Korean sentences, when training.<br>
Back to the point, the decoder also obtains **query**, **key**, and **value**.
However, in the decoder, **key** and **value** from the encoder and **query** from decoder will only be used.
By using $$(K, V)$$ from the encoder and $$(Q)$$ from the decoder, the decoder is able to apply self-attention mechanism to find the correlations between the data we want to interpret and the data we aims.

## Model Architecture

<div align="center">
    {% include figure.html path="../assets/img/dynaformer/fig2-3.png" class="img-fluid rounded z-depth-1" zoomable=true width="75%" %}
</div>

<p align="center">
    <i>Figure 3.</i>
    Representation of the components of Dynaformer
    <a href="#1">[1]</a>
</p>

The proposed Dynaformer is basically the same with the Transformer model but with the difference of the data type.
Since the paper is to predict EoD, input data types are current and voltage curves.
Note that the inputs of the encoder are current and voltage curves and the inputs of the decoder are the rest of the current curves and the output of the encoder,
then eventually outputs full discharge voltage curves.

<div align="center">
    {% include figure.html path="../assets/img/dynaformer/fig4.jpg" class="img-fluid rounded z-depth-1" zoomable=true width="75%" %}
</div>

<p align="center">
    <i>Figure 4.</i>
    Dynaformer - model architecture
    <a href="#1">[1]</a>
</p>

_Figure 3._ and _Figure 4._ are the same model architecture, but for the sake of easy understanding of the Dynaformer using the Transformer-style architecture representation, _Figure 3._ can be redrawn as _Figure 4._.
The following figures specify each part from the Dynaformer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="../assets/img/dynaformer/fig4-1.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="../assets/img/dynaformer/fig4-2.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="../assets/img/dynaformer/fig4-3.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<p align="center">
    <i>Figure 4-1, 4-2, 4-3.</i>
    Dynaformer - detailed model architecture
    <a href="#1">[1]</a>.
    (Left) Positional embedding
    (Centre) Encoder
    (Right) Decoder
</p>

### Embedding

<div align="center">
    {% include figure.html path="../assets/img/dynaformer/fig5.jpg" class="img-fluid rounded z-depth-1" zoomable=true width="75%" %}
</div>

<p align="center">
    <i>Figure 5.</i>
    Dynaformer - (Context) Embedding
    <a href="#1">[1]</a>
</p>

Here, Dynaformer gets information regarding the battery's profiles,
which are current profile (curve) and voltage profile (curve).
While (positional) embedding, nothing special happens but reshapes the data into $$(Q, K, V)$$ including positional, context information.
The $$Q$$ represents current and voltage curves, the $$K$$ serves as a specifier to find the $$Q$$, and the $$V$$ contains positional information such as time for the matching $$Q$$.

### Encoder

<div align="center">
    {% include figure.html path="../assets/img/dynaformer/fig6.jpg" class="img-fluid rounded z-depth-1" zoomable=true width="75%" %}
</div>

<p align="center">
    <i>Figure 6.</i>
    Dynaformer - Encoder
</p>

The encoder's inputs are $$(Q, K, V)$$ from the embedding.
Here, it's main role is to find the correlations between the input current, voltage, and time which will eventually extracts degradation information.
In order to find such information, multi-head self-attention cells are used. For further information regarding self-attention mechanism, please see [[2]](#2).

### Decoder

<div align="center">
    {% include figure.html path="../assets/img/dynaformer/fig7.jpg" class="img-fluid rounded z-depth-1" zoomable=true width="75%" %}
</div>

<p align="center">
    <i>Figure 7.</i>
    Dynaformer - Decoder
    <a href="#1">[1]</a>
</p>

The decoder predicts EoD as a final ouput, given the ageing inference from the encoder's ouput and the rest of the current curves which are from the decoder's input.
Using the ageing inference, $$(K, V)$$, and the rest of the current curves, $$Q$$,
the Dynaformer is now able to predict the voltage curves corresponding to the current curves from the input of the decoder exploiting the ageing inference information from the encoder.

## Results

### Experimental Setup

Just like I've mentioned from the [Background - Battery](#battery) section, there are two main limitations when using data-driven method for predicting EoD and ageing inference:
requires large labeled dataset and requires long time series dataset which is inefficient for training.<br>
The proposed Dynaformer solves the problems originally had when using data-driven method.

First, the Dynaformer mitigated the problem of lack of dataset for training by using the following:
[[3]](#3) and [[4]](#4).
[[3]](#3) is a NASA Prognostics Model library which helps creating a simulated training dataset.
The paper experiments the model by changing two degradation parameters, $$q_max$$ and $$R_0$$, to observe the performance change when the parameters changed.
Also [[4]](#4) is a NASA real-world Dataset for batteries and is used as a input for the simulator, [[3]](#3).<br>
However, there may be a simulation-to-real gap (sim2real gap) since the training dataset is a simulation-based data.
To mitigate such concern, the paper applied transfer learning. As [[5]](#5) suggests, training the model with simulated data then using a limited amount of real data to adapt the model can reduce the model's bias towards simulated data.

Second, the Dynaformer gets the **tokenized** input.
In other words, instead of feeding a full length of the curves,
the Dynaformer only accepts a small sized curves in sequence.
Please see _Figure 3._ for better understanding of the concept of **token**.
Training the model with long time series data is in fact inefficient.
However, using the technique from [[6]](#6) solves such problem.

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
