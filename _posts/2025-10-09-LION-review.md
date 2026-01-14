---
layout: post
title: LION, Empowering MLLM with Dual-Level Visual Knowledge
date: 2025-10-09
description: Paper review of LION
tags: MLLM
categories: paper-review
giscus_comments: false
related_posts: false
related_publications: true
---

The review is done with the following paper:<br>
[LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge](https://openaccess.thecvf.com/content/CVPR2024/liquid/Chen_LION_Empowering_Multimodal_Large_Language_Model_with_Dual-Level_Visual_Knowledge_CVPR_2024_paper.html).

## Abstract

Existing MLLMs mainly use vision encoders that are pre-trained on coarsely aligned image-text pairs, which often leads to vague and inaccurate responses.
These issues are due to the insufficient extraction and reasoning of visual knowledge.
In other words, the existing models struggle from region-level tasks.

To tackle this problem, the paper proposes the LION model.
The objective of the model is to inject two levels of visual knowledge, which are image-level and region-level understanding.
To make this possible, the model incorporates fine-grained spatial-aware knowledge, and applies soft prompting of high-level semantic visual evidence.
These two enable the MLLM  to capture both global and local visual information from a given image.

{% include figure.liquid path="../assets/img/LION/LION_MLLM_comp.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 1.</i>
    Comparison between existing MLLMs and LION
    <a href="#1">[1]</a>.
</p>

## Related Works -- Visual Grounding

{% include figure.liquid path="../assets/img/LION/bbox_rep.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 2.</i>
    Representation of object description and bounding box which follows Markdown link format
    <a href="#2">[1]</a>.
</p>

{% include figure.liquid path="assets/img/LION/kosmos-2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
Figure 3. Kosmos-2 offering object description with bounding box <d-cite key="peng2023kosmos"></d-cite>.

While there are numerous works on assigning visual grounding tasks to MLLMs, Kosmos-2 is a great example for comparison with the LION model.
Kosmos-2 converts existing datasets into a Markdown-style link format.
This format represents a bounding box which includes spatial coordinates enclosed in square brackets as shown in Figure 2 and 3.

The reason this Markdown link format matters is that it provides a tokenizer-friendly representation, making it easier for the model to understand both the semantic meaning of the image tags and their spatial locations.
In this way, it effectively bridges the gap between text and visual grounding tasks.
However, the Kosmos-2 still fall short in handling broader aspect of visual tasks beyond region-level grounding.

## Method

### Pipeline

{% include figure.liquid path="../assets/img/LION/lion_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 4.</i>
    Overview of LION
    <a href="#1">[1]</a>.
</p>

### Spatial Visual Knowledge

{% include figure.liquid path="../assets/img/LION/lion_spatial.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 5.</i>
    Representation of how LION handle spatial visual knowledge
    <a href="#1">[1]</a>.
</p>

To incorporate spatial-aware visual knowledge into the model, the paper suggests reformatting the datasets into a unified format, that combines natural language descriptions and object coordinates enclosed in square brackets for instruction tuning, just like Kosmos-2 did.

Still, it remains two main internal conflicts when the model tries to learn both image-level and region-level visual tasks.
One is the need of region-level modality-alignment pre-training.
This is because most MLLMs are only trained with global image features.
And the second is the gap between the input-output modes of image-level and region level visual tasks.
As mentioned earlier, the reformatted data contains text and coordinates, which can confuse the model when trained together.

To address these two conflicts, a stage-wise instruction tuning strategy is applied, which is a three-stage training strategy.

1. Image-level
   For the first stage, the model learns general vision-language understanding, by fine-tuning the Q-Former and the image-level adapter in the LLM.
   This offers the model with image-level knowledge.

2. Region-level
   And for the second stage, the model focuses on fine-grained spatial knowledge, by using a vision aggregator to capture detailed visual features along with the region-level adapter.
   At this stage, the model learns more about region-level knowledge, and as the model is trained on region-level AFTER the image-level, it can avoid severe interference between the two levels.

3. Mixture of the both
   Finally at the third stage, the model combines the outputs from the previous two stages using a router, which dynamically balances image-level and region-level knowledge.
   The key component here is the router.
   The router not only balances between the two levels of knowledge, but also aligns the input-output mismatch by assigning adaptive weights to each adapters’ output based on the task type.

{% include figure.liquid path="../assets/img/LION/stage-wise_instruction-tuning.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 6.</i>
    The stage-wise instruction-tuning strategy
    <a href="#1">[1]</a>.
</p>

$$
\begin{align*}
    O^t = F(X) + \sum_{k=1}^{K=2}G_k^t \odot H_k(X),
\end{align*}
$$

where $$H_k(X)$$ is an adapter for $$k$$-th adapter and $$F(X)$$ is the output of FFN.

To make the stage-wise instruction possible, we must first look at the placement of the adapters in the LLM.
Each adapter is inserted at every Feed-Forward Network layer in a parallel manner within the frozen LLM.
In this arrangement, the output features generated by the standard FFN are simply added to the output features generated by the adapter layer.

Sine each adapter is trained separately, we treat these specialized components as distinct experts.

However, as previously mentioned, the router is the key component that enables the model to use these adapters effectively.
The router module dynamically decides how much to rely on each adapter based on the task by learning a weight vector G for every task.

 
For example, if the input involves spatial reasoning, the router increases the contribution of the region-level adapter by updating its weight, and vice versa for global reasoning.

{% include figure.liquid path="../assets/img/LION/vision_aggregator.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 7.</i>
    Vision Aggregator.
</p>

The next challenge is to ensure that the model can capture sufficient visual details.
To capture the fine-grained spatial-aware visual knowledge needed for tasks like visual grounding, the paper introduced a component called the Vision Aggregator.

The Vision Aggregator functions as a tiny transformer-style network where that improves LION's understanding of object boundaries, spatial relations, and fine object attributes.

Ablation studies demonstrate that the VA promotes the extraction of this fine-grained knowledge and significantly improves referring expression comprehension (REC) performance.

### Soft Prompting

#### Image Tag Extraction

To improve LION's capabilities, the paper included semantic comprehension.
The authors used an off-the-shelf image tag extractor called Recognize Anything Model (RAM).

#### Soft prompting

Since the predicted tags from this model are not flawless, they can mislead the model.
So, LION uses a soft prompt to mitigate the influence of these imperfect tags.
A trainable embedding is added to the instruction text that teaches the model how to use the tag information.
In the paper, the phrase "According to <i>hint</i>, you are allowed to use or partially use the following tags: ...".
This method helps guide the model to select valuable information from the tags and ignore the incorrect ones.

{% include figure.liquid path="../assets/img/LION/soft-prompting.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Figure 8.</i>
    Instruction template with soft prompt
    <a href="#1">[1]</a>.
</p>

## Experimental Results

{% include figure.liquid path="../assets/img/LION/tab1.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Table 1.</i>
    Comparison on image vaptioning and Visual Question Answering (VQA). The best and second performances for each benchmark are indicated in bold and underline, respectively
    <a href="#1">[1]</a>.
</p>

{% include figure.liquid path="../assets/img/LION/tab2.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Table 2.</i>
    Comparison on Referring Expression Comprehension (REC)
    <a href="#1">[1]</a>.
</p>

{% include figure.liquid path="../assets/img/LION/tab3.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<p align="center">
    <i>Table 3.</i>
    Evaluation of object hallucination
    <a href="#1">[1]</a>.
</p>

The LION architecture effectively addresses the challenge of insufficient visual knowledge extraction and reasoning, which affects existing Multimodal Large Language Models (MLLMs) that typically rely only on coarsely aligned image-text pairs.

 
The core innovation of LION is the injection of dual-level visual knowledge where, first, the Fine-grained spatial-aware knowledge is incorporated using the mixture-of-adapters using a router and a Vision Aggregator, and, second, the High-level semantic visual evidence is provided by image tags through a soft prompting method.

## Reference

<a id="1" href="https://openaccess.thecvf.com/content/CVPR2024/liquid/Chen_LION_Empowering_Multimodal_Large_Language_Model_with_Dual-Level_Visual_Knowledge_CVPR_2024_paper.html">[1]</a>
Chen, G., Shen, L., Shao, R., Deng, X., & Nie, L. (2024). Lion: Empowering multimodal large language model with dual-level visual knowledge.
In <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition</i> (pp. 26540-26550).

<a id="2" href="https://arxiv.org/abs/2306.14824">[1]</a>
Peng, Z., Wang, W., Dong, L., Hao, Y ., Huang, S., Ma, S., & Wei, F. (2023). Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824.
