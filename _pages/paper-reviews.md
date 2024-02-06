---
layout: page
title: Paper Reviews
permalink: /paper-reviews/
description: A collection of the papers I've read.
nav: true
nav_order: 2
display_categories: [DRL, BNN, Transformer]
horizontal: false
---

<!-- pages/projects.md -->
<div class="paper-reviews">
{%- if site.enable_paper-reviews_categories and page.display_categories %}
  <!-- Display categorized projects -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_paper-reviews = site.paper-reviews | where: "category", category -%}
  {%- assign sorted_paper-reviews = categorized_paper-reviews | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for paper-review in sorted_paper-reviews -%}
      {% include paper-reviews_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for paper-review in sorted_paper-reviews -%}
      {% include paper-reviews.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
  {% endfor %}

{%- else -%}

<!-- Display projects without categories -->

{%- assign sorted_paper-reviews = site.paper-reviews | sort: "importance" -%}

  <!-- Generate cards for each project -->

{% if page.horizontal -%}

  <div class="container">
    <div class="row row-cols-2">
    {%- for paper-review in sorted_paper-review -%}
      {% include paper-reviews_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for paper-review in sorted_paper-reviews -%}
      {% include paper-reviews.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
{%- endif -%}
</div>
