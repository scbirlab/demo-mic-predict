---
title: Predict MICs against bacterial species with uncertainty
emoji: ⚡
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.23.3
python_version: 3.11.10
app_file: app.py
pinned: true
license: mit
short_description: Predict MIC (with uncertainty) against bacterial species
models:
  - scbirlab/spark-dv-2510-all
datasets:
  - scbirlab/thomas-2018-spark-all
---

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/scbirlab/mic-predict)

Predictions are from an AI model trained the [SPARK
dataset](https://doi.org/10.1021/acsinfecdis.8b00193), available to browse 
[here](https://huggingface.co/datasets/scbirlab/thomas-2018-spark-wt).

Predictions are given in micromolar (µM) and µg/mL. You can optionally have uncertainty scores 
calculated. These can take a few minutes, so please be patient.

This [model](https://huggingface.co/scbirlab/spark-dv-2510-all) was generated using 
[our DuvidNN framework](https://github.com/scbirlab/duvidnn), as a result of 
hyperparameter searches and selecting the model that performs best on unseen 
test data (from a scaffold split). 
Duvida also allows the calculation of uncertainty metrics based on training data.

Click [here](https://huggingface.co/scbirlab/spark-dv-2510-all) for training 
details, model configurations, and evaluation metrics.
