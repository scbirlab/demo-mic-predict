---
title: Predict MICs against bacterial species with uncertainty
emoji: ⚡
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.23.1
python_version: 3.11.10
app_file: app.py
pinned: false
license: mit
short_description: Predict MIC (with uncertainty) against bacterial species
models:
  - scbirlab/spark-dv-fp-2503-abau
  - scbirlab/spark-dv-fp-2503-babo
  - scbirlab/spark-dv-fp-2503-ecol
  - scbirlab/spark-dv-fp-2503-ftul
  - scbirlab/spark-dv-fp-2503-kpne
  - scbirlab/spark-dv-fp-2503-paer
  - scbirlab/spark-dv-fp-2503-saur
  - scbirlab/spark-dv-fp-2503-spne
  - scbirlab/spark-dv-fp-2503-yent
  - scbirlab/spark-dv-fp-2503-ypes
datasets:
  - scbirlab/thomas-2018-spark-wt
---

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/scbirlab/mic-predict)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference