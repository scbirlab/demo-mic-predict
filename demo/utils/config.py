
CACHE = "./cache"

import csv
import json
import os
import sys
os.environ["XDG_CACHE_HOME"] = CACHE
os.environ["HF_HOME"] = CACHE
os.environ["DUVIDNN_CACHE"] = CACHE
csv.field_size_limit(sys.maxsize)

from duvidnn.autoclass import AutoModelBox
import gradio as gr
import torch

THEME = gr.themes.Default()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_ROWS = 1000
BATCH_SIZE = 16
HEADER_FILE = os.path.join("sources", "header.md")

with open(os.path.join("data", "examples", "examples.json"), "r") as f:
    EXAMPLES = json.load(f)
with open(os.path.join("data", "repos.json"), "r") as f:
    MODEL_REPOS = json.load(f)

with open(os.path.join("data", "species-dropdown.json"), "r") as f:
    DROPDOWN = json.load(f)

MODELBOXES = {
    key: AutoModelBox.from_pretrained(
        val,
        cache_dir=CACHE,
    )
    for key, val in MODEL_REPOS["models"].items()
}
[mb.to(DEVICE) for mb in MODELBOXES.values()]

EXTRA_METRICS = {
    "log10(variance)": lambda modelbox, candidates: (
        modelbox.prediction_variance(
            candidates=candidates, 
            batch_size=BATCH_SIZE,
            cache=CACHE,
        )
        .map(
            lambda x: {
                modelbox._variance_key: torch.log10(x[modelbox._variance_key])
            },
            batched=True,
            batch_size=BATCH_SIZE,
        )
    ), 
    "Tanimoto nearest neighbor to training data": lambda modelbox, candidates: modelbox.tanimoto_nn(data=candidates, batch_size=BATCH_SIZE), 
    "Doubtscore": lambda modelbox, candidates: (
        modelbox.doubtscore(
            candidates=candidates, 
            batch_size=BATCH_SIZE,
            cache=CACHE,
        )
        .map(
            lambda x: {"doubtscore": torch.log10(x["doubtscore"])},
            batched=True,
            batch_size=BATCH_SIZE,
        )
    ), 
    "Information sensitivity (approx.)": lambda modelbox, candidates: (
        modelbox.information_sensitivity(
            candidates=candidates, 
            batch_size=BATCH_SIZE, 
            optimality_approximation=True, 
            last_layer_only=True,
            approximator="exact_diagonal",
            cache=CACHE,
        )
        .map(
            lambda x: {"information sensitivity": torch.log10(x["information sensitivity"])},
            batched=True,
            batch_size=BATCH_SIZE,
        )
    ),
}
