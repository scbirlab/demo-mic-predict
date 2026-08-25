

from typing import Iterable, List, Optional, Union
import csv
import itertools
import os
import sys
csv.field_size_limit(sys.maxsize)

from carabiner import print_err
import gradio as gr
import nemony as nm
import numpy as np
import pandas as pd
from rdkit.Chem import Draw, Mol
from schemist.converting import _FROM_FUNCTIONS
import torch

from .config import (
    DEVICE, 
    DROPDOWN,
    EXTRA_METRICS,
    MODEL_REPOS,
)


def get_dropdown_options(df, _type = str):
    if _type == str:
        cols = list(df.select_dtypes(exclude=[np.number]))
    else:
        cols = list(df.select_dtypes([np.number]))
    non_none = [col for col in cols if col is not None]
    if len(cols) > 0:
        default_value = non_none[0]
    else:
        default_value = ""
    print_err(f"Dropdown default value is {default_value}")
    return gr.update(
        choices=cols, 
        interactive=True, 
        value=default_value, 
        visible=True,
        allow_custom_value=True,
    )


def _init_dropdown(
    *args,
    **kwargs
):
    defaults = {
        "interactive": True,
        "visible": True,
        "render": False,
    }
    return gr.Dropdown(
        *args,
        **(defaults | kwargs),
    )


def _init_plot(
    *args,
    **kwargs
):
    defaults = {
        "height": 500,
        "visible": False,
        "render": False,
    }
    return gr.ScatterPlot(
        *args,
        **(defaults | kwargs),
    )

def _initial_setup(
    device=DEVICE, 
    from_fn=_FROM_FUNCTIONS, 
    repos=MODEL_REPOS, 
    extras=EXTRA_METRICS,
    dropdowns=DROPDOWN
):

    """Set up blocks.

    """
    print_err(f"Default torch device is {device}")

    line_inputs = {
        "format": _init_dropdown(
            label="Input string format",
            choices=list(from_fn),
            value="smiles",
        ),
        "model": _init_dropdown(
            label="model for prediction",
            choices=list(repos["models"]),
            value=list(repos["models"])[0],
        ),
        "species": _init_dropdown(
            label="Species for prediction",
            choices=list(dropdowns["species"]),
            value=list(dropdowns["species"])[:1],
            allow_custom_value=True,
        ),
        "extras": gr.CheckboxGroup(
            label="Extra metrics (Doubtscore & Information Sensitivity can increase calculation time to a couple of minutes!)",
            choices=list(extras),
            value=list(extras)[:2],
            interactive=True,
            render=False,
        ),
        "strings": gr.Textbox(
            label="Input",
            placeholder="Paste your molecule here, one per line.",
            lines=2,
            interactive=True,
            submit_btn=True,
            render=False,
        ),
    }
    output_line = gr.DataFrame(
        label="Predictions (scroll left and right)",
        interactive=False,
        visible=True,
        render=False,
    )
    download_single = gr.DownloadButton(
        label="Download predictions",
        visible=True,
        render=False,
    )
    drawing = gr.Image(
        label="Chemical structures",
        render=False,
    )

    file_inputs = {
        "file": gr.File(
            label="Upload a table of chemical compounds here",
            file_types=[".xlsx", ".csv", ".tsv", ".txt"],
            render=False,
        ),
        "column": _init_dropdown(
            label="Input column name",
            choices=[],
            allow_custom_value=True,
        ),
        "format": _init_dropdown(
            label="Input string format",
            choices=list(from_fn),
            value="smiles",
        ),
        "species": [
            _init_dropdown(
                label="Species 1 for prediction",
                choices=list(dropdowns["species"]),
                value=list(dropdowns["species"])[0],
                allow_custom_value=True,
            ),
            _init_dropdown(
                label="Species 2 for prediction",
                choices=list(dropdowns["species"]),
                value=None,
                allow_custom_value=True,
            ),
        ],
        "extras": gr.CheckboxGroup(
            label="Extra metrics (Information Sensitivity can increase calculation time)",
            choices=list(extras),
            value=list(extras)[:2],
            interactive=True,
            render=False,
        ),
    }

    input_dataframe = gr.Dataframe(
        label="Input data",
        max_height=500,
        visible=True,
        interactive=False,
        buttons=[
            "fullscreen",
            "copy",
        ],
        show_search="filter",
        max_chars=45,
        render=False,
    )
    download = gr.DownloadButton(
        label="Download predictions",
        visible=False,
        render=False,
    )
    plot_button = gr.Button(
        value="Plot!",
        visible=False,
        render=False,
    )

    left_plot_inputs = {
        "observed": _init_dropdown(
            label="Observed column (y-axis) for left plot",
            choices=[],
            value=None,
            allow_custom_value=True,
        ),
        "color": _init_dropdown(
            label="Color for left plot",
            choices=[],
            value=None,
            allow_custom_value=True,
        )
    }

    right_plot_inputs = {
        "x": _init_dropdown(
            label="x-axis for right plot",
            choices=[],
            value=None,
            allow_custom_value=True,
        ),
        "y": _init_dropdown(
            label="y-axis for right plot",
            choices=[],
            value=None,
            allow_custom_value=True,
        ),
        "color": _init_dropdown(
            label="Color for right plot",
            choices=[],
            value=None,
            allow_custom_value=True,
        )
    }
    plots = {
        "left": _init_plot(),
        "right": _init_plot(),
    }

    return (
        line_inputs, 
        output_line, 
        download_single,
        drawing,
        file_inputs, 
        input_dataframe, 
        download,
        plot_button,
        left_plot_inputs, 
        right_plot_inputs, 
        plots,
    )