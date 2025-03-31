"""Gradio demo for schemist."""

from typing import Iterable, List, Optional, Union
import csv
from functools import partial
from io import TextIOWrapper
import itertools
import json
import os
import sys

csv.field_size_limit(sys.maxsize)

from carabiner import cast, print_err
from carabiner.pd import read_table
from duvida.autoclass import AutoModelBox
import gradio as gr
import nemony as nm
import numpy as np
import pandas as pd
from rdkit.Chem import Draw, Mol
from schemist.converting import (
    _FROM_FUNCTIONS, 
    convert_string_representation, 
    _x2mol,
)
from schemist.tables import converter
import torch
from duvida.stateless.config import config

THEME = gr.themes.Default()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE = "./cache"
MAX_ROWS = 500
BATCH_SIZE = 32
HEADER_FILE = os.path.join("sources", "header.md")
with open("repos.json", "r") as f:
    MODEL_REPOS = json.load(f)

MODELBOXES = {
    key: AutoModelBox.from_pretrained(val, cache_dir=os.path.join(CACHE, "duvida"))
    for key, val in MODEL_REPOS.items()
}
[mb.to(DEVICE) for mb in MODELBOXES.values()]

EXTRA_METRICS = {
    "log10(variance)": lambda modelbox, candidates: modelbox.prediction_variance(candidates=candidates, batch_size=BATCH_SIZE, cache=CACHE).map(lambda x: {modelbox._variance_key: torch.log10(x[modelbox._variance_key])}), 
    "Tanimoto nearest neighbor to training data": lambda modelbox, candidates: modelbox.tanimoto_nn(candidates=candidates, batch_size=BATCH_SIZE), 
    "Doubtscore": lambda modelbox, candidates: modelbox.doubtscore(candidates=candidates, cache=CACHE, batch_size=BATCH_SIZE).map(lambda x: {"doubtscore": torch.log10(x["doubtscore"])}), 
    "Information sensitivity (approx.)": lambda modelbox, candidates: modelbox.information_sensitivity(candidates=candidates, batch_size=BATCH_SIZE, optimality_approximation=True, approximator="squared_jacobian", cache=CACHE).map(lambda x: {"information sensitivity": torch.log10(x["information sensitivity"])}),
}

with open(os.path.join("example-data", "examples.json"), "r") as f:
    EXAMPLES = json.load(f)

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
    return gr.Dropdown(
        choices=cols, 
        interactive=True, 
        value=default_value, 
        visible=True,
        allow_custom_value=True,
    )


def load_input_data(file: Union[TextIOWrapper, str], return_pd: bool = False) -> pd.DataFrame:
    file = file if isinstance(file, str) else file.name
    print_err(f"Loading {file}")
    df = read_table(file, nrows=MAX_ROWS)
    print_err(df.head())
    if return_pd:
        return (df, gr.Dataframe(value=df, visible=True)), get_dropdown_options(df, str)
    else:
        return gr.Dataframe(value=df, visible=True), get_dropdown_options(df, str)
    

def _clean_split_input(strings: str) -> List[str]:
    return [
        s2.split(":")[-1].strip() 
        for s in strings.split("\n") 
        for s2 in s.split(",")
    ]


def _convert_input(
    strings: str,
    input_representation: str = 'smiles', 
    output_representation: Union[Iterable[str], str] = 'smiles'
) -> List[str]:
    strings = _clean_split_input(strings)
    converted = convert_string_representation(
        strings=strings, 
        input_representation=input_representation, 
        output_representation=output_representation,
    )
    return {key: list(map(str, cast(val, to=list))) for key, val in converted.items()}


def convert_one(
    strings: str,
    input_representation: str = 'smiles', 
    output_representation: Union[Iterable[str], str] = 'smiles',
):
    output_representation = cast(output_representation, to=list)
    for rep in output_representation:
        message = f"Converting from {input_representation} to {rep}..."
        gr.Info(message, duration=10)

    df = pd.DataFrame({
        input_representation: _clean_split_input(strings),
    })

    return convert_file(
        df=df,
        column=input_representation,
        input_representation=input_representation,
        output_representation=output_representation,
    )


def _prediction_loop(
    df: pd.DataFrame,
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None
) -> pd.DataFrame:
    species_to_predict = cast(predict, to=list)
    prediction_cols = []
    if extra_metrics is None:
        extra_metrics = []
    else:
        extra_metrics = cast(extra_metrics, to=list)
    for species in species_to_predict:
        message = f"Predicting for species: {species}"
        print_err(message)
        gr.Info(message, duration=3)
        this_modelbox = MODELBOXES[species]
        this_features = this_modelbox._input_cols
        this_labels = this_modelbox._label_cols
        this_prediction_input = (
            df
            .rename(columns={
                "smiles": this_features[0],
            })
            .assign(**{label: np.nan for label in this_labels})
        )
        print(this_prediction_input)
        prediction = this_modelbox.predict(
            data=this_prediction_input,
            features=this_features,
            labels=this_labels,
            aggregator="mean",
            cache=CACHE,
        ).with_format("numpy")["__prediction__"].flatten()
        print(prediction)
        this_col = f"{species}: predicted MIC (µM)"
        df[this_col] = np.power(10., -prediction) * 1e6
        prediction_cols.append(this_col)
        this_col = f"{species}: predicted MIC (µg / mL)"
        df[this_col] = np.power(10., -prediction) * 1e3 * df["mwt"]
        prediction_cols.append(this_col)

        for extra_metric in extra_metrics:
            message = f"Calculating {extra_metric} for species: {species}"
            print_err(message)
            gr.Info(message, duration=10)
            # this_modelbox._input_training_data = this_modelbox._input_training_data.remove_columns([this_modelbox._in_key])
            this_col = f"{species}: {extra_metric}"
            prediction_cols.append(this_col)
            print(">>>", this_modelbox._input_training_data)
            print(">>>", this_modelbox._input_training_data.format)
            print(">>>", this_modelbox._in_key, this_modelbox._out_key)
            this_extra = (
                EXTRA_METRICS[extra_metric](
                    this_modelbox,
                    this_prediction_input,
                )
                .with_format("numpy")
            )
            df[this_col] = this_extra[this_extra.column_names[-1]]

    return prediction_cols, df


def predict_one(
    strings: str,
    input_representation: str = 'smiles', 
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None,
    return_pd: bool = False
):
    prediction_df = convert_one(
        strings=strings,
        input_representation=input_representation,
        output_representation=['id', 'pubchem_name', 'pubchem_id', 'smiles', 'inchikey', "mwt", "clogp"],
    )
    prediction_cols, prediction_df = _prediction_loop(
        prediction_df,
        predict=predict,
        extra_metrics=extra_metrics,
    )
    df = prediction_df[
        ['id', 'pubchem_name', 'pubchem_id'] 
        + prediction_cols 
        + ['smiles', 'inchikey', "mwt", "clogp"]
    ]
    gradio_opts = {
        "label": "Predictions",
        "value": df,
        "pinned_columns": 3,
        "visible": True,
        "wrap": True,
        "column_widths": [120] * 3 + [250] * (prediction_df.shape[1] - 3),
    }
    if return_pd:
        return df, gr.DataFrame(**gradio_opts)
    else:
        return gr.DataFrame(**gradio_opts)

def convert_file(
    df: pd.DataFrame, 
    column: str = 'smiles',
    input_representation: str = 'smiles',
    output_representation: Union[str, Iterable[str]] = 'smiles'
):
    output_representation = cast(output_representation, to=list)
    for rep in output_representation:
        message = f"Converting from {input_representation} to {rep}..."
        gr.Info(message, duration=10)
    print_err(df.head())
    print_err(message)
    gr.Info(message, duration=3)
    errors, df = converter(
        df=df,
        column=column,
        input_representation=input_representation,
        output_representation=output_representation,
    )
    df = df[
        cast(output_representation, to=list) +
        [col for col in df if col not in output_representation]
    ]
    all_err = sum(err for key, err in errors.items())
    message = (
        f"Converted {df.shape[0]} molecules from "
        f"{input_representation} to {output_representation} "
        f"with {all_err} errors!"
    )
    print_err(message)
    gr.Info(message, duration=5)
    return df


def predict_file(
    df: pd.DataFrame, 
    column: str = 'smiles',
    input_representation: str = 'smiles',
    predict: str = 'smiles', 
    predict2: Optional[str] = None, 
    extra_metrics: Optional[Union[Iterable[str], str]] = None,
    return_pd: bool = False
):
    predict = cast(predict, to=list)
    if predict2 is not None and predict2 in MODELBOXES:
        predict += cast(predict2, to=list)
    if extra_metrics is None:
        extra_metrics = []
    else:
        extra_metrics = cast(extra_metrics, to=list)

    if df.shape[0] > MAX_ROWS:
        message = f"Truncating input to {MAX_ROWS} rows"
        print_err(message)
        gr.Info(message, duration=15)
        df = df.iloc[:MAX_ROWS]

    prediction_df = convert_file(
        df,
        column=column,
        input_representation=input_representation,
        output_representation=["id", "smiles", "inchikey", "mwt", "clogp"],
    )
    prediction_cols, prediction_df = _prediction_loop(
        prediction_df,
        predict=predict,
        extra_metrics=extra_metrics,
    )
    main_cols = set(
        ['id', 'inchikey', 'smiles', "mwt", "clogp"] 
        + [column] 
        + prediction_cols
    )
    other_cols = [
        col for col in prediction_df 
        if col not in main_cols
    ]
    prediction_df = prediction_df[
        ['id', 'inchikey'] 
        + [column] 
        + prediction_cols + other_cols 
        + ['smiles', "mwt", "clogp"]
    ]
    plot_dropdown = get_dropdown_options(prediction_df, _type="number")
    plot_dropdown = (plot_dropdown,) * 5
    gradio_opts = {
        "label": "Predictions",
        "value": prediction_df,
        "pinned_columns": 3,
        "visible": True,
        "wrap": True,
        "column_widths": [120] * 3 + [250] * (prediction_df.shape[1] - 3),
    }

    if return_pd:
        return ((prediction_df, gr.Dataframe(**gradio_opts)),) + (plot_dropdown,)
    else:
        return (gr.Dataframe(**gradio_opts),) + (plot_dropdown,)


def draw_one(
    df,
    smiles_col: str = "smiles",
    legends: Optional[Union[str, Iterable[str]]] = None
):
    if legends is None:
        legends = ["inchikey", "id", "pubchem_name"]
    else:
        legends = []
    message = f"Drawing {df.shape[0]} molecules..."
    gr.Info(message, duration=10)
    _ids = {col: df[col].tolist() for col in legends}
    mols = cast(_x2mol(df[smiles_col], "smiles"), to=list)
    if isinstance(mols, Mol):
        mols = [mols]
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=min(3, len(mols)), 
        subImgSize=(450, 450),
        legends=[
            "\n".join(
                _x if _x is not None else "" 
                for _x in items
            ) for items in zip(*_ids.values())
        ],
    )


def log10_if_all_positive(df, col):
    if np.all(df[col] > 0.):
        df[col] = np.log10(df[col])
        title = f"log10[ {col} ]"
    else:
        title = col
    return title, df


def plot_x_vs_y(
    df,
    x: str,
    y: str,
    color: Optional[str] = None,
):  
    message = f"Plotting x={x}, y={y}, color={color}..."
    gr.Info(message, duration=10)
    print_err(df.head())
    y_title = y
    cols = ["id", "inchikey", "smiles", "mwt", "clogp", x, y]
    if color is not None and color not in cols:
        cols.append(color)
    cols = list(set(cols))
    x_title, df = log10_if_all_positive(df, x)
    y_title, df = log10_if_all_positive(df, y)
    color_title, df = log10_if_all_positive(df, color)

    return gr.ScatterPlot(
        value=df[cols],
        x=x,
        y=y,
        color=color,
        x_title=x_title,
        y_title=y_title,
        color_title=color_title,
        tooltip="all",
        visible=True,
    )


def plot_pred_vs_observed(
    df,
    species: str,
    observed: str,
    color: Optional[str] = None,
):  
    print_err(df.head())
    xcol = f"{species}: predicted MIC (µM)"
    ycol = observed
    return plot_x_vs_y(
        df,
        x=xcol,
        y=ycol,
        color=color,
    ) 


def download_table(
    df: pd.DataFrame
) -> str:
    df_hash = nm.hash(pd.util.hash_pandas_object(df).values)
    filename = os.path.join(CACHE, "downloads", f"predicted-{df_hash}.csv")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df.to_csv(filename, index=False)
    return gr.DownloadButton(value=filename, visible=True)


def _predict_then_draw_then_download(
    strings: str,
    input_representation: str = 'smiles', 
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None,
    smiles_col: str = "smiles",
    legends: Optional[Union[str, Iterable[str]]] = None,
    progress = gr.Progress(track_tqdm=True)
):
    df, gr_df = predict_one(
        strings=strings,
        input_representation=input_representation, 
        predict=predict, 
        extra_metrics=extra_metrics,
        return_pd=True,
    )
    img = draw_one(
        df,
        smiles_col="smiles",
    )
    return gr_df, img, download_table(df)


def _load_then_predict_then_download_then_reveal_plot(
    file: str,
    column: str = 'smiles',
    input_representation: str = 'smiles',
    predict: str = 'smiles', 
    predict2: Optional[str] = "", 
    extra_metrics: Optional[Union[Iterable[str], str]] = None,
    progress = gr.Progress(track_tqdm=True)
):
    (df, df_gr), col_opts = load_input_data(
        file, 
        return_pd=True,
    )
    (df, df_gr), plot_opts = predict_file(
        df,
        column=column,
        input_representation=input_representation,
        predict=predict,
        predict2=None if predict2 == "" else predict2,
        extra_metrics=extra_metrics,
        return_pd=True,
    )
    print_err(df.head())
    return (
        df_gr, 
        download_table(df),
    ) + plot_opts


def _initial_setup():

    """Set up blocks.

    """
    print_err(f"Duvida config is {config}")
    print_err(f"Default torch device is {DEVICE}")

    line_inputs = {
        "format": gr.Dropdown(
            label="Input string format",
            choices=list(_FROM_FUNCTIONS),
            value="smiles",
            interactive=True,
        ),
        "species": gr.CheckboxGroup(
            label="Species for prediction",
            choices=list(MODEL_REPOS),
            value=list(MODEL_REPOS)[:1],
            interactive=True,
        ),
        "extras": gr.CheckboxGroup(
            label="Extra metrics (Doubtscore & Information Sensitivity can increase calculation time to a couple of minutes!)",
            choices=list(EXTRA_METRICS),
            value=list(EXTRA_METRICS)[:2],
            interactive=True,
        ),
        "strings": gr.Textbox(
            label="Input",
            placeholder="Paste your molecule here, one per line.",
            lines=2,
            interactive=True,
            submit_btn=True,
        ),
    }
    output_line = gr.DataFrame(
        label="Predictions (scroll left and right)",
        interactive=False,
        visible=True,
    )
    download_single = gr.DownloadButton(
        label="Download predictions",
        visible=True,
    )
    drawing = gr.Image(label="Chemical structures")

    file_inputs = {
        "file": gr.File(
            label="Upload a table of chemical compounds here",
            file_types=[".xlsx", ".csv", ".tsv", ".txt"],
        ),
        "column": gr.Dropdown(
            label="Input column name",
            choices=[],
            allow_custom_value=True,
            visible=True,
            interactive=True,
        ),
        "format": gr.Dropdown(
            label="Input string format",
            choices=list(_FROM_FUNCTIONS),
            value="smiles",
            interactive=True,
            visible=True,
        ),
        "species": [
            gr.Dropdown(
                label="Species 1 for prediction",
                choices=list(MODEL_REPOS),
                value=list(MODEL_REPOS)[0],
                interactive=True,
                allow_custom_value=True,
            ),
            gr.Dropdown(
                label="Species 2 for prediction",
                choices=list(MODEL_REPOS),
                value=None,
                interactive=True,
                allow_custom_value=True,
            ),
        ],
        "extras": gr.CheckboxGroup(
            label="Extra metrics (Information Sensitivity can increase calculation time)",
            choices=list(EXTRA_METRICS),
            value=list(EXTRA_METRICS)[:2],
            interactive=True,
        ),
    }

    input_dataframe = gr.Dataframe(
        label="Input data",
        max_height=500,
        visible=True,
        interactive=False,
        show_fullscreen_button=True,
        show_search="filter",
        max_chars=45,
    )
    download = gr.DownloadButton(
        label="Download predictions",
        visible=False,
    )
    plot_button = gr.Button(
        value="Plot!",
        visible=False,
    )

    left_plot_inputs = {
        "observed": gr.Dropdown(
            label="Observed column (y-axis) for left plot",
            choices=[],
            value=None,
            interactive=True,
            visible=True,
            allow_custom_value=True,
        ),
        "color": gr.Dropdown(
            label="Color for left plot",
            choices=[],
            value=None,
            interactive=True,
            visible=True,
            allow_custom_value=True,
        )
    }

    right_plot_inputs = {
        "x": gr.Dropdown(
            label="x-axis for right plot",
            choices=[],
            value=None,
            interactive=True,
            visible=True,
            allow_custom_value=True,
        ),
        "y": gr.Dropdown(
            label="y-axis for right plot",
            choices=[],
            value=None,
            interactive=True,
            visible=True,
            allow_custom_value=True,
        ),
        "color": gr.Dropdown(
            label="Color for right plot",
            choices=[],
            value=None,
            interactive=True,
            visible=True,
            allow_custom_value=True,
        )
    }
    plots = {
        "left": gr.ScatterPlot(
            height=500,
            visible=False,
        ),
        "right": gr.ScatterPlot(
            height=500,
            visible=False,
        ),
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

if __name__ == "__main__":
    (
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
    ) = _initial_setup()
    with gr.Blocks(theme=THEME) as demo:
        with open(HEADER_FILE, 'r') as f:
            header_md = f.read()
        gr.Markdown(header_md)

        with gr.Tab(label="Paste one per line"):
            examples = gr.Examples(
                examples=[
                    [
                        "\n".join(eg["strings"]), 
                        "smiles", 
                        eg["species"], 
                        list(EXTRA_METRICS)[:2],
                    ] 
                    for eg in EXAMPLES["line input examples"]
                ],
                example_labels=[
                    eg["label"] for eg in EXAMPLES["line input examples"]
                ],
                inputs=[
                    line_inputs["strings"], 
                    line_inputs["format"], 
                    line_inputs["species"],
                    line_inputs["extras"],
                ],
                fn=_predict_then_draw_then_download,
                outputs=[
                    output_line,
                    drawing,
                    download_single,
                ],
                cache_examples=True,
                cache_mode="eager",
            )

            for val in line_inputs.values():
                val.render()
            # with gr.Row():
            output_line.render()
            download_single.render()
            drawing.render()
            line_inputs["strings"].submit(
                fn=_predict_then_draw_then_download,
                inputs=[
                    line_inputs["strings"], 
                    line_inputs["format"], 
                    line_inputs["species"],
                    line_inputs["extras"],
                ],
                outputs=[
                    output_line,
                    drawing,
                    download_single,
                ],
            )
        with gr.Tab(f"Predict on structures from a file (max. {MAX_ROWS} rows, ≤ 2 species)"):
            plot_dropdowns = list(itertools.chain(
                left_plot_inputs.values(),
                right_plot_inputs.values(), 
            ))
            file_examples = gr.Examples(
                examples=[
                    [
                        eg["file"],
                        eg["column"], 
                        "smiles",
                        eg["species"],
                        "",
                        list(EXTRA_METRICS)[:2],
                    ] for eg in EXAMPLES["file examples"]
                ],
                example_labels=[
                    eg["label"] for eg in EXAMPLES["file examples"]
                ],
                fn=_load_then_predict_then_download_then_reveal_plot,
                inputs=[
                    file_inputs["file"],
                    file_inputs["column"], 
                    file_inputs["format"], 
                    *file_inputs["species"],
                    file_inputs["extras"],
                ],
                outputs=[
                    input_dataframe,
                    download,
                    *plot_dropdowns,
                ],
                cache_examples=True,  ## appears to cause CSV load error
                cache_mode="eager",
            )
            file_inputs["file"].render()
            with gr.Row():
                for key in ("column", "format"):
                    file_inputs[key].render()
            with gr.Row():
                for item in file_inputs["species"]:
                    item.render()
            file_inputs["extras"].render()
            
            go_button2 = gr.Button(value="Predict!")

            input_dataframe.render()
            download.render()
            with gr.Row():
                for val in left_plot_inputs.values():
                    val.render()
            with gr.Row():
                for val in right_plot_inputs.values():
                    val.render()
            plot_button.render()

            with gr.Row():
                for val in plots.values():
                    val.render()
            
            file_inputs["file"].upload(
                fn=load_input_data,
                inputs=file_inputs["file"],
                outputs=[
                    input_dataframe, 
                    file_inputs["column"],
                ],
            )
            go2_click_event = go_button2.click(
                _load_then_predict_then_download_then_reveal_plot,
                inputs=[
                    file_inputs["file"],
                    file_inputs["column"], 
                    file_inputs["format"], 
                    *file_inputs["species"],
                    file_inputs["extras"],
                ],
                outputs=[
                    input_dataframe,
                    download,
                    *plot_dropdowns,
                ],
                scroll_to_output=True,
            ).then(
                lambda: gr.Button(visible=True),
                outputs=[plot_button],
                js=True,
            )

            file_examples.load_input_event.then(
                lambda: gr.Button(visible=True),
                outputs=[plot_button],
                js=True,
            )

            plot_button.click(
                plot_pred_vs_observed,
                inputs=[
                    input_dataframe,
                    file_inputs["species"][0],
                    left_plot_inputs["observed"],
                    left_plot_inputs["color"],
                ],
                outputs=[plots["left"]],
                scroll_to_output=True,
            ).then(
                plot_x_vs_y,
                inputs=[
                    input_dataframe,
                    right_plot_inputs["x"],
                    right_plot_inputs["y"],
                    right_plot_inputs["color"],
                ],
                outputs=[plots["right"]],
            )
    demo.queue()
    demo.launch(share=True)