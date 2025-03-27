"""Gradio demo for schemist."""

from typing import Iterable, List, Optional, Union
from functools import partial
from io import TextIOWrapper
import os
# os.environ["COMMANDLINE_ARGS"] = "--no-gradio-queue"

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

CACHE = "./cache"
HEADER_FILE = os.path.join("sources", "header.md")
MODEL_REPOS = {
    "Klebsiella pneumoniae": "hf://scbirlab/spark-dv-fp-2503-kpn",
}

MODELBOXES = {
    key: AutoModelBox.from_pretrained(val, cache_dir=CACHE)
    for key, val in MODEL_REPOS.items()
}

EXTRA_METRICS = {
    "log10(variance)": lambda modelbox, candidates: modelbox.prediction_variance(candidates=candidates, cache=CACHE).map(lambda x: {modelbox._variance_key: torch.log10(x[modelbox._variance_key])}), 
    "Tanimoto nearest neighbor to training data": lambda modelbox, candidates: modelbox.tanimoto_nn(candidates=candidates), 
    "Doubtscore": lambda modelbox, candidates: modelbox.doubtscore(candidates=candidates, cache=CACHE).map(lambda x: {"doubtscore": torch.log10(x["doubtscore"])}), 
    "Information sensitivity (approx.)": lambda modelbox, candidates: modelbox.information_sensitivity(candidates=candidates, optimality_approximation=True, approximator="squared_jacobian", cache=CACHE).map(lambda x: {"information sensitivity": torch.log10(x["information sensitivity"])}),
}

def get_dropdown_options(df, _type = str):
    if _type == str:
        cols = list(df.select_dtypes(exclude=[np.number]))
    else:
        cols = list(df.select_dtypes([np.number]))
    return gr.Dropdown(choices=cols, interactive=True, value=cols[0], visible=True)

def load_input_data(file: Union[TextIOWrapper, str]) -> pd.DataFrame:
    file = file if isinstance(file, str) else file.name
    print_err(f"Loading {file}")
    df = read_table(file)
    print_err(df.head())
    return gr.Dataframe(value=df, visible=True), get_dropdown_options(df, str)
    

def _clean_split_input(strings: str) -> List[str]:
    return [s2.strip() for s in strings.split("\n") for s2 in s.split(",")]


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
    output_representation: Union[Iterable[str], str] = 'smiles'
):

    df = pd.DataFrame({
        input_representation: _clean_split_input(strings),
    })

    return convert_file(
        df=df,
        column=input_representation,
        input_representation=input_representation,
        output_representation=output_representation,
    )


def predict_one(
    strings: str,
    input_representation: str = 'smiles', 
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None
):
    if extra_metrics is None:
        extra_metrics = []
    else:
        extra_metrics = cast(extra_metrics, to=list)
    prediction_df = convert_one(
        strings=strings,
        input_representation=input_representation,
        output_representation=['id', 'pubchem_name', 'pubchem_id', 'smiles', 'inchikey', "mwt", "clogp"],
    )
    species_to_predict = cast(predict, to=list)
    prediction_cols = []
    for species in species_to_predict:
        message = f"Predicting for species: {species}"
        print_err(message)
        gr.Info(message, duration=3)
        this_modelbox = MODELBOXES[species]
        this_features = this_modelbox._input_cols
        this_labels = this_modelbox._label_cols
        this_prediction_input = (
            prediction_df
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
        prediction_df[this_col] = np.power(10., -prediction) * 1e6
        prediction_cols.append(this_col)
        this_col = f"{species}: predicted MIC (µg / mL)"
        prediction_df[this_col] = np.power(10., -prediction) * 1e3 * prediction_df["mwt"]
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
            prediction_df[this_col] = this_extra[this_extra.column_names[-1]]
        
    return gr.DataFrame(
        prediction_df[['id', 'pubchem_name', 'pubchem_id'] + prediction_cols + ['smiles', 'inchikey', "mwt", "clogp"]],
        visible=True
    )


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
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None
):
    if extra_metrics is None:
        extra_metrics = []
    else:
        extra_metrics = cast(extra_metrics, to=list)

    prediction_df = convert_file(
        df,
        column=column,
        input_representation=input_representation,
        output_representation=["id", "smiles", "inchikey", "mwt", "clogp"],
    )
    if prediction_df.shape[0] > 1000:
        message = f"Truncating input to 1000 rows"
        print_err(message)
        gr.Info(message, duration=15)
        prediction_df = prediction_df.iloc[:1000]
    species_to_predict = cast(predict, to=list)
    prediction_cols = []
    for species in species_to_predict:
        message = f"Predicting for species: {species}"
        print_err(message)
        gr.Info(message, duration=3)
        this_modelbox = MODELBOXES[species]
        this_features = this_modelbox._input_cols
        this_labels = this_modelbox._label_cols
        this_prediction_input = (
            prediction_df
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
        prediction_df[this_col] = np.power(10., -prediction) * 1e6
        prediction_cols.append(this_col)
        this_col = f"{species}: predicted MIC (µg / mL)"
        prediction_df[this_col] = np.power(10., -prediction) * 1e3 * prediction_df["mwt"]
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
            prediction_df[this_col] = this_extra[this_extra.column_names[-1]]
    other_cols = [col for col in prediction_df if col not in ['id', 'inchikey', 'smiles', "mwt", "clogp"] + [column] + prediction_cols]

    return prediction_df[['id', 'inchikey'] + [column] + prediction_cols + other_cols + ['smiles', "mwt", "clogp"]]

def draw_one(
    strings: Union[Iterable[str], str],
    input_representation: str = 'smiles'
):
    _ids = _convert_input(
        strings, 
        input_representation, 
        ["inchikey", "id", "pubchem_name"],
    )
    mols = cast(_x2mol(_clean_split_input(strings), input_representation), to=list)
    if isinstance(mols, Mol):
        mols = [mols]
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=min(3, len(mols)), 
        subImgSize=(450, 450),
        legends=["\n".join(items) for items in zip(*_ids.values())],
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
    y_title = f"Observed ({ycol})"
    cols = ["id", "inchikey", "smiles", "mwt", "clogp", xcol, ycol]
    color_title = color
    if color is not None and color not in cols:
        cols.append(color)
    cols = list(set(cols))
    print_err(df[cols].columns)
    if np.all(df[xcol] > 0):
        df[xcol] = np.log10(df[xcol])
        x_title = f"Predicted log10[MIC(µM)]"

    return gr.ScatterPlot(
        value=df[cols],
        x=xcol,
        y=ycol,
        color=color,
        x_title=x_title,
        y_title=y_title,
        color_title=color_title,
        tooltip="all",
        visible=True,
    )


def download_table(
    df: pd.DataFrame
) -> str:
    df_hash = nm.hash(pd.util.hash_pandas_object(df).values)
    filename = f"converted-{df_hash}.csv"
    df.to_csv(filename, index=False)
    return gr.DownloadButton(value=filename, visible=True)

with gr.Blocks() as demo:

    with open(HEADER_FILE, 'r') as f:
        header_md = f.read()
    gr.Markdown(header_md)

    with gr.Tab(label="Paste one per line"):
        input_format_single = gr.Dropdown(
            label="Input string format",
            choices=list(_FROM_FUNCTIONS),
            value="smiles",
            interactive=True,
        )
        input_line = gr.Textbox(
            label="Input",
            placeholder="Paste your molecule here, one per line",
            lines=2,
            interactive=True,
            submit_btn=True,
        )
        output_species_single = gr.CheckboxGroup(
            label="Species for prediction",
            choices=list(MODEL_REPOS),
            value=list(MODEL_REPOS)[:1],
            interactive=True,
        )
        extra_metric = gr.CheckboxGroup(
            label="Extra metrics (Information Sensitivity can increase calculation time!)",
            choices=list(EXTRA_METRICS),
            value=list(EXTRA_METRICS)[:2],
            interactive=True,
        )
        examples = gr.Examples(
            examples=[
                [
                    '\n'.join([
                        "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
                        "CN1C(=NC(=O)C(=O)N1)SCC2=C(N3[C@@H]([C@@H](C3=O)NC(=O)/C(=N\OC)/C4=CSC(=N4)N)SC2)C(=O)O",
                        "CC(C)(C(=O)O)O/N=C(/C1=CSC(=N1)N)\C(=O)N[C@H]2[C@@H]3N(C2=O)C(=C(CS3)C[N+]4(CCCC4)CCNC(=O)C5=C(C(=C(C=C5)O)O)Cl)C(=O)[O-]",
                        "CC(=O)NC[C@H]1CN(C(=O)O1)C2=CC(=C(C=C2)N3CCOCC3)F",
                        "C1CC2=CC(=NC=C2OC1)CNC3CCN(CC3)C[C@@H]4CN5C(=O)C=CC6=C5N4C(=O)C=N6",
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                 ],  # cipro, ceftriaxone, cefiderocol, linezolid, gepotidacin
                [
                    '\n'.join([
                        "C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",
                        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C",
                        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C", 
                        "C[C@@H]1[C@@H]2[C@H](C(=O)N2C(=C1S[C@H]3C[C@H](NC3)C(=O)N(C)C)C(=O)O)[C@@H](C)O",
                        "C[C@@]1([C@H]2C[C@H]3[C@@H](C(=O)C(=C([C@]3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
                        "CC1=C2C=CC=C(C2=C(C3=C1C[C@H]4[C@@H](C(=O)C(=C([C@]4(C3=O)O)O)C(=O)N)N(C)C)O)O",
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                ],  # doxorubicin, ampicillin, amoxicillin, meropenem, tetracycline, anhydrotetracycline
                [
                    '\n'.join([
                        "C1=C(SC(=N1)SC2=NN=C(S2)N)[N+](=O)[O-]",
                        "C1CN(CCC12C3=CC=CC=C3NC(=O)O2)CCC4=CC=C(C=C4)C(F)(F)F",
                        "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
                        "CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N",
                        "C1[C@@H]([C@H]([C@@H]([C@H]([C@@H]1NC(=O)[C@H](CCN)O)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)N)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CN)O)O)O)N\nC1=CN=CC=C1C(=O)NN", 
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                ],  # Halicin, Abaucin, Trimethoprim, Sulfamethoxazole, Amikacin, Isoniazid
                [
                     '\n'.join([
                        "CC[C@H](C)[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N2CCC[C@@H]2C(=O)N3CCC[C@H]3C(=O)N[C@H](C(=O)N[C@H](C(=O)N1)CC4=CNC5=CC=CC=C54)[C@@H](C)O)CO)C)CCN)CCN)CC6=CNC7=CC=CC=C76)CCN)CCN)CCCN)CCN",
                        "C[C@H]1[C@H]([C@@](C[C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)[C@H]([C@H](C(=O)N[C@H](C(=O)N[C@H]5C(=O)N[C@@H]7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)[C@H](NC(=O)[C@H]([C@@H](C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)[C@@H](CC(C)C)NC)O)Cl)CO)O)O)(C)N)O",
                        "CN1[C@H](C(=O)NCC2=C(C=CC=C2SC3=C(CN[C@H](C(=O)N[C@H](C1=O)CCCCN)CCCN)C=CC=N3)C4=CC=C(C=C4)C(=O)O)CC5=CNC6=CC=CC=C65", 
                        "C[C@@]1(CO[C@@H]([C@@H]([C@H]1NC)O)O[C@H]2[C@@H](C[C@@H]([C@H]([C@@H]2O)O[C@@H]3[C@@H](CC=C(O3)CNCCO)N)N)NC(=O)[C@H](CCN)O)O",
                        "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC",
                        "C[C@H]1/C=C/C=C(\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C",
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                ],  # murepavadin, vancomycin, zosurabalpin, plazomicin, Gentamicin, rifampicin
                [
                     '\n'.join([
                        "CC1=C(OC2=CC=CC=C12)CN(C)C(=O)/C=C/C3=CC4=C(NC(=O)CC4)N=C3",
                        "CC1=C(OC2=CC=CC=C12)CN(C)C(=O)/C=C/C3=CC4=C(NC(=O)[C@@H](C4)N)N=C3",
                        "CC1=C(OC2=CC=CC=C12)CN(C)C(=O)/C=C/C3=CC4=C(NC(=O)[C@H](CC4)[NH3+])N=C3.[Cl-]",
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                ],  # Debio1452, Debio-1452-NH3, Fabimycin, 

            ],
            example_labels=[
                "Ciprofloxacin, Ceftriaxone, Cefiderocol, Linezolid, Gepotidacin",
                "Doxorubicin, Ampicillin, Amoxicillin, Meropenem, Tetracycline, Anhydrotetracycline",
                "Halicin, Abaucin, Trimethoprim, Sulfamethoxazole, Amikacin, Isoniazid",
                "Murepavadin, Vancomycin, Zosurabalpin, Plazomicin, Gentamicin, Rifampicin",
                "Debio-1452, Debio-1452-NH3, Fabimycin",

            ],
            inputs=[input_line, output_species_single, extra_metric],
            cache_mode="eager",
        )
        download_single = gr.DownloadButton(
            label="Download predictions",
            visible=False,
        )
        # with gr.Row():
        output_line = gr.DataFrame(
            label="Predictions",
            interactive=False,
            visible=False,
        )
        drawing = gr.Image(label="Chemical structures")

        gr.on(
            [
                input_line.submit,
            ],
            fn=predict_one,
            inputs=[
                input_line, 
                input_format_single,
                output_species_single,
                extra_metric,
            ],
            outputs={
                output_line,
            }
        ).then(
            draw_one,
            inputs=[
                input_line, 
                input_format_single,
            ],
            outputs=drawing,
        ).then(
            download_table,
            inputs=output_line,
            outputs=download_single
        )

    with gr.Tab("Predict on structures from a file (max. 1000 rows, single species)"):
        input_file = gr.File(
            label="Upload a table of chemical compounds here",
            file_types=[".xlsx", ".csv", ".tsv", ".txt"],
        )
        with gr.Row():
            input_column = gr.Dropdown(
                label="Input column name",
                choices=[],
                allow_custom_value=True,
                visible=False,
            )
            input_format = gr.Dropdown(
                label="Input string format",
                choices=list(_FROM_FUNCTIONS),
                value="smiles",
                interactive=True,
                visible=True,
            )
        output_species = gr.Radio(
            label="Species for prediction",
            choices=list(MODEL_REPOS),
            value=list(MODEL_REPOS)[0],
            interactive=True,
        )
        extra_metric_file = gr.CheckboxGroup(
            label="Extra metrics (Information Sensitivity can increase calculation time)",
            choices=list(EXTRA_METRICS),
            value=list(EXTRA_METRICS)[:2],
            interactive=True,
        )
        
        go_button2 = gr.Button(
            value="Predict!",
        )

        download = gr.DownloadButton(
            label="Download predictions",
            visible=False,
        )
        input_data = gr.Dataframe(
            label="Input data",
            max_height=500,
            visible=False,
            interactive=False,
        )
        with gr.Row():
            observed_col = gr.Dropdown(
                label="Observed column (y-axis) for comparison plot",
                choices=[],
                value=None,
                interactive=True,
                visible=False,
            )
            color_col = gr.Dropdown(
                label="Color for comparison plot",
                choices=[],
                value=None,
                interactive=True,
                visible=False,
            )
        plot_button = gr.Button(
            value="Plot!",
            visible=False,
        )
        file_examples = gr.Examples(
            examples=[
                [
                    "example-data/stokes2020-eco-1000.csv", 
                    "SMILES", 
                    "Klebsiella pneumoniae",
                    "Mean_Inhibition",
                    "Klebsiella pneumoniae: Doubtscore",
                    list(EXTRA_METRICS)[:3]],
            ],
            example_labels=[
                "Stokes J. et al., Cell, 2020",
            ],
            inputs=[input_file, input_column, output_species, observed_col, color_col, extra_metric_file],
            cache_mode="eager",
        )
        pred_vs_observed = gr.ScatterPlot(
            label="Prediction vs observed", 
            x_title="Predicted MIC (µM)",
            y_title="Observed",
            visible=False,
            height=600,
        )
        
        file_examples.load_input_event.then(
            load_input_data, 
            inputs=[input_file], 
            outputs=[input_data, input_column],
        )
        input_file.upload(
            load_input_data, 
            inputs=[input_file], 
            outputs=[input_data, input_column]
        )
        go_button2.click(
            predict_file,
            inputs=[
                input_data, 
                input_column,
                input_format,
                output_species,
                extra_metric_file,
            ],
            outputs={
                input_data,
            }
        ).then(
            download_table,
            inputs=input_data,
            outputs=download
        ).then(
            partial(get_dropdown_options, _type="number"),
            inputs=[input_data],
            outputs=[observed_col],
        ).then(
            partial(get_dropdown_options, _type="number"),
            inputs=[input_data],
            outputs=[color_col],
        ).then(
            lambda: gr.Button(visible=True),
            outputs=[plot_button],
        )

        plot_button.click(
            plot_pred_vs_observed,
            inputs=[
                input_data,
                output_species,
                observed_col,
                color_col,
            ],
            outputs=pred_vs_observed,
        )

if __name__ == "__main__":
    demo.queue() 
    demo.launch(share=True)

