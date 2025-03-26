"""Gradio demo for schemist."""

from typing import Iterable, List, Optional, Union
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
    _TO_FUNCTIONS,
    _FROM_FUNCTIONS, 
    convert_string_representation, 
    _x2mol,
)
from schemist.tables import converter
import torch

HEADER_FILE = os.path.join("sources", "header.md")
MODEL_REPOS = {
    "Klebsiella pneumoniae": "hf://scbirlab/spark-dv-fp-2503-kpn",
}

MODELBOXES = {
    key: AutoModelBox.from_pretrained(val, cache_dir="./cache")
    for key, val in MODEL_REPOS.items()
}

EXTRA_METRICS = {
    "log10(variance)": lambda modelbox, candidates: modelbox.prediction_variance(candidates=candidates).map(lambda x: {modelbox._variance_key: torch.log10(x[modelbox._variance_key])}), 
    "Tanimoto nearest neighbor to training data": lambda modelbox, candidates: modelbox.tanimoto_nn(candidates=candidates), 
    "Doubtscore": lambda modelbox, candidates: modelbox.doubtscore(candidates=candidates).map(lambda x: {"doubtscore": torch.log10(x["doubtscore"])}), 
    "Information sensitivity (approx.)": lambda modelbox, candidates: modelbox.information_sensitivity(candidates=candidates, optimality_approximation=True, approximator="squared_jacobian").map(lambda x: {"information sensitivity": torch.log10(x["information sensitivity"])}),
}

def load_input_data(file: TextIOWrapper) -> pd.DataFrame:
    df = read_table(file.name)
    string_cols = list(df.select_dtypes(exclude=[np.number]))
    df = gr.Dataframe(value=df, visible=True)
    return df, gr.Dropdown(choices=string_cols, interactive=True)
    

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
        output_representation=['id', 'pubhem_name', 'pubchem_id', 'smiles', 'inchikey', "mwt", "clogp"],
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
            cache="./cache"
        ).with_format("numpy")["__prediction__"].flatten()
        print(prediction)
        this_col = f"{species}: predicted MIC (µM)"
        prediction_df[this_col] = np.power(10., -prediction) * 1e6
        prediction_cols.append(this_col)

        for extra_metric in extra_metrics:
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
        prediction_df[['id', 'pubhem_name', 'pubhem_id'] + prediction_cols + ['smiles', 'inchikey', "mwt", "clogp"]],
        visible=True
    )


def convert_file(
    df: pd.DataFrame, 
    column: str = 'smiles',
    input_representation: str = 'smiles',
    output_representation: Union[str, Iterable[str]] = 'smiles'
):
    message = f"Converting from {input_representation} to {output_representation}..."
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
    species_to_predict = cast(predict, to=list)
    prediction_cols = []
    for species in species_to_predict:
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
        prediction = this_modelbox.predict(
            data=this_prediction_input,
            features=this_features,
            labels=this_labels,
            cache="./cache"
        ).with_format("numpy")["__prediction__"].flatten()
        print(prediction)
        this_col = f"{species}: predicted MIC (µM)"
        prediction_df[this_col] = np.power(10., -prediction) * 1e6
        prediction_cols.append(this_col)

        for extra_metric in extra_metrics:
            # this_modelbox._input_training_data = this_modelbox._input_training_data.remove_columns([this_modelbox._in_key])
            this_col = f"{species}: {extra_metric}"
            prediction_cols.append(this_col)
            print(">>>", this_modelbox._input_training_data)
            this_extra = (
                EXTRA_METRICS[extra_metric](
                    this_modelbox,
                    this_prediction_input,
                )
                .with_format("numpy")
            )
            prediction_df[this_col] = this_extra[this_extra.column_names[0]]

    return prediction_df[['id'] + prediction_cols + ['smiles', 'inchikey', "mwt", "clogp"]]

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
            label="Extra metrics (can increase calculation time!)",
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
                        "CC(=O)NC[C@H]1CN(C(=O)O1)C2=CC(=C(C=C2)N3CCOCC3)F",
                        "C1CC2=CC(=NC=C2OC1)CNC3CCN(CC3)C[C@@H]4CN5C(=O)C=CC6=C5N4C(=O)C=N6",
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                 ],  # cipro, ceftriaxone, linezolid, gepotidacin
                [
                    '\n'.join([
                        "C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",
                        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C",
                        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C", 
                    ]), 
                    list(MODEL_REPOS)[0],
                    list(EXTRA_METRICS)[:2],
                ],  # doxorubicin, ampicillin, amoxicillin
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
            ],
            example_labels=[
                "Ciprofloxacin, Ceftriaxone, Linezolid, Gepotidacin",
                "Doxorubicin, Ampicillin, Amoxicillin",
                "Halicin, Abaucin, Trimethoprim, Sulfamethoxazole, Amikacin, Isoniazid"
            ],
            inputs=[input_line, output_species_single, extra_metric],
            cache_mode="eager",
        )
        download_single = gr.DownloadButton(
            label="Download predictions",
            visible=False,
        )
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

    with gr.Tab("Convert a file"):
        input_file = gr.File(
            label="Upload a table of chemical compounds here",
            file_types=[".xlsx", ".csv", ".tsv", ".txt"],
        )
        with gr.Row():
            input_column = gr.Dropdown(
                label="Input column name",
                choices=[],
            )
            input_format = gr.Dropdown(
                label="Input string format",
                choices=list(_FROM_FUNCTIONS),
                value="smiles",
                interactive=True,
            )
        output_species = gr.CheckboxGroup(
            label="Species for prediction",
            choices=list(MODEL_REPOS),
            value=list(MODEL_REPOS)[:1],
            interactive=True,
        )
        extra_metric_file = gr.CheckboxGroup(
            label="Extra metrics (can increase calculation time!)",
            choices=list(EXTRA_METRICS),
            value=list(EXTRA_METRICS)[:2],
            interactive=True,
        )
        go_button2 = gr.Button(
            value="Predict!",
        )

        download = gr.DownloadButton(
            label="Download converted data",
            visible=False,
        )
        input_data = gr.Dataframe(
            label="Input data",
            max_height=100,
            visible=False,
            interactive=False,
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
        )

if __name__ == "__main__":
    demo.queue() 
    demo.launch(share=True)

