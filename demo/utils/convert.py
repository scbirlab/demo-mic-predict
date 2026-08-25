
from typing import TYPE_CHECKING, Any, Iterable, List, Union

if TYPE_CHECKING:
    from pandas import DataFrame
    from io import TextIOWrapper
else:
    DataFrame, TextIOWrapper = Any, Any

from carabiner import cast, print_err
from carabiner.pd import read_table
import gradio as gr
import numpy as np
from schemist.converting import convert_string_representation
from schemist.tables import converter

from .config import MAX_ROWS
from .interface import get_dropdown_options


def load_input_data(
    file: Union[TextIOWrapper, str], return_pd: bool = False) -> DataFrame:
    file = file if isinstance(file, str) else file.name
    print_err(f"Loading {file}")
    df = read_table(file, nrows=MAX_ROWS)
    print_err(f"{df.head()=}")
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
    import pandas as pd
    output_representation = cast(output_representation, to=list)
    for rep in output_representation:
        message = f"Converting from {input_representation} to {rep}..."
        gr.Info(message, duration=3)

    df = pd.DataFrame({
        input_representation: _clean_split_input(strings),
    })

    return convert_file(
        df=df,
        column=input_representation,
        input_representation=input_representation,
        output_representation=output_representation,
    )


def convert_file(
    df: DataFrame, 
    column: str = 'smiles',
    input_representation: str = 'smiles',
    output_representation: Union[str, Iterable[str]] = 'smiles'
):
    output_representation = cast(output_representation, to=list)
    message = f"Converting from {input_representation} to {', '.join(output_representation)}..."
    gr.Info(message, duration=5)
    print_err(message)
    print_err(df.shape)
    print_err(df.head())

    converted = convert_string_representation(
        strings=df[column].tolist(), 
        input_representation=input_representation, 
        output_representation=output_representation,
    )
    converted = {key: cast(v, to=list) for key, v in converted.items()}
    print_err(converted)
    df = df.assign(**converted)
    print_err(df)
    df = df[
        output_representation +
        [col for col in df if col not in output_representation]
    ]
    print_err(df)
    all_err = sum(item is None for item in converted)
    message = (
        f"Converted {df.shape[0]} molecules from "
        f"{input_representation} to {output_representation} "
        f"with {all_err} errors!"
    )
    print_err(message)
    gr.Info(message, duration=5)
    return df
