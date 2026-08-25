

from typing import Iterable, Optional, Union
import os

from carabiner import print_err
import gradio as gr
import nemony as nm
import pandas as pd

from .config import CACHE
from .convert import load_input_data
from .predict import predict_one, predict_file
from .drawing import draw_one


def download_table(
    df: pd.DataFrame
) -> str:
    df_hash = nm.hash(pd.util.hash_pandas_object(df).values)
    filename = os.path.join(CACHE, "downloads", f"predicted-{df_hash}.csv")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df.to_csv(filename, index=False)
    return gr.update(
        value=filename, 
        visible=True,
    )


def _predict_then_draw_then_download(
    strings: str,
    input_representation: str = 'smiles', 
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None,
    smiles_col: str = "smiles",
    legends: Optional[Union[str, Iterable[str]]] = None
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
    extra_metrics: Optional[Union[Iterable[str], str]] = None
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
    print_err(f"{df.head()=}")
    return (
        df_gr, 
        download_table(df),
    ) + plot_opts
