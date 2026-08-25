from typing import List, Optional

from carabiner import print_err
import gradio as gr
import numpy as np


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
    print_err(f"{df.head()=}")
    y_title = y
    cols = ["id", "inchikey", "smiles", "mwt", "clogp", x, y]
    if color is not None and color not in cols:
        cols.append(color)
    cols = list(set(cols))
    x_title, df = log10_if_all_positive(df, x)
    y_title, df = log10_if_all_positive(df, y)
    color_title, df = log10_if_all_positive(df, color)

    return gr.update(
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
    xcol = f"Predicted MIC (µM): {species}"
    ycol = observed
    return plot_x_vs_y(
        df,
        x=xcol,
        y=ycol,
        color=color,
    ) 