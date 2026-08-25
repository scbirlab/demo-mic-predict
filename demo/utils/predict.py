from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Union
if TYPE_CHECKING:
    from pandas import DataFrame
    from io import TextIOWrapper
else:
    DataFrame, TextIOWrapper = Any, Any

from carabiner import cast, print_err
import gradio as gr
import numpy as np

from .convert import convert_file, convert_one
from .config import CACHE, EXTRA_METRICS, MAX_ROWS, MODELBOXES
from .interface import get_dropdown_options


def _prediction_loop(
    df: DataFrame,
    species: str,
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None
) -> DataFrame:
    # species_to_predict = cast(predict, to=list)
    prediction_cols = []
    if extra_metrics is None:
        extra_metrics = []
    else:
        extra_metrics = cast(extra_metrics, to=list)

    this_modelbox = MODELBOXES["Wild type"]
    this_features = this_modelbox._input_cols
    this_labels = this_modelbox._label_cols
    this_use_context = this_modelbox._use_context
    this_prediction_input = (
        df
        .assign(**{
            label: np.nan 
            for label in this_labels
        })
    )
    print_err(f"{this_features=}", f"{this_labels=}", f"{this_use_context=}")
    print_err(f"{this_prediction_input=}")
    print_err(f"{this_modelbox=}")

    prediction = np.asarray(
        this_modelbox.predict(
            data=this_prediction_input,
            aggregator="mean",
            cache=CACHE,
        )
        .with_format("numpy")
        [this_modelbox._prediction_key]
    ).flatten()
    print_err(f"{prediction=}")
    this_col = f"Predicted MIC (µM): {species}"
    df[this_col] = np.power(10., -prediction) * 1_000_000.
    prediction_cols.append(this_col)
    this_col = f"Predicted MIC (µg / mL): {species}"
    df[this_col] = np.power(10., -prediction) * 1000. * df["mwt"]
    prediction_cols.append(this_col)

    for extra_metric in extra_metrics:
        message = f"Calculating {extra_metric}"
        print_err(message), gr.Info(message, duration=10)
        # this_modelbox._input_training_data = this_modelbox._input_training_data.remove_columns([this_modelbox._in_key])
        this_col = f"{extra_metric}"
        prediction_cols.append(this_col)
        print_err(">>>", this_modelbox._input_training_data)
        print_err(">>>", this_modelbox._input_training_data.format)
        print_err(">>>", this_modelbox._in_key, this_modelbox._out_key)
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
    species: str = "Escherichia coli",
    input_representation: str = 'smiles', 
    predict: Union[Iterable[str], str] = 'smiles', 
    extra_metrics: Optional[Union[Iterable[str], str]] = None,
    return_pd: bool = False
):
    prediction_df = convert_one(
        strings=strings,
        input_representation=input_representation,
        output_representation=[
            'id', 
            'pubchem_name', 'pubchem_id', 
            'smiles', 'inchikey', "mwt", "clogp",
        ],
    )
    prediction_df["full_strain_name"] = species
    prediction_df["mic_method"] = "Broth microdilution"
    prediction_cols, prediction_df = _prediction_loop(
        prediction_df,
        species=species,
        predict=predict,
        extra_metrics=extra_metrics,
    )
    df = prediction_df[
        [
            'id', #'pubchem_name', 'pubchem_id'
        ] 
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
        return df, gr.update(**gradio_opts)
    else:
        return gr.update(**gradio_opts)


def predict_file(
    df: DataFrame, 
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
        print_err(message), gr.Info(message, duration=15)
        df = df.iloc[:MAX_ROWS]

    prediction_df = convert_file(
        df,
        column=column,
        input_representation=input_representation,
        output_representation=["id", "smiles", "inchikey", "mwt", "clogp"],
    )
    preds = []
    pred_cols = []
    for _species in predict:
        prediction_df["full_strain_name"] = _species
        prediction_df["mic_method"] = "Broth microdilution"
        prediction_cols, predicted = _prediction_loop(
            prediction_df,
            species=_species,
            predict=predict,
            extra_metrics=extra_metrics,
        )
        preds.append(predicted)
        pred_cols += prediction_cols
    prediction_df = preds[0]
    for _pred in preds[1:]:
        prediction_df = prediction_df.merge(_pred, how="left")
    pred_cols = sorted(set(pred_cols))
    print_err(f"{prediction_df=}")
    left_cols = ['id', 'inchikey']
    end_cols = ["smiles", "mwt", "clogp"] 
    main_cols = set(
        left_cols 
        + end_cols
        + [column] 
        + prediction_cols
    )
    other_cols = list(set(prediction_df) - main_cols)
    return_cols = (
        left_cols 
        + [column] 
        + prediction_cols 
        + other_cols 
        + end_cols
    )
    deduplicated_cols = []
    for col in return_cols:
        if not col in deduplicated_cols:
            deduplicated_cols.append(col)
    prediction_df = prediction_df[deduplicated_cols]

    plot_dropdown = get_dropdown_options(prediction_df, _type="number")
    plot_dropdown = tuple(
        get_dropdown_options(prediction_df, _type="number")
        for _ in range(5)
    )
    gradio_opts = {
        "label": "Predictions",
        "value": prediction_df,
        "pinned_columns": 3,
        "visible": True,
        "wrap": True,
        "column_widths": [120] * 3 + [250] * (prediction_df.shape[1] - 3),
    }

    if return_pd:
        return ((prediction_df, gr.update(**gradio_opts)), plot_dropdown)
    else:
        return (gr.update(**gradio_opts), plot_dropdown)
