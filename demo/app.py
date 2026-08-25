"""Gradio demo for schemist."""

from typing import List
import itertools

import gradio as gr
import numpy as np
import pandas as pd

from utils.config import (
    EXAMPLES,
    EXTRA_METRICS,
    HEADER_FILE,
    MAX_ROWS,
    THEME
)
from utils.convert import load_input_data
from utils.interface import _initial_setup
from utils.plots import plot_x_vs_y, plot_pred_vs_observed
from utils.routines import _predict_then_draw_then_download, _load_then_predict_then_download_then_reveal_plot

if __name__ == "__main__":
    with gr.Blocks() as demo:
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
                        list(EXTRA_METRICS)[:1],
                    ] 
                    for eg in EXAMPLES["line input examples"]
                ],
                example_labels=[
                    eg["label"] for eg in EXAMPLES["line input examples"]
                ],
                examples_per_page=100,
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

            for val in itertools.chain(
                line_inputs.values(), 
                (output_line, download_single, drawing),
            ):
                val.render()

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
                        list(EXTRA_METRICS)[:1],
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
                # cache_examples=True,  ## appears to cause CSV load error
                # cache_mode="eager",
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
                lambda: gr.update(visible=True),
                outputs=[plot_button],
                js=True,
            )

            file_examples.load_input_event.then(
                lambda: gr.update(visible=True),
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
    # demo.queue()
    demo.launch(theme=THEME, share=True)
