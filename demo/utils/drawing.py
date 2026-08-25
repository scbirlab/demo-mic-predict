
from typing import Iterable, List, Optional, Union
from io import TextIOWrapper

from carabiner import cast
import gradio as gr
from rdkit.Chem import Draw, Mol
from schemist.converting import (
    _FROM_FUNCTIONS, 
    convert_string_representation, 
    _x2mol,
)

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
    gr.Info(message, duration=2)
    _ids = {
        col: df[col].tolist() 
        for col in legends 
        if col in df
    }
    mols = cast(_x2mol(df[smiles_col], "smiles"), to=list)
    if isinstance(mols, Mol):
        mols = [mols]
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=min(5, len(mols)), 
        subImgSize=(600, 600),
        legends=[
            "\n".join(
                _x if _x is not None else "" 
                for _x in items
            ) for items in zip(*_ids.values())
        ],
    )
