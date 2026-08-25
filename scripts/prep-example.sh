#!/usr/bin/env bash

INPUT="$1"
STRAIN_NAME="$2"
OUTPUT="$3"
FIELDS="$4"
NLINES="${5:-100}"

set -euox pipefail

python -c '
import polars as pl

fields = ["id", "inchikey", "pubchem_name", "pubchem_id", "smiles", "scaffold", "mwt", "clogp", "tpsa", "normalized_inhibition"]
df = pl.read_csv("'"$INPUT"'")

if "Sau_Mean_10uM_inhibition" in df.columns:
    df = df.rename({"Sau_Mean_10uM_inhibition": "normalized_inhibition"})
if "Mean_inhibition" in df.columns:
    df = df.rename({"Mean_inhibition": "normalized_inhibition"})
df = (
    df
    .select(fields)
    .sample("'"$NLINES"'", seed=42)
    .sort("id")
    .with_columns(
        SMILES=pl.col("smiles"),
        mic_method=pl.lit("Broth microdilution"), 
        full_strain_name=pl.lit("'"$STRAIN_NAME"'"),
    )
)
df.write_csv("'"$OUTPUT"'")
'
