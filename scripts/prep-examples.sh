#!/usr/bin/env bash

set -euox pipefail

INPUTS=('hf://datasets/scbirlab/stokes-2020-ai/*-eco0-*.csv.gz' 'hf://datasets/scbirlab/wong-2024-ai/*-*.csv.gz'  'hf://datasets/scbirlab/liu-2023-ai/*-*.csv.gz')
NAMES=('Escherichia coli K-12' 'Staphylococcus aureus USA300' 'Acinetobacter baumannii ATCC 17978')
OUTPUTS=('stokes20-eco-1000.csv' 'wong24-sau-tox-1000.csv' 'liu23-abau-1000.csv')
FIELDS="${1:-'"id", "inchikey", "pubchem_name", "pubchem_id", "smiles", "scaffold", "mwt", "clogp", "tpsa", "normalized_inhibition"'}"
NLINES="${2:-100}"

python -m venv .schemist \
&& .schemist/bin/pip install "pandas" "polars" "pyarrow" "schemist>=0.0.4" \
&& source .schemist/bin/activate


for i in "${!INPUTS[@]}"
do 
    output=data/examples/"${OUTPUTS[$i]}"
    source scripts/prep-example.sh "${INPUTS[$i]}" "${NAMES[$i]}" "$output" "$FIELDS" $NLINES
done
