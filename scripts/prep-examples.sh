#!/usr/bin/env bash

INPUT="$1"
OUTPUT="$2"
FIELDS="$3"
NLINES="${4:-1000}"

python -m venv .schemist && .schemist/bin/pip install "pandas schemist>=0.0.4" && .schemist/bin/activate

# Some functions for convenience 
logger () (
    local message="$1"
    local _date=$(date)
    local prefix=${2:-"$_date"}
    >&2 echo "$prefix :: $message"
)

pandas () (
    local cmd="$1"
    local sep1=${2:-,}
    local idx=${3:-False}
    local sep2=${4:-"$sep1"}
    python -c 'import sys; import pandas as pd; df = pd.read_csv(sys.stdin, sep="'"$sep1"'", low_memory=False)'"$cmd"'.to_csv(sys.stdout, index='"$idx"', sep="'"$sep2"'")'
)

set -e
set -x

pandas '[['"$FIELDS"']].sample('"$NLINES"')' \
< "$INPUT" \
| schemist convert -c SMILES -2 id smiles inchikey pubchem_id mwt clogp tpsa -f CSV \
| pandas '.sort_values(["id"])' \
> "$OUTPUT"
