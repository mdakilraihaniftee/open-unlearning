#!/bin/bash

# Continual Unlearning Execution Script
# This script runs the continual test-time unlearning loop.

METHOD=${1:-graddiff} # Default to graddiff, can be "ascent"
MAX_STEPS=${2:-100}   # Default to 100 steps for verification, set to -1 for full run

if [ "$METHOD" == "ascent" ]; then
    EXPERIMENT="unlearn/beavertails/continual"
elif [ "$METHOD" == "graddiff" ]; then
    EXPERIMENT="unlearn/beavertails/continual_graddiff"
else
    echo "Unknown method: $METHOD. Use 'ascent' or 'graddiff'."
    exit 1
fi

echo "Running Continual Unlearning with $METHOD method..."

python src/train.py \
    experiment=$EXPERIMENT \
    mode=unlearn \
    trainer.args.max_steps=$MAX_STEPS \
    trainer.args.eval_on_start=false \
    trainer.args.do_eval=false \
    trainer.args.eval_strategy=no \
    +trainer.args.remove_unused_columns=false \
    eval=null
