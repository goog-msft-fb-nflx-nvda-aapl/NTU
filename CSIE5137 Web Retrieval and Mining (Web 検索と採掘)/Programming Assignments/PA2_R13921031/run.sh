#!/bin/bash

# Default to BPR-MF if no algorithm is specified
ALGORITHM=${2:-"bpr"}

if [ "$#" -lt 1 ]; then
    echo "Usage: ./run.sh <output_path> [algorithm]"
    echo "  output_path: Path to save the recommendations"
    echo "  algorithm: 'bpr' or 'bce' (default: bpr)"
    exit 1
fi

OUTPUT_PATH=$1

# Run the specified algorithm
if [ "$ALGORITHM" == "bpr" ]; then
    echo "Running BPR-MF algorithm..."
    python BPR-MF.py $OUTPUT_PATH
elif [ "$ALGORITHM" == "bce" ]; then
    echo "Running BCE-MF algorithm..."
    python BCE-MF.py $OUTPUT_PATH
else
    echo "Invalid algorithm: $ALGORITHM. Use 'bpr' or 'bce'."
    exit 1
fi

echo "Finished running $ALGORITHM-MF algorithm. Results saved to $OUTPUT_PATH"