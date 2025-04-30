# Vector Space Model with BM25 Implementation

This project implements a BM25-based information retrieval system for NTCIR Chinese news article retrieval with Rocchio relevance feedback.

## Overview

The system implements the following features:
- BM25 ranking algorithm for document retrieval
- Rocchio relevance feedback for query expansion
- Support for unigrams and bigrams in Chinese text
- Performance evaluation using Mean Average Precision (MAP)

## Known Issues

There is a significant bug in the query tokenization process. The current implementation does not properly handle mixed Chinese and English text, treating every character as a single token. A better approach would be to use specialized NLP tools like NLTK or jieba for proper tokenization.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - `xml.etree.ElementTree` (standard library)
  - `collections` (standard library)
  - `math` (standard library)
  - `time` (standard library)
  - `argparse` (standard library)
  - `csv` (standard library)
  - `os` (standard library)
  - `sys` (standard library)

No external dependencies beyond the Python standard library are required.

## File Structure

- `vsm.py`: Main implementation of the VSM_BM25 class and utility functions
- `compile.sh`: Script to make the main Python file executable
- `execute.sh`: Script to run the program with command-line arguments

## Usage

### Basic Execution

```bash
./compile.sh
./execute.sh -i <query-file> -o <output-file> -m <model-dir> -d <ntcir-dir>
```

### Required Parameters

- `-i <query-file>`: Path to the input query XML file
- `-o <output-file>`: Path for the output ranked list file
- `-m <model-dir>`: Directory containing model files (vocab.all, file-list, inverted-file)
- `-d <ntcir-dir>`: Directory containing NTCIR documents (path to CIRB010 directory)

### Optional Parameters

- `-r`: Enable relevance feedback
- `-g <ground-truth>`: Ground truth file for evaluation
- `--iterations <int>`: Number of feedback iterations (default: 1)
- `--alpha <float>`: Weight for original query in Rocchio (default: 1.0)
- `--beta <float>`: Weight for relevant documents in Rocchio (default: 0.75)
- `--gamma <float>`: Weight for non-relevant documents in Rocchio (default: 0)
- `--threshold <float>`: Score threshold for document retrieval (default: 550)
- `--k1 <float>`: BM25 k1 parameter (default: 1.2)
- `--b <float>`: BM25 b parameter (default: 0.75)
- `--k3 <float>`: BM25 k3 parameter (default: 8)

### Example Commands

Basic retrieval without relevance feedback:
```bash
./execute.sh -i /path/to/query-train.xml -o output.csv -m model-dir -d /path/to/CIRB010
```

Retrieval with relevance feedback: (Use the default setting will get my best result on the Kaggle leaderboard)
```bash
./execute.sh -r -i /path/to/query-train.xml -o output_feedback.csv -m model-dir -d /path/to/CIRB010
```

Retrieval with custom parameters:
```bash
./execute.sh -r -i /path/to/query-train.xml -o output_custom.csv -m model-dir -d /path/to/CIRB010 --iterations 2 --alpha 0.8 --beta 0.75 --threshold 600 --k1 1.5 --b 0.75 --k3 10
```

### Grid Search

To find optimal parameters using grid search:
```bash
./execute.sh --grid-search -i /path/to/query-train.xml -o grid_search -m model-dir -d /path/to/CIRB010 -g /path/to/ans_train.csv
```

## Output Format

The output file is in CSV format with the following columns:
- `query_id`: Query ID (the last three digits of the query number)
- `retrieved_docs`: Space-separated list of retrieved document IDs

## Performance Evaluation

When a ground truth file is provided with the `-g` option, the system will calculate and output the Mean Average Precision (MAP) score.