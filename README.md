# Pubmed_classifier
# Research Paper Classification

This project is designed to classify research papers based on their titles and abstracts using a pre-trained BERT model. The classification focuses on identifying specific methods mentioned in the papers, such as "text mining" and "computer vision." The output includes relevant details about the papers, including their classification status.

## Features

- Classifies research papers based on their titles and abstracts.
- Identifies methods such as "text mining" and "computer vision."
- Outputs a DataFrame containing the title, abstract, methods used, and classification status.

## Requirements

To run this project, you will need:

- Python 3.6 or higher
- The following Python packages:
  - `pandas`
  - `transformers`
  - `torch`

You can install the required packages using pip:

```bash
pip install pandas transformers torch
