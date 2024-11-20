# Pubmed_classifier
# Research Paper Classification

This project provides a solution for classifying research papers based on their titles and abstracts. It utilizes a pre-trained BERT model to analyze the text and determine if the papers discuss specific methodologies such as "text mining" or "computer vision". The results are filtered to include only relevant papers with a classification label of "5 star".

## Components

1. **Data Loading**: 
   - The code reads a CSV file containing research paper titles and abstracts.
   - It requires that the CSV file has at least two columns named `title` and `abstract`.

2. **Text Cleaning**: 
   - The text from titles and abstracts is cleaned to remove leading and trailing whitespace.

3. **BERT Model Initialization**: 
   - A pre-trained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) is loaded for sentiment analysis.
   - The model can utilize a GPU if available for faster processing.

4. **Classification Process**:
   - Each paper's title and abstract are combined for analysis.
   - The model predicts whether the paper is classified as "6 star".
   - The code checks for the presence of keywords ("text mining" and "computer vision") in the title or abstract:
     - If both keywords are present, the methods column is set to "both".
     - If only one keyword is present, it sets the methods column accordingly.
     - If neither keyword is present, it sets the methods column to "none".

5. **Output**:
   - The final output includes only four columns: `Title`, `Abstract`, `methods`, and `classification`.
   - The output is printed to the console.

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - transformers
  - torch (with CUDA support if using GPU)

You can install the required libraries using pip:

```bash
pip install pandas transformers torch
```
## Usage

1. Prepare your CSV file with at least two columns: title and abstract.
2. Place your CSV file in the same directory as the script or provide an appropriate path.
3. Modify the line in the script where it calls main() to point to your CSV file:

```bash
main("path_to_your_dataset.csv")
```
4. Run the script:
```bash
python your_script_name.py
```
5.The output will display relevant papers with their titles, abstracts, methods, and classifications.

## Acknowledgments

- This project uses Hugging Face's Transformers library for implementing BERT.
- Special thanks to the open-source community for providing valuable resources and models.
For any questions or issues, please feel free to open an issue on this repository.

### Explanation of Sections

- **Project Title**: A brief title for your project.
- **Components**: Describes each part of the solution, including data loading, text cleaning, model initialization, classification process, and output.
- **Requirements**: Lists necessary Python libraries and instructions on how to install them.
- **Usage**: Provides step-by-step instructions on how users can prepare their data and run the code.
- **Acknowledgments**: Credits any external libraries or resources used in the project.
