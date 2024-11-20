import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# Load dataset
def load_data(filepath):
    """Loads a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame containing the loaded data
    """

    return pd.read_csv(filepath)

# Clean text function
def clean_text(text):
    """
    Cleans the input text by removing unwanted characters and whitespace.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text with leading and trailing whitespace removed.
        If the input is not a string, an empty string is returned.
    """
    if isinstance(text, str):
        return text.strip()  # Basic cleaning; expand as needed
    return ""

# Initialize BERT model and tokenizer
def initialize_model():
    """
    Initializes a BERT model and tokenizer for sequence classification.

    The model used is specified by the `model_name` variable and is loaded
    using the `from_pretrained` method from the `transformers` library. The
    tokenizer is also loaded from the same library.

    If a GPU is available, the model is moved to the GPU to speed up
    computation.

    Args:
        None

    Returns:
        tuple: A tuple containing the tokenizer and model objects
    """

    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'  # Example model; replace with a suitable model for your task
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_name, local_files_only= True)
        # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("using cuda")
    
    return tokenizer, model

# Filter papers using semantic analysis
def filter_papers(data, tokenizer, model):
    """
    Filters papers based on semantic analysis using BERT.

    This function takes in a Pandas DataFrame `data` containing the papers to be filtered,
    a `tokenizer` for tokenizing the text, and a pre-trained BERT `model` for performing
    sequence classification.

    The function iterates over the rows of the input DataFrame, cleans the title and abstract
    of each paper, and combines them into a single string for analysis. The string is then
    passed to the BERT model for prediction, and the paper is considered relevant if the
    predicted label is "5 stars".

    The function returns a new Pandas DataFrame containing only the relevant papers, with
    columns for the title, abstract, methods used (if any), and classification label.

    Parameters:
        data (pd.DataFrame): DataFrame containing the papers to be filtered
        tokenizer (transformers.BertTokenizer): Tokenizer for tokenizing the text
        model (transformers.BertForSequenceClassification): Pre-trained BERT model for sequence classification

    Returns:
        pd.DataFrame: DataFrame containing the relevant papers with title, abstract, methods, and classification label
    """
    device = 0 if torch.cuda.is_available() else -1

    print("using device:", device) 
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)
    
    relevant_papers = []
    print("length of data", len(data))
    for _, row in data.iterrows():
        paper_title = clean_text(row['Title'])  # Clean title
        paper_abstract = clean_text(row['Abstract'])  # Clean abstract
        
        # Combine title and abstract for analysis
        text = f"{paper_title} {paper_abstract}"
        
        # Ensure text is not empty after cleaning
        if text.strip():
            # Get prediction from BERT using the original text input with padding and truncation
            result = classifier(text, padding=True, truncation=True, max_length=512)
            # print('Result', result)
            classification_label = result[0]['label'] if result else None
            if classification_label == "5 stars":
                row['classification'] = classification_label
                
                has_text_mining = 'text mining' in paper_title.lower() or 'text mining' in paper_abstract.lower()
                has_computer_vision = 'computer vision' in paper_title.lower() or 'computer vision' in paper_abstract.lower()
                
                if has_text_mining and has_computer_vision:
                    row['methods'] = "both"
                elif has_text_mining:
                    row['methods'] = "text mining"
                elif has_computer_vision:
                    row['methods'] = "computer vision"
                else:
                    row['methods'] = "none"

                relevant_papers.append({
                    "Title": paper_title,
                    "Abstract": paper_abstract,
                    "methods": row['methods'],
                    "classification": row['classification']
                })
    # print("relevant papers", relevant_papers)
    return pd.DataFrame(relevant_papers)

# Main function to execute the workflow
def main(filepath):
    # Load data
    data = load_data(filepath)
    
    # Drop rows with NaN values in critical columns (title and abstract)
    data.dropna(subset=['Title', 'Abstract'], inplace=True)

    # Initialize model
    tokenizer, model = initialize_model()
    
    # Filter papers
    relevant_papers = filter_papers(data, tokenizer, model)
    
    print(f"Total relevant papers: {len(relevant_papers)}")
    output_filepath = "classification_results.xlsx"
    relevant_papers.to_excel(output_filepath, index=False)
    
    print(f"Results saved to {output_filepath}")

# Run the main function with your CSV file path
if __name__ == "__main__":
    main("collection_with_abstracts.csv")
