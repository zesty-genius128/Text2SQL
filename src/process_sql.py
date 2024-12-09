import json
import spacy
import logging
import os
import re
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Ensure the 'logs' directory exists
if not os.path.exists('../logs'):
    os.makedirs('../logs')

# Ensure the 'results' directory exists to store the model and tokenizer
if not os.path.exists('../results'):
    os.makedirs('../results')

# Initialize logging
logging.basicConfig(filename='../logs/sql_generator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the model and tokenizer
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
nlp_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load spaCy for NLP tasks
nlp = spacy.load("en_core_web_sm")

# File to store the results
result_file = "../results/generated_sql_results.txt"

def preprocess_question(question):
    """
    Preprocess the input question by removing special characters,
    extra spaces, and stopwords using spaCy.
    """
    # Remove unnecessary punctuation and extra spaces
    question = re.sub(r"[^a-zA-Z0-9\s?']", "", question)
    question = " ".join(question.split())

    # Lowercase the question
    question = question.lower()
    
    # Use spaCy to remove stopwords
    doc = nlp(question)
    filtered_question = " ".join([token.text for token in doc if not token.is_stop])

    logging.info(f"Preprocessed question: {filtered_question}")
    return filtered_question

def preprocess_schema(schema):
    """
    Preprocess schema column names by lowercasing and removing unnecessary spaces.
    """
    return [col.lower().strip() for col in schema]

def get_schema_from_json(fpath):
    """
    Extract table schemas from a JSONL file.
    """
    schema = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            table_id = str(entry['id'].lower())
            columns = preprocess_schema(entry['header'])
            schema[table_id] = columns
    return schema

def load_data(fpath):
    """
    Load examples from the dataset file.
    """
    data = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    logging.info(f"Loaded {len(data)} examples from {fpath}")
    return data

def generate_sql(question, schema_columns):
    """
    Generate SQL query from an input question and table schema.
    """
    # Preprocess question and schema
    preprocessed_question = preprocess_question(question)
    schema_str = ", ".join(schema_columns)

    # Build input string for model
    input_str = (f"translate English to SQL: {preprocessed_question} | "
                 f"Table schema: {schema_str} | match criteria: reference and comparison")

    # Generate SQL using the model
    logging.info(f"Model input: {input_str}")
    sql_query = nlp_model(input_str, max_length=150)[0]['generated_text']
    return sql_query

def log_results(question, generated_sql, result_file):
    """
    Log the generated SQL results into a text file.
    """
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(f"Time: {str(datetime.now())}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Generated SQL: {generated_sql}\n\n")

def save_model():
    """
    Save the trained model and tokenizer.
    """
    model.save_pretrained("../results/final_model")
    tokenizer.save_pretrained("../results/final_tokenizer")
    logging.info("Model and tokenizer saved.")

def main():
    # Load schema and data
    schema = get_schema_from_json('../datasets/data/train.tables.jsonl')
    data = load_data('../datasets/data/train.jsonl')

    # Process data
    for i, example in enumerate(tqdm(data, desc="Processing")):
        question = example['question']
        table_id = example['table_id']
        
        logging.info(f"Processing question: {question}")
        
        if table_id in schema:
            # Generate SQL query
            generated_sql = generate_sql(question, schema[table_id])
            print(f"Question: {question}")
            print(f"Generated SQL: {generated_sql}")
            
            # Log results
            log_results(question, generated_sql, result_file)
        else:
            logging.warning(f"Schema not found for table_id: {table_id}")
        
        # Optional: limit for testing
        if i >= 500:  
            break

    # Save model
    save_model()

if __name__ == "__main__":
    main()
