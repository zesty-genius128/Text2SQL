import json
import spacy
import logging
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Ensure the 'logs' directory exists
if not os.path.exists('../logs'):
    os.makedirs('../logs')

# Ensure the 'results' directory exists to store the model and tokenizer
if not os.path.exists('../results'):
    os.makedirs('../results')

# Initialize logging in the correct folder
logging.basicConfig(filename='../logs/sql_generator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the text-to-SQL model
model_name = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
nlp_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load spaCy for NLP tasks
nlp = spacy.load("en_core_web_sm")

# File to store the results
result_file = "../results/generated_sql_results.txt"

def get_schema_from_json(fpath):
    schema = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            table_id = str(entry['id'].lower())
            columns = [str(col.lower()) for col in entry['header']]
            schema[table_id] = columns
    return schema

def load_data(fpath):
    data = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    logging.info(f"Loaded {len(data)} examples from {fpath}")
    return data

def generate_sql(question, schema_columns):
    # Convert schema columns list to a comma-separated string
    schema_str = ", ".join(schema_columns)
    
    # Use spaCy to filter stopwords and structure the question to enhance model understanding
    doc = nlp(question)
    filtered_question = " ".join([token.text for token in doc if not token.is_stop])
    
    # Include a hint for relational criteria in the input prompt
    input_str = f"translate English to SQL: {filtered_question} | Table schema: {schema_str} | match criteria: reference and comparison"

    # Generate SQL using the text-to-SQL model
    sql_query = nlp_model(input_str, max_length=150)[0]['generated_text']
    
    return sql_query


def log_results(question, generated_sql, result_file):
    # Log results to a file with UTF-8 encoding
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(f"Time: {str(datetime.now())}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Generated SQL: {generated_sql}\n\n")

def save_model():
    # Save the trained model and tokenizer
    model.save_pretrained("../results/final_model")
    tokenizer.save_pretrained("../results/final_tokenizer")
    logging.info("Model and tokenizer saved.")

def main():
    # Load schema from the jsonl file
    schema = get_schema_from_json('../datasets/data/train.tables.jsonl')
    logging.info("Schema loaded successfully")

    # Load data from the train set
    data = load_data('../datasets/data/train.jsonl')

    # Process examples
    for i, example in enumerate(tqdm(data, desc="Processing")):
        question = example['question']
        table_id = example['table_id']
        
        logging.info(f"Processing question: {question}")
        
        # Generate SQL query
        generated_sql = generate_sql(question, schema[table_id])
        
        print(f"Question: {question}")
        print(f"Generated SQL: {generated_sql}")
        
        # Log results
        log_results(question, generated_sql, result_file)

        # Optional: Break the loop for demonstration purposes
        if i >= 500:  # Limit to 12 queries for testing
             break

    # Save the model after processing
    save_model()

if __name__ == "__main__":
    main()
