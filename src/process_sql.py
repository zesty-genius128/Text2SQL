import json
import spacy
import logging
import os
import re
import sqlparse
import sqlite3
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from tqdm import tqdm
from datasets import Dataset
import evaluate

# Ensure necessary directories exist
os.makedirs('../logs', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# Logging setup
logging.basicConfig(filename='../logs/sql_generator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy for NLP preprocessing
nlp = spacy.load("en_core_web_sm")

# Model and Tokenizer setup
MODEL_NAME = "mrm8488/t5-base-finetuned-wikiSQL"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
nlp_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# File paths
RESULT_FILE = "../results/generated_sql_results.txt"
MODEL_SAVE_DIR = "../results/final_model"
TOKENIZER_SAVE_DIR = "../results/final_tokenizer"

# Metric initialization
accuracy_metric = evaluate.load("accuracy")

# Preprocessing Functions
def preprocess_question(question):
    question = re.sub(r"[^a-zA-Z0-9\s]", "", question).lower()
    doc = nlp(question)
    return " ".join([token.text for token in doc if not token.is_stop])

def preprocess_schema(schema):
    return [col.lower().strip() for col in schema]

def clean_generated_sql(sql_query):
    sql_query = re.sub(r"Match criteria:.*", "", sql_query).strip()
    sql_query = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
    return sql_query

# Schema and Data Loading
def get_schema_from_json(fpath):
    schema = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            schema[entry['id'].lower()] = preprocess_schema(entry['header'])
    return schema

def load_data(fpath):
    data = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Save Model
def save_model():
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
    logging.info("Model and tokenizer saved.")

# SQL Generation and Evaluation
def run_and_save_results(data, schema):
    predictions = []
    references = []

    with open(RESULT_FILE, "w", encoding="utf-8") as result_f:
        for example in tqdm(data, desc="Generating SQL"):
            question = example.get('question', '')
            table_id = example.get('table_id', '')
            ground_truth_sql = example.get('query', None)

            if not ground_truth_sql:
                logging.warning(f"Missing 'query' key for example: {example}")
                continue

            if table_id in schema:
                schema_columns = schema[table_id]
                preprocessed_question = preprocess_question(question)
                schema_str = ", ".join(schema_columns)
                input_str = f"translate English to SQL: {preprocessed_question} | Table schema: {schema_str}"

                try:
                    generated_sql = nlp_model(input_str, max_length=150)[0]['generated_text']
                    generated_sql = clean_generated_sql(generated_sql)
                except Exception as e:
                    logging.error(f"Failed to generate SQL for {question}: {e}")
                    generated_sql = ""

                result_f.write(f"Question: {question}\n")
                result_f.write(f"Generated SQL: {generated_sql}\n")
                result_f.write(f"Ground Truth SQL: {ground_truth_sql}\n\n")

                predictions.append(generated_sql)
                references.append(ground_truth_sql)

    return predictions, references

# Main Function
def main():
    schema_path = '../datasets/data/train.tables.jsonl'
    data_path = '../datasets/data/train.jsonl'

    schema = get_schema_from_json(schema_path)
    data = load_data(data_path)

    # Run and save results
    predictions, references = run_and_save_results(data, schema)

    # Save model for Streamlit
    save_model()

    # Evaluate Accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    logging.info(f"Accuracy: {accuracy['accuracy']:.4f}")
    print(f"Accuracy: {accuracy['accuracy']:.4f}")

if __name__ == "__main__":
    main()
