import json
import os
import pandas as pd
from tqdm import tqdm

# Paths to Local WikiSQL Files
DATASET_DIR = "../datasets/data"  # Adjust this path
TRAIN_FILE = os.path.join(DATASET_DIR, "train.jsonl")
TRAIN_TABLES_FILE = os.path.join(DATASET_DIR, "train.tables.jsonl")
OUTPUT_FILE = os.path.join(DATASET_DIR, "train_human_readable.json")
OUTPUT_CSV = os.path.join(DATASET_DIR, "train_human_readable.csv")

def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
                continue
    return data

def process_wikisql_to_readable(train_data, tables_data):
    """
    Process WikiSQL data into a human-readable format with questions and SQL queries.
    """
    table_id_to_headers = {}

    # Create a mapping of table_id to table headers
    for table in tables_data:
        table_id = table["id"]
        headers = table["header"]
        table_id_to_headers[table_id] = headers

    # Process each question into human-readable SQL
    human_readable_data = []
    for entry in tqdm(train_data, desc="Processing"):
        table_id = entry["table_id"]
        question = entry["question"]
        sql = entry["sql"]

        # Handle cases where the table_id is missing
        if table_id not in table_id_to_headers:
            continue

        headers = table_id_to_headers[table_id]

        # Map column indices to column names
        readable_conditions = []
        for cond in sql["conds"]:
            column_idx, operator, value = cond
            if column_idx < len(headers):
                column_name = headers[column_idx]
                readable_conditions.append(f"{column_name} {['=', '>', '<'][operator]} '{value}'")

        # Build the SQL query
        selected_column = headers[sql["sel"]] if sql["sel"] < len(headers) else "UNKNOWN"
        readable_sql = f"SELECT {selected_column} FROM table WHERE " + " AND ".join(readable_conditions)

        human_readable_data.append({
            "question": question,
            "human_readable_sql": readable_sql
        })

    return human_readable_data

def save_results_to_file(data, json_output, csv_output):
    """
    Save the processed data to JSON and CSV for human readability.
    """
    # Save as JSON
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON output to {json_output}")

    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_output, index=False)
    print(f"Saved CSV output to {csv_output}")

def main():
    # Load the dataset files
    print("Loading WikiSQL files...")
    train_data = load_jsonl(TRAIN_FILE)
    tables_data = load_jsonl(TRAIN_TABLES_FILE)

    print("Processing WikiSQL data into human-readable form...")
    human_readable_data = process_wikisql_to_readable(train_data, tables_data)

    print("Saving results...")
    save_results_to_file(human_readable_data, OUTPUT_FILE, OUTPUT_CSV)

if __name__ == "__main__":
    main()
