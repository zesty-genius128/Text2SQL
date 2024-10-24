import json
import re

def load_data(fpath):
    """
    Load the WikiSQL data from a jsonl file.
    """
    data = []
    with open(fpath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_sql(schema, query):
    """
    Generate SQL based on the schema and natural language query.
    """
    # Extract column names from schema
    columns = schema['header']
    
    # Use regex to find if any of the columns are mentioned in the query
    selected_column = None
    for col in columns:
        if re.search(col, query, re.IGNORECASE):
            selected_column = col
            break
    
    # If no specific column is mentioned, default to '*'
    if not selected_column:
        selected_column = '*'
    
    # Try to extract conditions from the query (e.g., for "WHERE" clauses)
    # For simplicity, let's just look for "for <value>" pattern
    condition_match = re.search(r'for (.+)', query, re.IGNORECASE)
    condition_value = condition_match.group(1).strip() if condition_match else None
    
    # Build the SQL query
    if condition_value:
        sql_query = f"SELECT {selected_column} FROM {schema['header'][0]} WHERE {columns[0]} = '{condition_value}';"
    else:
        sql_query = f"SELECT {selected_column} FROM {schema['header'][0]};"
    
    return sql_query
def main():
    # Load the schema from the tables.jsonl file
    train_schema = load_data('../datasets/data/train.tables.jsonl')

    # Load the train data from train.jsonl
    train_data = load_data('../datasets/data/train.jsonl')

    print(f"Loaded {len(train_data)} training examples.")

    # Process the first query
    first_entry = train_data[3]
    question = first_entry['question']
    table_id = first_entry['table_id']

    # Find the schema corresponding to table_id
    schema = next(item for item in train_schema if item['id'] == table_id)

    # Generate SQL (placeholder logic for now)
    sql_query = get_sql(schema, question)
    
    print(f"Question: {question}")
    print(f"Generated SQL: {sql_query}")

if __name__ == "__main__":
    main()
