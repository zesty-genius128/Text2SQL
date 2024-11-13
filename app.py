import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load the trained model and tokenizer
model_path = "results/final_model"
tokenizer_path = "results/final_tokenizer"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

# Initialize the text-to-SQL pipeline
sql_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def generate_sql(query, table_structure):
    # Clean the table structure to generate input for the model
    table_structure_str = ", ".join(table_structure)
    input_text = f"translate English to SQL: {query} | Table schema: {table_structure_str}"
    
    # Generate SQL using the fine-tuned model
    sql_query = sql_pipeline(input_text, max_length=150)[0]['generated_text']
    
    return sql_query

# Streamlit app interface
st.title("Text-to-SQL Generator")
st.write("Enter a natural language query and table structure:")

# User input for natural language query
query = st.text_input("Enter your query", value="What are the names of employees older than 30?")

# User input for table structure
table_structure = st.text_input("Enter table structure (comma-separated)", value="id, name, age")

if st.button("Generate SQL"):
    # Parse the table structure and generate SQL query
    table_structure_list = [col.strip() for col in table_structure.split(",")]
    generated_sql = generate_sql(query, table_structure_list)
    
    st.write("Generated SQL Query:")
    st.code(generated_sql)
