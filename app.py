import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import re

# Load the model and tokenizer from the previous code
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

sql_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def basic_normalize(text):
    text = text.lower().strip()
    text = " ".join(text.split())
    return text

def generate_sql(query, table_structure):
    query = basic_normalize(query)
    columns = [c.strip().lower() for c in table_structure.split(",")]
    col_str = ", ".join(columns)
    input_text = f"translate English to SQL: {query} | Table schema: {col_str}"
    output = sql_pipeline(input_text, max_length=128)[0]['generated_text']
    return output

st.title("Text-to-SQL Generator")
st.write("Enter a natural language question and your table columns:")

user_query = st.text_input("Question:", "What are the names of employees older than 30?")
user_schema = st.text_input("Table Schema (comma-separated):", "id, name, age")

if st.button("Generate SQL"):
    sql = generate_sql(user_query, user_schema)
    st.write("Generated SQL:")
    st.code(sql)