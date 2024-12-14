import json
import random
import warnings
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Model and Tokenizer Loading
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

# --------------------------------------------------------
# Post-processing Function
# --------------------------------------------------------
# This function attempts to tidy the output SQL query.
def postprocess_sql(sql_query):
    # Replace unusual quotation marks and dashes
    sql_query = sql_query.replace("–", "-").replace("’", "'")
    # Clean up spacing around operators
    sql_query = re.sub(r"\s*([=<>])\s*", r" \1 ", sql_query)
    # Add quotes around values if they appear to be strings without quotes
    # This is a heuristic and may not always be correct
    sql_query = re.sub(r"= ([A-Za-z][\w\s\-]*)", r"= '\1'", sql_query)
    # Remove incomplete WHERE clauses
    sql_query = re.sub(r"WHERE\s+[A-Za-z0-9_]+\s*(=|>|<)\s*$", "", sql_query)
    # Collapse multiple spaces
    sql_query = re.sub(r"\s+", " ", sql_query).strip()
    return sql_query

# --------------------------------------------------------
# Generation Function
# --------------------------------------------------------
def get_sql(question, max_length=64, num_beams=5, repetition_penalty=1.2):
    # Prompt format based on the model's training instructions
    prompt = f"translate English to SQL: {question}"
    inputs = tokenizer([prompt], return_tensors='pt', truncation=True)
    
    output_tokens = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        repetition_penalty=repetition_penalty
    )
    
    predicted_sql = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    predicted_sql = postprocess_sql(predicted_sql)
    return predicted_sql

# --------------------------------------------------------
# Data Loading
# --------------------------------------------------------
df=pd.read_csv("../datasets/data/train_human_readable.csv")

# --------------------------------------------------------
# Sampling and Evaluation
# --------------------------------------------------------
# We sample some entries to test
for index, row in df.iterrows():
    question = row['question']
    predicted_sql = get_sql(question)
    true_sql= row['human_readable_sql']
    #print(dev_data[0])
    
    
    print(f"Question: {question}")
    print(f"Predicted SQL: {predicted_sql}")
    print(f"True SQL: {true_sql}\n")
    
    if index == 9:  # Limit to 10 iterations
        break