import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import warnings
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Model and Tokenizer Loading
# --------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

# --------------------------------------------------------
# Enhanced Post-processing Function for SQL Normalization
# --------------------------------------------------------
def normalize_sql(sql_query):
    # Lowercase the query
    sql_query = sql_query.lower()
    # Standardize spaces
    sql_query = re.sub(r"\s+", " ", sql_query)
    # Ensure consistent quotes around values
    sql_query = re.sub(r"= (\d+)", r"= '\1'", sql_query)  # Add quotes to numbers
    sql_query = re.sub(r"= (\w+)", r"= '\1'", sql_query)  # Add quotes to strings
    sql_query = sql_query.replace('"', "'")  # Standardize quotes
    # Remove any trailing or leading spaces
    return sql_query.strip()

def postprocess_sql(sql_query):
    sql_query = sql_query.replace("–", "-").replace("’", "'")
    sql_query = re.sub(r"\s*([=<>])\s*", r" \1 ", sql_query)
    sql_query = re.sub(r"= ([A-Za-z][\w\s\-]*)", r"= '\1'", sql_query)
    sql_query = re.sub(r"WHERE\s+[A-Za-z0-9_]+\s*(=|>|<)\s*$", "", sql_query)
    sql_query = re.sub(r"\s+", " ", sql_query).strip()
    return sql_query

# --------------------------------------------------------
# Generation Function
# --------------------------------------------------------
def get_sql(question, max_length=64, num_beams=5, repetition_penalty=1.2):
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
    return postprocess_sql(predicted_sql)

# --------------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------------
def calculate_bleu(predicted_sql, true_sql):
    predicted_tokens = normalize_sql(predicted_sql).split()
    true_tokens = normalize_sql(true_sql).split()
    return sentence_bleu([true_tokens], predicted_tokens)

def calculate_exact_match(predicted_sql, true_sql):
    norm_pred = normalize_sql(predicted_sql)
    norm_true = normalize_sql(true_sql)
    return 1 if norm_pred == norm_true else 0

# --------------------------------------------------------
# Data Loading
# --------------------------------------------------------
df = pd.read_csv("../datasets/data/train_human_readable.csv")  # Path to your dataset

# --------------------------------------------------------
# Sampling and Evaluation
# --------------------------------------------------------
bleu_scores = []
exact_matches = []

print("=== Evaluation Results Per Example ===\n")

for index, row in df.iterrows():
    question = row['question']
    true_sql = row['human_readable_sql']
    predicted_sql = get_sql(question)

    bleu = calculate_bleu(predicted_sql, true_sql)
    em = calculate_exact_match(predicted_sql, true_sql)

    bleu_scores.append(bleu)
    exact_matches.append(em)

    print(f"Question: {question}")
    print(f"Predicted SQL: {predicted_sql}")
    print(f"True SQL: {true_sql}")
    print(f"BLEU Score: {bleu:.4f}, Exact Match: {em}\n")

    if index == 9:  # Limit to 10 iterations for demonstration
        break

# --------------------------------------------------------
# Aggregated Results
# --------------------------------------------------------
average_bleu = sum(bleu_scores) / len(bleu_scores)
exact_match_accuracy = sum(exact_matches) / len(exact_matches)

print("=== Final Aggregated Evaluation Metrics ===")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
