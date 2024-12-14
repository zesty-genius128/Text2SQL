import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
import re
import warnings
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
# Enhanced Post-processing and Normalization
# --------------------------------------------------------
def normalize_sql(sql_query):
    sql_query = sql_query.lower()
    sql_query = re.sub(r"\s+", " ", sql_query)  # Standardize spaces
    sql_query = re.sub(r"= (\d+)", r"= '\1'", sql_query)  # Add quotes to numbers
    sql_query = sql_query.replace('"', "'")  # Standardize quotes
    return sql_query.strip()

def postprocess_sql(sql_query):
    sql_query = sql_query.replace("–", "-").replace("’", "'")
    sql_query = re.sub(r"\s*([=<>])\s*", r" \1 ", sql_query)
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
    return 1 if normalize_sql(predicted_sql) == normalize_sql(true_sql) else 0

# --------------------------------------------------------
# Load Public Dataset (WikiSQL)
# --------------------------------------------------------
print("Loading WikiSQL dataset...")
dataset = load_dataset("wikisql", split="validation")  # Use validation split for evaluation

# --------------------------------------------------------
# Sampling and Evaluation
# --------------------------------------------------------
bleu_scores = []
exact_matches = []

print("=== Evaluation Results Per Example ===\n")
for i, example in enumerate(dataset):
    question = example['question']
    true_sql = example['sql']['human_readable']  # Ground truth SQL
    predicted_sql = get_sql(question)

    bleu = calculate_bleu(predicted_sql, true_sql)
    em = calculate_exact_match(predicted_sql, true_sql)

    bleu_scores.append(bleu)
    exact_matches.append(em)

    print(f"Question: {question}")
    print(f"Predicted SQL: {predicted_sql}")
    print(f"True SQL: {true_sql}")
    print(f"BLEU Score: {bleu:.4f}, Exact Match: {em}\n")

    if i == 9:  # Limit to 10 examples for demonstration
        break

# --------------------------------------------------------
# Aggregated Results
# --------------------------------------------------------
average_bleu = sum(bleu_scores) / len(bleu_scores)
exact_match_accuracy = sum(exact_matches) / len(exact_matches)

print("=== Final Aggregated Evaluation Metrics ===")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
