# Text-to-SQL Generator Project

This project is a **Text-to-SQL** pipeline that converts natural language queries into SQL queries using a fine-tuned Transformer model. It includes preprocessing, query generation, validation, and an interactive frontend for testing the results.

---

## Project Overview

This repository includes the following components:

1. **Data Processing (`process_sql.py`)**: Prepares and processes input data, generates SQL queries, and logs results.
2. **Interactive App (`app.py`)**: A Streamlit-based interface to test natural language queries and table schemas.
3. **Validation (`validation.py`)**: Ensures the dataset files are in valid JSON Lines format.
4. **Dependencies**: Provided in `requirements.txt` for easy setup.

---

## Folder Structure

```plaintext
NLP_TEXT2SQL/
├── datasets/
│   └── data/
│       ├── dev.db
│       ├── dev.jsonl
│       ├── dev.tables.jsonl
│       ├── test.db
│       ├── test.jsonl
│       ├── test.tables.jsonl
│       ├── train.db
│       ├── train.jsonl
│       └── train.tables.jsonl
├── logs/
│   ├── sql_generator.log
│   └── training.log
├── results/
│   ├── final_model/           # Fine-tuned model directory
│   ├── final_tokenizer/       # Tokenizer directory
│   └── generated_sql_results.txt
├── src/
│   ├── process_sql.py         # Main script for SQL generation
│   └── validation.py          # Script for validating JSONL files
├── venv/                      # Virtual environment (optional)
│   └── ...
├── app.py                     # Streamlit app for interactive testing
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
└── .gitignore                 # Git ignore file
```
---

## Setup Instructions

### 1. Clone the Repository
```
git clone <https://github.com/zesty-genius128/NLP_Text2SQL.git>

cd NLP_TEXT2SQL
```
### 2. Set Up the Environment
Create a virtual environment and install dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Download Required NLP Model
Download the small English model for spaCy:

```
python -m spacy download en_core_web_sm
```
---
## Running the Project
### 1. Process SQL Generation
To process the input questions and generate SQL queries, run:

```
python src/process_sql.py
```
This script:

Loads data from train.jsonl and train.tables.jsonl.
Generates SQL queries using the model.
Logs output to results/generated_sql_results.txt.
### 2. Interactive App
Run the Streamlit app for testing:
```
streamlit run app.py
Access the app in your browser: http://localhost:8501.
Input a natural language query and a table schema.
Click "Generate SQL" to see the result.
```
### 3. Validate JSONL Files
To validate dataset files:
```
python src/validation.py
```
This script checks for invalid JSON lines and reports errors.

---
## Dependencies
Dependencies are provided in requirements.txt:

```
transformers==4.46.0
streamlit==1.39.0
spacy==3.8.2
torch==2.5.0
pandas==2.2.3
tqdm==4.66.5
Flask==3.0.3
```
**Install with:**

```
pip install -r requirements.txt
```
### Logs and Outputs
Logs are stored in the logs/ directory.
Generated SQL queries are saved in results/generated_sql_results.txt.
Fine-tuned models are saved in results/final_model and results/final_tokenizer.

- ***Example Usage***
- Input
```
Question: What are the names of employees older than 30?
Table Schema: id, name, age
```
- Output
```
SELECT name FROM table WHERE age > 30;
```
<!--License
This project is licensed under the MIT License.

Contact
For queries or issues, contact:
Name: 
Email: 2001.arjunmalik@gmail.com
GitHub: arjunmalik11
-->