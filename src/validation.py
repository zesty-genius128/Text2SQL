import json
import sys

def validate_jsonl_file(file_path):
    """
    Validate each line in a JSON Lines (jsonl) file.
    Prints out invalid lines and their line numbers if any are found.
    """
    invalid_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            # Skip empty lines if they are not supposed to be valid JSON
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                invalid_lines.append((i, line, str(e)))

    if invalid_lines:
        print("Invalid JSON found in the following lines:")
        for (line_num, text, error) in invalid_lines:
            print(f"Line {line_num}: {text}\nError: {error}\n")
    else:
        print("All lines are valid JSON.")

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python validate_jsonl.py ../datasets/data/train_human_readable.jso")
    #     sys.exit(1)

    file_path = "../datasets/data/train_human_readable.json"
    validate_jsonl_file(file_path)
