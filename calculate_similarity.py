import json
import argparse
import re
from pathlib import Path
import pandas as pd
from thefuzz import fuzz

def parse_product_info(content):
    """
    Parses product information from the user content string for both English and Japanese formats.
    Handles escaped newlines (\\n) as well.
    """
    prod1, prod2 = None, None
    # Regex to capture content between markers, robust to different newline styles
    pattern_en = re.compile(r"Product 1:\s*(.*?)\s*Product 2:\s*(.*?)\s*Do these refer", re.DOTALL)
    match_en = pattern_en.search(content.replace('\\n', '\n'))

    if match_en:
        prod1 = match_en.group(1).strip()
        prod2 = match_en.group(2).strip()
        return prod1, prod2

    pattern_jp = re.compile(r"商品情報1:\s*(.*?)\s*商品情報2:\s*(.*?)\s*これらは同一の商品ですか", re.DOTALL)
    match_jp = pattern_jp.search(content.replace('\\n', '\n'))
    
    if match_jp:
        prod1 = match_jp.group(1).strip()
        prod2 = match_jp.group(2).strip()
        return prod1, prod2
        
    return None, None

def get_label(assistant_content):
    """
    Extracts the label from the assistant's response.
    """
    first_line = assistant_content.strip().split('\\n')[0].split('\n')[0]
    if first_line in ["Yes", "はい"]:
        return "Positive"
    elif first_line in ["No", "いいえ"]:
        return "Negative"
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Calculate string similarity for product pairs in fine-tuning data.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                user_content = None
                assistant_content = None

                for msg in messages:
                    if msg.get("role") == "user":
                        user_content = msg.get("content")
                    elif msg.get("role") == "assistant":
                        assistant_content = msg.get("content")

                if user_content and assistant_content:
                    prod1_text, prod2_text = parse_product_info(user_content)
                    label = get_label(assistant_content)
                    
                    if prod1_text and prod2_text:
                        # Using token_set_ratio which is good for different word order and partial matches
                        similarity = fuzz.token_set_ratio(prod1_text, prod2_text)
                        
                        results.append({
                            "product_1": prod1_text.replace("\n", " ").replace("\\n", " "),
                            "product_2": prod2_text.replace("\n", " ").replace("\\n", " "),
                            "label": label,
                            "similarity_score": similarity
                        })

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
            except Exception as e:
                print(f"An error occurred processing line: {line.strip()}. Error: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully processed {len(df)} records.")
        print(f"Results saved to {output_path}")

        # Print average similarity for each label
        avg_similarity = df.groupby('label')['similarity_score'].mean()
        print("\nAverage similarity scores:")
        print(avg_similarity)
    else:
        print("No valid data was processed.")


if __name__ == "__main__":
    main()
