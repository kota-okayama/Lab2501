import json
import argparse
import re
from pathlib import Path
from collections import Counter

def parse_and_extract_brands(content):
    """
    Parses product information and extracts brand names from the text.
    """
    content = content.replace('\\n', '\n')
    brands = []

    # Simplified patterns to find all brand fields
    # English pattern
    pattern_en = re.compile(r"Product 1:\s*(.*?)\s*Product 2:\s*(.*?)\s*(?:Do these refer|Answer:)", re.DOTALL)
    match_en = pattern_en.search(content)
    if match_en:
        prod1_text = match_en.group(1)
        prod2_text = match_en.group(2)
        
        brand1_match = re.search(r"Brand:\s*(.*)", prod1_text)
        if brand1_match:
            brands.append(brand1_match.group(1).strip())
            
        brand2_match = re.search(r"Brand:\s*(.*)", prod2_text)
        if brand2_match:
            brands.append(brand2_match.group(1).strip())
        return brands

    # Japanese pattern
    pattern_jp = re.compile(r"商品情報1:\s*(.*?)\s*商品情報2:\s*(.*?)\s*(?:これらは同一の商品ですか|回答:)", re.DOTALL)
    match_jp = pattern_jp.search(content)
    if match_jp:
        prod1_text = match_jp.group(1)
        prod2_text = match_jp.group(2)

        brand1_match = re.search(r"ブランド:\s*(.*)", prod1_text)
        if brand1_match:
            brands.append(brand1_match.group(1).strip())

        brand2_match = re.search(r"ブランド:\s*(.*)", prod2_text)
        if brand2_match:
            brands.append(brand2_match.group(1).strip())
        return brands
        
    return brands

def main():
    parser = argparse.ArgumentParser(description="Analyze and display the brand distribution in fine-tuning data.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    brand_counter = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                user_content = next((msg.get("content") for msg in messages if msg.get("role") == "user"), None)
                
                if user_content:
                    brands = parse_and_extract_brands(user_content)
                    # Filter out empty or placeholder brand names
                    valid_brands = [b for b in brands if b and len(b) > 1]
                    brand_counter.update(valid_brands)

            except Exception as e:
                print(f"Warning: Error processing line {i+1}. Error: {e}")

    print(f"\n--- Top 10 Brand Distribution for: {input_path.name} ---")
    if not brand_counter:
        print("No brand information could be extracted.")
    else:
        for brand, count in brand_counter.most_common(10):
            print(f"{brand:<25} | Count: {count}")
    print("-" * 50)


if __name__ == "__main__":
    main()
