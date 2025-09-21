import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

def parse_product_info(content):
    """
    Parses product information from the user content string.
    Handles both English and Japanese formats and escaped newlines.
    """
    # Replace escaped newlines with actual newlines for consistent parsing
    content = content.replace('\\n', '\n')

    prod1, prod2 = None, None
    
    # English pattern
    pattern_en = re.compile(r"Product 1:\s*(.*?)\s*Product 2:\s*(.*?)\s*(?:Do these refer|Answer:)", re.DOTALL)
    match_en = pattern_en.search(content)
    if match_en:
        prod1 = match_en.group(1).strip()
        prod2 = match_en.group(2).strip()
        # Clean up any remaining prompt fragments from the end of product 2
        prod2 = re.sub(r'\s*Do these refer to the same product\?\s*Answer:\s*$', '', prod2, flags=re.DOTALL).strip()
        return prod1, prod2

    # Japanese pattern
    pattern_jp = re.compile(r"商品情報1:\s*(.*?)\s*商品情報2:\s*(.*?)\s*(?:これらは同一の商品ですか|回答:)", re.DOTALL)
    match_jp = pattern_jp.search(content)
    if match_jp:
        prod1 = match_jp.group(1).strip()
        prod2 = match_jp.group(2).strip()
        # Clean up any remaining prompt fragments
        prod2 = re.sub(r'\s*これらは同一の商品ですか？\s*回答:\s*$', '', prod2, flags=re.DOTALL).strip()
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
    parser = argparse.ArgumentParser(description="Find swapped (A,B) -> (B,A) pairs in fine-tuning data.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                user_content = next((msg.get("content") for msg in messages if msg.get("role") == "user"), None)
                assistant_content = next((msg.get("content") for msg in messages if msg.get("role") == "assistant"), None)

                if user_content and assistant_content:
                    prod1, prod2 = parse_product_info(user_content)
                    label = get_label(assistant_content)
                    
                    if prod1 and prod2 and label != "Unknown":
                        records.append({
                            "line": i + 1,
                            "prod1": prod1,
                            "prod2": prod2,
                            "label": label
                        })
            except Exception as e:
                print(f"Warning: Error processing line {i+1}. Error: {e}")

    # Use a dictionary to find swapped pairs
    # Key: frozenset of (prod1, prod2) to handle order invariance
    # Value: list of records matching that pair
    pair_map = defaultdict(list)
    for record in records:
        # The key is a frozenset of the two product strings, making it order-independent
        pair_key = frozenset([record["prod1"], record["prod2"]])
        pair_map[pair_key].append(record)

    print(f"Finding swapped pairs in {input_path.name}...")
    found_count = 0
    for key, found_records in pair_map.items():
        # If more than one record shares the same frozenset key, it's a duplicate or swapped pair
        if len(found_records) > 1:
            found_count += 1
            print("-" * 50)
            print(f"Found Swapped/Duplicate Pair Set #{found_count}:")
            
            is_consistent = len(set(r['label'] for r in found_records)) == 1
            print(f"Label Consistency: {'CONSISTENT' if is_consistent else '!!! INCONSISTENT !!!'}")

            for record in found_records:
                print(f"  - Line {record['line']}: Label = {record['label']}")
                # print(f"    Product 1: {record['prod1'][:80]}...") # Optional: print snippets
                # print(f"    Product 2: {record['prod2'][:80]}...") # Optional: print snippets

    if found_count == 0:
        print("\nNo swapped or duplicate pairs were found.")
    else:
        print("-" * 50)
        print(f"\nTotal sets of swapped/duplicate pairs found: {found_count}")


if __name__ == "__main__":
    main()
