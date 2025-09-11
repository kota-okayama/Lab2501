import json
import argparse
import re

def get_prompts(data_type):
    """
    Returns the English system prompt for a given data type.
    (Copied from siamese_model_pytorch/prepare_finetuning_data.py)
    """
    prompt_map = {
        "bib": (
            "You are an expert at determining whether two bibliographic records refer to essentially the same publication.\\n"
            "First, please clearly answer 'Yes' if you believe the two bibliographic records refer to the same publication, or 'No' otherwise.\\n"
            "Next, provide a confidence score from 0.0 (completely different) to 1.0 (completely identical) indicating your certainty in this judgment.\n"
            "Your judgment must strictly follow these rules:\n"
            " - If the confidence score is 0.5 or higher, your answer must be 'Yes'.\n"
            " - If the confidence score is below 0.5, your answer must be 'No'.\n"
        ),
        "music": (
            "You are an expert at determining whether two music records refer to essentially the same musical work.\\n"
            "First, please clearly answer 'Yes' if you believe the two music records refer to the same work, or 'No' otherwise.\\n"
            "Next, provide a confidence score from 0.0 (completely different) to 1.0 (completely identical) indicating your certainty in this judgment.\n"
            "Your judgment must strictly follow these rules:\n"
            " - If the confidence score is 0.5 or higher, your answer must be 'Yes'.\n"
            " - If the confidence score is below 0.5, your answer must be 'No'.\n"
        ),
        "person": (
            "You are an expert at determining whether two person records refer to essentially the same individual.\\n"
            "First, please clearly answer 'Yes' if you believe the two person records refer to the same individual, or 'No' otherwise.\\n"
            "Next, provide a confidence score from 0.0 (completely different) to 1.0 (completely identical) indicating your certainty in this judgment.\n"
            "Your judgment must strictly follow these rules:\n"
            " - If the confidence score is 0.5 or higher, your answer must be 'Yes'.\n"
            " - If the confidence score is below 0.5, your answer must be 'No'.\n"
        ),
        "walmart_amazon_product": (
            "You are an expert at determining whether two product records refer to essentially the same product.\\n"
            "First, please clearly answer 'Yes' if you believe the two product records refer to the same product, or 'No' otherwise.\\n"
            "Next, provide a confidence score from 0.0 (completely different) to 1.0 (completely identical) indicating your certainty in this judgment.\n"
            "Your judgment must strictly follow these rules:\n"
            " - If the confidence score is 0.5 or higher, your answer must be 'Yes'.\n"
            " - If the confidence score is below 0.5, your answer must be 'No'.\n"
        ),
        "wdc_product": (
            "You are an expert at determining whether two product records refer to essentially the same product.\\n"
            "First, please clearly answer 'Yes' if you believe the two product records refer to the same product, or 'No' otherwise.\\n"
            "Next, provide a confidence score from 0.0 (completely different) to 1.0 (completely identical) indicating your certainty in this judgment.\n"
            "Your judgment must strictly follow these rules:\n"
            " - If the confidence score is 0.5 or higher, your answer must be 'Yes'.\n"
            " - If the confidence score is below 0.5, your answer must be 'No'.\n"
        ),
        "unknown": (
            "You are an expert at determining whether two records refer to essentially the same entity.\\n"
            "First, please clearly answer 'Yes' if you believe the two records refer to the same entity, or 'No' otherwise.\\n"
            "Next, provide a confidence score from 0.0 (completely different) to 1.0 (completely identical) indicating your certainty in this judgment.\n"
            "Your judgment must strictly follow these rules:\n"
            " - If the confidence score is 0.5 or higher, your answer must be 'Yes'.\n"
            " - If the confidence score is below 0.5, your answer must be 'No'.\n"
        ),
    }
    return prompt_map.get(data_type, prompt_map["unknown"])

def get_translation_maps(data_type):
    """
    Returns translation maps for user prompt and assistant response based on data type.
    """
    # General translations for assistant responses
    assistant_map = {
        'はい': 'Yes',
        'いいえ': 'No',
        '類似度スコア': 'Confidence Score',
    }

    # Base structure for user prompt translations
    string_replacements = {}
    regex_patterns = []

    # Data-type specific translations for user prompts
    if data_type in ["walmart_amazon_product", "wdc_product"]:
        string_replacements = {
            '以下の2つの商品情報が、実質的に同一の商品を指しているかどうかを判断してください。':
            'Please determine whether the following two product records refer to essentially the same product.',
            '商品名': 'Product Name', 'ブランド': 'Brand', '説明': 'Description', '価格': 'Price', 'モデル番号': 'Model Number',
            'これらは同一の商品ですか？': 'Do these refer to the same product?',
            '回答:': 'Answer:'
        }
        regex_patterns.append((re.compile(r'商品情報(\d)'), r'Product \1'))

    elif data_type == "music":
        string_replacements = {
            '以下の2つの音楽情報が、実質的に同一の楽曲を指しているかどうかを判断してください。':
            'Please determine whether the following two music records refer to essentially the same musical work.',
            'タイトル': 'Title', 'アーティスト': 'Artist', 'アルバム': 'Album', 'リリース日': 'Release Date', '長さ': 'Length',
            'これらは同一の楽曲ですか？': 'Do these refer to the same work?',
            '回答:': 'Answer:'
        }
        regex_patterns.append((re.compile(r'音楽情報(\d)'), r'Record \1'))

    elif data_type == "person":
        string_replacements = {
            '以下の2つの個人情報が、実質的に同一の人物を指しているかどうかを判断してください。':
            'Please determine whether the following two person records refer to essentially the same individual.',
            '名': 'Given Name', '姓': 'Surname', '郵便番号': 'Postcode', '郊外': 'Suburb',
            'これらは同一の人物ですか？': 'Do these refer to the same person?',
            '回答:': 'Answer:'
        }
        regex_patterns.append((re.compile(r'個人情報(\d)'), r'Record \1'))
        
    elif data_type == "bib":
        string_replacements = {
            '以下の2つの書誌情報が、実質的に同一の出版物を指しているかどうかを判断してください。':
            'Please determine whether the following two bibliographic records refer to essentially the same publication.',
            'タイトル': 'Title', '著者': 'Author', '出版社': 'Publisher', '出版日': 'Publication Date',
            'これらは同一の出版物ですか？': 'Do these refer to the same publication?',
            '回答:': 'Answer:'
        }
        regex_patterns.append((re.compile(r'書誌情報(\d)'), r'Record \1'))
        
    else:
        # Fallback for unknown data types
        print(f"Warning: No specific translation map for data_type '{data_type}'. Using a generic map.")
        string_replacements = {
            r'以下の2つの.*?情報が、実質的に同一の.*?を指しているかどうかを判断してください。':
            'Please determine whether the following two records refer to essentially the same entity.',
            '回答:': 'Answer:',
        }
        regex_patterns.append((re.compile(r'.*?情報(\d)'), r'Record \1'))

    user_map = {'strings': string_replacements, 'regex': regex_patterns}
    return user_map, assistant_map


def translate_user_prompt(content, user_map):
    """
    Translates the user prompt content from Japanese to English using the provided map.
    """
    translated_content = content
    
    # Perform simple string replacements
    if 'strings' in user_map:
        for ja, en in user_map['strings'].items():
            translated_content = translated_content.replace(ja, en)
            
    # Perform regex-based substitutions
    if 'regex' in user_map:
        for pattern, replacement in user_map['regex']:
            translated_content = pattern.sub(replacement, translated_content)

    return translated_content

def translate_assistant_response(content, assistant_map):
    """
    Translates the assistant response content from Japanese to English.
    """
    translated_content = content
    for ja, en in assistant_map.items():
        translated_content = translated_content.replace(ja, en)
    return translated_content

def main():
    """
    Main function to run the translation script.
    """
    parser = argparse.ArgumentParser(
        description='Translate Japanese finetuning data (JSONL) to English.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_file', 
        required=True, 
        help='Path to the input Japanese JSONL file.'
    )
    parser.add_argument(
        '--output_file', 
        required=True, 
        help='Path to the output English JSONL file.'
    )
    parser.add_argument(
        '--data_type', 
        required=True, 
        choices=['bib', 'music', 'person', 'walmart_amazon_product', 'wdc_product'],
        help='The type of data being processed.'
    )
    args = parser.parse_args()

    user_map, assistant_map = get_translation_maps(args.data_type)

    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(args.output_file, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                try:
                    data = json.loads(line)
                    messages = data.get("messages")

                    if not messages or len(messages) < 3:
                        print(f"Warning: Skipping malformed line {i+1} in {args.input_file}")
                        continue

                    # 1. Translate system prompt
                    messages[0]['content'] = get_prompts(args.data_type)
                    
                    # 2. Translate user prompt
                    messages[1]['content'] = translate_user_prompt(messages[1]['content'], user_map)

                    # 3. Translate assistant response
                    messages[2]['content'] = translate_assistant_response(messages[2]['content'], assistant_map)
                    
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {i+1} in {args.input_file}")
                except Exception as e:
                    print(f"An error occurred on line {i+1}: {e}")

        print(f"Successfully translated {args.input_file} to {args.output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
