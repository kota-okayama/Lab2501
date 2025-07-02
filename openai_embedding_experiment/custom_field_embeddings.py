"""
カスタムフィールド指定エンベディング生成スクリプト
ユーザーが任意のフィールド組み合わせを指定できる
"""

import openai
import os
import pickle
import numpy as np
import sys
import json
import argparse

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data_processing.load_yaml_data import load_bibliographic_data


def create_text_representation(record_data, fields):
    """指定フィールドからテキスト表現を作成"""
    parts = []
    for field in fields:
        if field in record_data and record_data[field]:
            value = record_data[field]
            if isinstance(value, list):
                parts.extend([str(v) for v in value if v])
            else:
                parts.append(str(value))
    return " ".join(parts) if parts else None


def get_openai_embeddings(texts_with_ids, model="text-embedding-ada-002"):
    """OpenAI APIでエンベディングを取得"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    results = []
    batch_size = 50
    
    for i in range(0, len(texts_with_ids), batch_size):
        batch = texts_with_ids[i:i + batch_size]
        texts = [item[1] for item in batch]
        ids = [item[0] for item in batch]
        
        try:
            response = openai.embeddings.create(input=texts, model=model)
            embeddings = [item.embedding for item in response.data]
            results.extend(zip(ids, embeddings))
            print(f"Processed batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
    
    return results


def parse_field_combinations(combinations_str):
    """フィールド組み合わせ文字列をパース"""
    combinations = {}
    
    # セミコロンで組み合わせを分割
    for combination in combinations_str.split(';'):
        if ':' in combination:
            name, fields_str = combination.split(':', 1)
            fields = [field.strip() for field in fields_str.split(',')]
            combinations[name.strip()] = fields
        else:
            # 名前が指定されていない場合、フィールド名を結合して名前にする
            fields = [field.strip() for field in combination.split(',')]
            name = '_'.join(fields)
            combinations[name] = fields
    
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description="カスタムフィールド指定エンベディング生成",
        epilog="""
使用例:
  # 単一フィールド
  --fields "bib1_title"
  
  # 複数フィールド
  --fields "bib1_title,bib1_author"
  
  # 複数の組み合わせ（名前付き）
  --fields "title_only:bib1_title;author_only:bib1_author;title_author:bib1_title,bib1_author"
  
  # 複数の組み合わせ（名前なし）
  --fields "bib1_title;bib1_author;bib1_title,bib1_author"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yaml_path", required=True, help="入力YAMLファイルのパス")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--fields", required=True,
        help="使用するフィールド。複数の組み合わせはセミコロン(;)で区切り、名前付きの場合は name:field1,field2 の形式"
    )
    parser.add_argument("--model", default="text-embedding-ada-002", help="OpenAIモデル名")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.yaml_path}")
    records = load_bibliographic_data(args.yaml_path)
    print(f"Loaded {len(records)} records")
    
    # フィールド組み合わせの解析
    combinations = parse_field_combinations(args.fields)
    print(f"Field combinations to process:")
    for name, fields in combinations.items():
        print(f"  {name}: {fields}")
    
    # 各組み合わせの処理
    results_summary = []
    
    for combo_name, fields in combinations.items():
        print(f"\nProcessing {combo_name} with fields: {fields}")
        
        # 利用可能なフィールドかチェック
        valid_fields = ['bib1_title', 'bib1_author', 'bib1_publisher', 'bib1_pubdate']
        invalid_fields = [f for f in fields if f not in valid_fields]
        if invalid_fields:
            print(f"Warning: Invalid fields detected: {invalid_fields}")
            print(f"Valid fields are: {valid_fields}")
            continue
        
        # テキスト表現の作成
        texts_for_api = []
        for record in records:
            record_id = record.get("record_id")
            if not record_id:
                continue
            text = create_text_representation(record.get("data", {}), fields)
            if text:
                texts_for_api.append((record_id, text))
        
        if not texts_for_api:
            print(f"No valid texts for {combo_name}")
            continue
            
        print(f"Found {len(texts_for_api)} valid records for {combo_name}")
        
        # エンベディング生成
        embeddings_with_ids = get_openai_embeddings(texts_for_api, model=args.model)
        
        if not embeddings_with_ids:
            print(f"No embeddings generated for {combo_name}")
            continue
        
        # 保存
        record_ids = [item[0] for item in embeddings_with_ids]
        embeddings = [item[1] for item in embeddings_with_ids]
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        emb_file = os.path.join(args.output_dir, f"embeddings_{combo_name}.npy")
        ids_file = os.path.join(args.output_dir, f"record_ids_{combo_name}.pkl")
        
        np.save(emb_file, embeddings_array)
        with open(ids_file, "wb") as f:
            pickle.dump(record_ids, f)
        
        print(f"Saved {len(record_ids)} embeddings for {combo_name}")
        print(f"  Embeddings: {emb_file}")
        print(f"  Record IDs: {ids_file}")
        
        results_summary.append({
            "name": combo_name,
            "fields": fields,
            "embeddings_file": emb_file,
            "ids_file": ids_file,
            "record_count": len(record_ids),
            "dimension": embeddings_array.shape[1]
        })
    
    # サマリー保存
    summary_file = os.path.join(args.output_dir, "embedding_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted! Summary saved to {summary_file}")
    print(f"Generated embeddings for {len(results_summary)} combinations:")
    for result in results_summary:
        print(f"  - {result['name']}: {result['record_count']} records (dim: {result['dimension']})")


if __name__ == "__main__":
    main() 