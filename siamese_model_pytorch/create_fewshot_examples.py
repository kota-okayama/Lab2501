# Few-shot学習のためのお手本（文脈内学習例）を生成するスクリプト
import argparse
import csv
import json
import os
import sys
import yaml


# グローバル変数
BIB_DATA = {}
AVAILABLE_FIELDS = []


def load_bib_data(yaml_path):
    """YAMLから書誌データをロードする"""
    global BIB_DATA, AVAILABLE_FIELDS
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        sys.exit(1)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        AVAILABLE_FIELDS = list(data.get("inf_attr", {}).keys())
        for cluster_id, records in data.get("records", {}).items():
            for record in records:
                if record_id := str(record.get("id")):
                    BIB_DATA[record_id] = record.get("data", {})
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中にエラー: {e}")
        sys.exit(1)


def get_record_details_for_prompt(record_id):
    """レコードの詳細なプロンプト文字列を生成する"""
    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"
    
    field_map = {
        'title': 'タイトル', 'artist': 'アーティスト', 'author': '著者',
        'album': 'アルバム', 'publisher': '出版社', 'year': '年',
        'pubdate': '出版日', 'length': '長さ',
        'bib1_title': 'タイトル', 'bib1_author': '著者', 
        'bib1_publisher': '出版社', 'bib1_pubdate': '出版日'
    }
    parts = []
    fields_to_use = AVAILABLE_FIELDS if AVAILABLE_FIELDS else list(bib_details.keys())
    for field in fields_to_use:
        if value := bib_details.get(field):
            display_name = field_map.get(field, field)
            parts.append(f"{display_name}: {value}")
    return "\\n".join(parts)


def get_prompts(data_type):
    """データタイプに応じたプロンプトのエンティティと情報を返す"""
    prompt_map = {
        "bib": {"entity": "文献", "info": "書誌情報"},
        "music": {"entity": "楽曲", "info": "楽曲情報"},
        "person": {"entity": "人物", "info": "人物情報"}
    }
    if data_type not in prompt_map:
        raise ValueError(f"未知のデータタイプです: {data_type}")
    return prompt_map[data_type]


def load_evaluation_details(filepath):
    """ペアごとの評価スコアと正解ラベルをロードする"""
    details = {}
    if not os.path.exists(filepath):
        return details
    try:
        with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = tuple(sorted((row['record_id_1'], row['record_id_2'])))
                    details[key] = {
                        "is_similar_gt": row['ground_truth_similar'].lower() == 'true',
                        "score_before": float(row.get('score_before') or 0.0),
                        "score_after": float(row.get('score_after') or 0.0),
                    }
                except (KeyError, ValueError):
                    continue
    except (IOError, csv.Error) as e:
        print(f"エラー: 評価詳細ファイル ({filepath}) の読み込み中にエラー: {e}")
    return details


def load_inconsistent_triangles(filepath):
    """矛盾三角形の情報をロードする"""
    triangles = []
    if not os.path.exists(filepath):
        return triangles
    try:
        with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    triangles.append({
                        "nodes": (
                            row['triangle_node1'],
                            row['triangle_node2'],
                            row['triangle_node3']
                        ),
                        "score": float(row['inconsistency_score'])
                    })
                except (KeyError, ValueError):
                    continue
    except (IOError, csv.Error) as e:
        print(f"エラー: 矛盾三角形ファイル ({filepath}) の読み込み中にエラー: {e}")
    return triangles


def select_hard_pairs(eval_details, num_hard_neg, num_hard_pos):
    """Hard NegativeとHard Positiveのペアを選定する"""
    hard_negatives = sorted(
        [p for p, d in eval_details.items() if not d['is_similar_gt']],
        key=lambda p: abs(eval_details[p]['score_after'] - 0.5)
    )[:num_hard_neg]
    
    hard_positives = sorted(
        [p for p, d in eval_details.items() if d['is_similar_gt']],
        key=lambda p: abs(eval_details[p]['score_after'] - 0.5)
    )[:num_hard_pos]
    
    return hard_negatives, hard_positives


def create_pair_example(pair, eval_details, data_type_info):
    """単一のペアからFew-shot用のUser-Assistant対話を作成"""
    id1, id2 = pair
    info1 = get_record_details_for_prompt(id1)
    info2 = get_record_details_for_prompt(id2)
    is_similar = eval_details[pair]['is_similar_gt']
    
    user_prompt = (
        f"以下の2つの{data_type_info['info']}が、実質的に同一の"
        f"{data_type_info['entity']}を指しているかどうかを"
        f"判断してください。\\n\\n"
        f"{data_type_info['info']}1:\\n{info1}\\n\\n"
        f"{data_type_info['info']}2:\\n{info2}\\n\\n"
        f"これらは同一の{data_type_info['entity']}ですか？\\n回答:"
    )
    assistant_response = ("はい\\n類似度スコア: 1.0" if is_similar
                          else "いいえ\\n類似度スコア: 0.0")
    
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]


def generate_rule_only_prompt(data_type_info):
    """ルールのみを説明するシステムプロンプトを生成"""
    entity = data_type_info['entity']
    return {
        "system_prompt": (
            f"あなたは2つの情報が同一の{entity}を指すかを判断する専門家です。"
            "あなたの判断は論理的に一貫している必要があります。\n"
            "例えば、AとBが同じで、BとCが同じなら、AとCも同じはずです。"
            "このような推移律の矛盾を避けるように、慎重に判断してください。"
        ),
        "fewshot_examples": []
    }


def generate_fewshot_only_prompt(eval_details, data_type_info, num_pairs=2):
    """HardペアのみをFew-shot例として使用するプロンプトを生成"""
    hard_neg, hard_pos = select_hard_pairs(eval_details, num_pairs, num_pairs)
    examples = []
    for pair in hard_neg + hard_pos:
        examples.extend(create_pair_example(pair, eval_details, data_type_info))
    
    system_prompt = (
        f"あなたは2つの情報が同一の{data_type_info['entity']}を指すかを判断する"
        "専門家です。まず「はい」か「いいえ」で答え、次に0.0から1.0の"
        "類似度スコアを示してください。"
    )
    return {"system_prompt": system_prompt, "fewshot_examples": examples}


def generate_hybrid_prompt(
    inconsistent_triangles, eval_details, data_type_info, num_pairs=2
):
    """ルール説明と矛盾三角形の例を組み合わせたプロンプトを生成"""
    entity = data_type_info['entity']
    info = data_type_info['info']

    # 1. 出力形式に関する厳格な指示を追加
    output_format_instruction = (
        f"まず、2つの{info}が同一の{entity}と思われる場合は「はい」、"
        "そうでない場合は「いいえ」で明確に回答してください。\\n"
        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から"
        "1.0（完全に同一）の範囲で提示してください。\\n"
        "あなたの判断は次のルールに厳密に従う必要があります：\\n"
        " - 類似度スコアが0.5以上の場合、回答は必ず「はい」にしてください。\\n"
        " - 類似度スコアが0.5未満の場合、回答は必ず「いいえ」にしてください。"
    )

    bad_example_texts = []
    # 2. 矛盾した三角形の例を最大3つまで使用する
    num_examples = min(len(inconsistent_triangles), 3)

    if num_examples > 0:
        for i in range(num_examples):
            triangle_data = inconsistent_triangles[i]
            nodes = triangle_data['nodes']
            n1, n2, n3 = nodes

            p1 = tuple(sorted((n1, n2)))
            p2 = tuple(sorted((n2, n3)))
            p3 = tuple(sorted((n1, n3)))

            s1 = eval_details.get(p1, {}).get('score_after', 0.0)
            s2 = eval_details.get(p2, {}).get('score_after', 0.0)
            s3 = eval_details.get(p3, {}).get('score_after', 0.0)

            llm_judgement1 = "はい" if s1 > 0.5 else "いいえ"
            llm_judgement2 = "はい" if s2 > 0.5 else "いいえ"
            llm_judgement3 = "はい" if s3 > 0.5 else "いいえ"

            # 3. IDだけでなく、具体的なレコード情報を含める
            info1_str = get_record_details_for_prompt(n1).replace("\\n", ", ")
            info2_str = get_record_details_for_prompt(n2).replace("\\n", ", ")
            info3_str = get_record_details_for_prompt(n3).replace("\\n", ", ")

            example_text = (
                f"【悪い判断の例 {i+1}】\\n"
                f"記録A: {info1_str}\\n"
                f"記録B: {info2_str}\\n"
                f"記録C: {info3_str}\\n"
                "あなたの以前の判断は以下の通りでした。\\n"
                f"- ペア(A, B) → {llm_judgement1} (スコア: {s1:.2f})\\n"
                f"- ペア(B, C) → {llm_judgement2} (スコア: {s2:.2f})\\n"
                f"- しかし、ペア(A, C) → {llm_judgement3} (スコア: {s3:.2f})\\n"
                "これは推移律 (A=B, B=C ならば A=C) に矛盾しています。"
            )
            bad_example_texts.append(example_text)

        all_bad_examples = "\\n\\n".join(bad_example_texts)
        consistency_prompt = (
            "さらに、あなたの判断は論理的に一貫している必要があります。\\n"
            "例えば、AとBが同じで、BとCが同じなら、AとCも同じでなければなりません。\\n"
            "以下に示すのは、過去のあなたの判断における推移律の矛盾の例です。"
            "このような矛盾した判断は避けてください。\\n\\n"
            f"{all_bad_examples}"
        )
    else:
        # 矛盾する三角形がない場合は、一般的なルールのみを提示
        consistency_prompt = (
            "あなたの判断は論理的に一貫している必要があります。\\n"
            "例えば、AとBが同じで、BとCが同じなら、AとCも同じはずです。"
            "このような推移律の矛盾を避けるように、慎重に判断してください。"
        )

    system_prompt = (
        f"あなたは2つの{info}が同一の{entity}を指すかを判断する専門家です。\\n\\n"
        f"{output_format_instruction}\\n\\n"
        f"{consistency_prompt}"
    )
    
    hard_neg, hard_pos = select_hard_pairs(eval_details, num_pairs, num_pairs)
    examples = []
    for pair in hard_neg + hard_pos:
        if pair in eval_details:
            examples.extend(create_pair_example(pair, eval_details, data_type_info))

    return {"system_prompt": system_prompt, "fewshot_examples": examples}


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot学習のためのお手本を生成するスクリプト"
    )
    parser.add_argument(
        "--inconsistent_triangles_csv", required=True,
        help="矛盾三角形の情報CSV"
    )
    parser.add_argument(
        "--evaluation_details_csv", required=True,
        help="LLM評価の詳細結果CSV"
    )
    parser.add_argument(
        "--ground_truth_yaml", required=True, help="正解情報を含むYAML"
    )
    parser.add_argument(
        "--output_json_path", required=True, help="出力するJSONファイルのパス"
    )
    parser.add_argument(
        "--data_type", required=True, choices=["bib", "music", "person"],
        help="データの種類"
    )
    parser.add_argument(
        "--strategy", required=True,
        choices=["rule_only", "fewshot_only", "hybrid"],
        help="生成するお手本の戦略"
    )
    parser.add_argument(
        "--num_fewshot_pairs", type=int, default=2,
        help="fewshot_only戦略で使うペア数 (positive/negativeそれぞれ)"
    )
    
    args = parser.parse_args()
    
    load_bib_data(args.ground_truth_yaml)
    eval_details = load_evaluation_details(args.evaluation_details_csv)
    inconsistent_triangles = load_inconsistent_triangles(
        args.inconsistent_triangles_csv
    )

    if not eval_details:
        print("エラー: 評価詳細ファイルが空か、読み込めませんでした。")
        sys.exit(1)

    data_type_info = get_prompts(args.data_type)
    
    output_data = {}
    if args.strategy == "rule_only":
        output_data = generate_rule_only_prompt(data_type_info)
    elif args.strategy == "fewshot_only":
        output_data = generate_fewshot_only_prompt(
            eval_details, data_type_info, args.num_fewshot_pairs
        )
    elif args.strategy == "hybrid":
        output_data = generate_hybrid_prompt(
            inconsistent_triangles, eval_details, data_type_info,
            args.num_fewshot_pairs
        )
        
    try:
        with open(args.output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"お手本データを {args.output_json_path} に保存しました。")
    except IOError as e:
        print(f"エラー: 出力ファイル ({args.output_json_path}) の書き込み中にエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 