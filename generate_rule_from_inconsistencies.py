import argparse
import csv
import json
import os
import sys
import yaml
import asyncio
from openai import AsyncOpenAI
import pandas as pd
from pathlib import Path

# グローバル変数
BIB_DATA = {}
AVAILABLE_FIELDS = []

def load_bib_data(yaml_path):
    """YAMLから書誌データをロードする"""
    global BIB_DATA, AVAILABLE_FIELDS
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}", file=sys.stderr)
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
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中にエラー: {e}", file=sys.stderr)
        sys.exit(1)

def get_record_details_for_prompt(record_id):
    """レコードの詳細なプロンプト文字列を生成する"""
    details = BIB_DATA.get(str(record_id))
    if not details:
        return f"レコードID {record_id} の情報なし"
    
    parts = []
    fields_to_use = AVAILABLE_FIELDS if AVAILABLE_FIELDS else list(details.keys())
    for field in fields_to_use:
        if value := details.get(field):
            parts.append(f"{field}: {value}")
    return "\n".join(parts)

async def generate_rule_with_llm(model_id, examples_text):
    """LLMを使用して矛盾からルールを生成する"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY 環境変数が設定されていません。", file=sys.stderr)
        sys.exit(1)
    
    client = AsyncOpenAI(api_key=api_key)

    system_prompt = (
        "あなたは、レコードリンケージ（名寄せ）の判断ミスを分析する超一流のデータサイエンティストです。"
        "あなたの仕事は、AIが犯した判断の矛盾を、レコードのフィールドレベルまで深く掘り下げて分析することです。"
        "特に、フィールドの欠損、表記の揺れ（略語など）、フォーマットの違いが、"
        "どのようにして矛盾した判断を引き起こしたのかを突き止めてください。"
        "その上で、AIが将来同じ過ちを犯さないようにするための、具体的で実践的なルールを生成してください。"
    )
    
    user_prompt = (
        "AIが「A=BかつB=CならばA=Cである」という推移律に反する判断を下しました。\n"
        "以下の具体例を、フィールドごとに詳細に比較・分析してください。\n"
        "その上で、矛盾の根本原因を解消するための、実践的なルールを箇条書きで生成してください。\n\n"
        f"{examples_text}\n\n"
        "**分析のポイント:**\n"
        "- **フィールドの欠損**: あるレコードに特定のフィールドが存在しないことが、矛盾の原因になっていませんか？\n"
        "- **表記の揺れ**: `Proc.` と `Proceedings` のような略語や、著者名の順序の違いが原因ではありませんか？\n"
        "- **内容の微妙な違い**: タイトルが少しだけ違う、年が1年ずれている、といった違いがどう影響しましたか？\n\n"
        "上記分析に基づき、矛盾の根本原因を解消するための、具体的で実践的なルールを箇条書きで生成してください。\n"
        "ルールは簡潔に、命令形で記述し、説明や前置きは一切含めないでください。"
    )

    try:
        print("\nLLMにルールの生成を依頼します...")
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        rule = completion.choices[0].message.content.strip()
        print(f"生成されたルール:\n---\n{rule}\n---")
        return rule
    except Exception as e:
        print(f"エラー: LLMによるルール生成中にエラーが発生しました: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="矛盾したLLMの判断から、改善のための普遍的なルールを生成するスクリプト"
    )
    parser.add_argument("--inconsistent_triangles_csv", required=True, help="矛盾三角形の情報CSV")
    parser.add_argument("--ground_truth_yaml", required=True, help="正解情報を含むYAML")
    parser.add_argument("--output_json_path", required=True, help="生成したルールを含むJSONの出力パス")
    parser.add_argument("--data_type", required=True, choices=["bib", "music", "person"], help="データの種類")
    parser.add_argument("--num_examples", type=int, default=3, help="LLMに提示する矛盾事例の数")
    parser.add_argument("--model_id", type=str, default="gpt-4o", help="ルール生成に使用するLLMのモデルID")
    args = parser.parse_args()

    load_bib_data(args.ground_truth_yaml)

    try:
        df = pd.read_csv(args.inconsistent_triangles_csv)
        # score_after がない場合を考慮
        if 'inconsistency_score' not in df.columns:
             print(f"エラー: {args.inconsistent_triangles_csv} に 'inconsistency_score' 列がありません。", file=sys.stderr)
             sys.exit(1)
        top_triangles = df.nlargest(args.num_examples, 'inconsistency_score')
    except FileNotFoundError:
        print(f"エラー: 入力CSVファイルが見つかりません: {args.inconsistent_triangles_csv}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"エラー: CSVファイルの処理中にエラー: {e}", file=sys.stderr)
        sys.exit(1)

    if top_triangles.empty:
        print("矛盾事例が見つからなかったため、ルールを生成できませんでした。")
        return

    examples_text_list = []
    for i, row in top_triangles.iterrows():
        n1, n2, n3 = row['triangle_node1'], row['triangle_node2'], row['triangle_node3']
        p12, p23, p31 = row['p_edge12'], row['p_edge23'], row['p_edge31']
        
        text = (
            f"【矛盾事例 {i+1}】\n"
            f"記録A:\n{get_record_details_for_prompt(n1)}\n\n"
            f"記録B:\n{get_record_details_for_prompt(n2)}\n\n"
            f"記録C:\n{get_record_details_for_prompt(n3)}\n\n"
            f"AIの判断:\n"
            f"- ペア(A, B) → {'同一' if p12 > 0.5 else '非同一'} (スコア: {p12:.2f})\n"
            f"- ペア(B, C) → {'同一' if p23 > 0.5 else '非同一'} (スコア: {p23:.2f})\n"
            f"- しかし、ペア(A, C) → {'同一' if p31 > 0.5 else '非同一'} (スコア: {p31:.2f})\n"
            f"この判断は、A=B, B=C ならば A=C という論理に反しており、矛盾しています。"
        )
        examples_text_list.append(text)
    
    examples_text = "\n\n".join(examples_text_list)
    
    generated_rule = asyncio.run(generate_rule_with_llm(args.model_id, examples_text))

    if not generated_rule:
        print("ルールの生成に失敗しました。処理を終了します。")
        sys.exit(1)
        
    # 元となるシステムプロンプトを生成 (create_fewshot_examples.pyから流用)
    info_name_map = {"bib": "書誌情報", "music": "楽曲情報", "person": "人物情報"}
    entity_name_map = {"bib": "文献", "music": "楽曲", "person": "人物"}
    info_name = info_name_map.get(args.data_type, "情報")
    entity_name = entity_name_map.get(args.data_type, "エンティティ")
    
    original_system_prompt = (
        f"あなたは2つの{info_name}が同一の{entity_name}を指すかを判断する専門家です。\n"
        "あなたの判断は次のルールに厳密に従う必要があります：\n"
        " - 類似度スコアが0.5以上の場合、回答は必ず「はい」にしてください。\n"
        " - 類似度スコアが0.5未満の場合、回答は必ず「いいえ」にしてください。\n"
        "また、あなたの判断は論理的に一貫している必要があります。"
        "例えば、AとBが同じで、BとCが同じなら、AとCも同じはずです。"
    )

    new_system_prompt = original_system_prompt + "\nさらに、以下のルールにも従ってください：\n" + generated_rule

    # --- Few-shot Exampleの生成 ---
    # examples_used_for_generationからfew-shot形式に変換する
    fewshot_examples = []
    for _, row in top_triangles.iterrows():
        n1, n2, n3 = row['triangle_node1'], row['triangle_node2'], row['triangle_node3']
        p12, p23, p31 = row['p_edge12'], row['p_edge23'], row['p_edge31']
        t12, t23, t31 = row['true_edge12'], row['true_edge23'], row['true_edge31']

        pairs = [
            (n1, n2, p12, t12, "A", "B"),
            (n2, n3, p23, t23, "B", "C"),
            (n3, n1, p31, t31, "C", "A")
        ]

        for r_id1, r_id2, score, is_true, name1, name2 in pairs:
            bib_info_1 = get_record_details_for_prompt(r_id1)
            bib_info_2 = get_record_details_for_prompt(r_id2)
            
            user_content = (
                f"以下の2つの{info_name}が、実質的に同一の{entity_name}を指しているかどうかを判断してください。\n\n"
                f"{info_name}1:\n{bib_info_1}\n\n"
                f"{info_name}2:\n{bib_info_2}\n\n"
                f"これらは同一の{entity_name}ですか？\n回答:"
            )

            answer = "はい" if is_true else "いいえ"
            # スコアは正解ラベルに基づいて生成する (LLMの予測スコアではなく)
            # ノイズを加えることも考えられるが、ここでは単純に 0.9 or 0.1 とする
            true_score = 0.9 if is_true else 0.1
            assistant_content = f"{answer}\n類似度スコア: {true_score}"

            fewshot_examples.append({"role": "user", "content": user_content})
            fewshot_examples.append({"role": "assistant", "content": assistant_content})


    output_data = {
        "system_prompt": new_system_prompt,
        "fewshot_examples": fewshot_examples
    }

    output_path = Path(args.output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n生成されたルールと新しいシステムプロンプトを {output_path} に保存しました。")
    except IOError as e:
        print(f"エラー: 出力ファイル ({output_path}) の書き込み中にエラー: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 