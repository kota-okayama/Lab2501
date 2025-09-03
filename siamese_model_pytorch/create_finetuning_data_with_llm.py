import argparse
import itertools
import json
import os
import random
import sys

import pandas as pd
import yaml
from openai import OpenAI


def get_prompt_for_selection(data_type: str) -> str:
    """データタイプに応じて、データ選択フェーズのタスク説明を返す"""
    prompt_map = {
        "bib": "2つのレコードが同じ書誌を指しているか",
        "music": "2つのレコードが同じ音楽作品を指しているか",
        "person": "2つのレコードが同じ人物を指しているか",
        "wdc_product": "2つのレコードが同じ商品を指しているか",
        "walmart_amazon_product": "2つのレコードが同じ商品を指しているか",
    }
    task_description = prompt_map.get(
        data_type, "2つのレコードが同じエンティティを指しているか"
    )
    return (
        "これから、2つのレコードのペアのリストを提示します。これらのペアは、"
        f"{task_description}を判定するタスクのデータです。"
    )


def get_prompt_for_finetuning(data_type: str) -> tuple[str, str]:
    """データタイプに応じて、ファインチューニング用のsystemプロンプトとuserプロンプトのテンプレートを返す"""
    jp_type = "情報"
    jp_work = "エンティティ"
    if data_type == "bib":
        jp_type, jp_work = "書誌", "文献"
    elif data_type == "music":
        jp_type, jp_work = "音楽", "作品"
    elif data_type == "person":
        jp_type, jp_work = "人物", "人物"
    elif "product" in data_type:
        jp_type, jp_work = "商品", "商品"

    system_prompt = (
        f"あなたは2つの{jp_type}情報が実質的に同一の{jp_work}を指すかどうかを"
        "判断する専門家です。\\n"
        f"まず、2つの{jp_type}情報が同一の{jp_work}と思われる場合は「はい」、"
        "そうでない場合は「いいえ」で明確に回答してください。\\n"
        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0"
        "（完全に同一）の範囲で提示してください。"
    )

    user_prompt_template = (
        f"以下の2つの{jp_type}情報が、実質的に同一の{jp_work}を指しているか"
        "どうかを判断してください。\\n\\n"
        f"{jp_type}情報1:\\n{{content1}}\\n\\n"
        f"{jp_type}情報2:\\n{{content2}}\\n\\n"
        f"これらは同一の{jp_work}ですか？\\n回答:"
    )
    return system_prompt, user_prompt_template


def format_record_data(data_dict, data_type):
    """
    レコードのデータ辞書をLLMが読みやすい改行区切りの文字列にフォーマットする
    (create_finetuning_data_from_strategies.py との前処理統一のため)
    """
    if data_type == "music":
        title = data_dict.get("title", "タイトル不明")
        artist = data_dict.get("artist", "アーティスト不明")
        album = data_dict.get("album", "アルバム不明")
        release_date = data_dict.get("release_date", "リリース日不明")
        return (f"タイトル: {title}\nアーティスト: {artist}\n"
                f"アルバム: {album}\nリリース日: {release_date}")
    elif data_type == "person":
        name = data_dict.get("name", "名前不明")
        affiliation = data_dict.get("affiliation", "所属不明")
        return f"名前: {name}\n所属: {affiliation}"
    elif data_type == "walmart_amazon_product":
        title = data_dict.get("title", "商品名不明")
        brand = data_dict.get("brand", "ブランド不明")
        modelno = data_dict.get("modelno", "モデル番号不明")
        price = data_dict.get("price", "価格不明")
        return f"商品名: {title}\nブランド: {brand}\nモデル番号: {modelno}\n価格: {price}"
    elif data_type == "wdc_product":
        title = data_dict.get("title", "商品名不明")
        brand = data_dict.get("brand", "ブランド不明")
        description = data_dict.get("description", "説明不明")
        price = data_dict.get("price", "価格不明")
        return f"商品名: {title}\nブランド: {brand}\n説明: {description}\n価格: {price}"
    else:  # bib or default
        title = data_dict.get("bib1_title", "タイトル不明")
        author = data_dict.get("bib1_author", "著者不明")
        publisher = data_dict.get("bib1_publisher", "出版社不明")
        pubdate = data_dict.get("bib1_pubdate", "出版日不明")
        return (f"タイトル: {title}\n著者: {author}\n"
                f"出版社: {publisher}\n出版日: {pubdate}")


def load_and_prepare_data_from_yml(yml_path, data_type, total_candidates=None):
    """
    YAMLファイルからレコードを読み込み、全てのペア候補とマスターデータを生成する
    """
    print(f"{yml_path} からデータを読み込んでいます...")
    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"エラー: YAMLファイルが見つかりません: {yml_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"エラー: YAMLファイルの解析に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    records_by_cluster = data.get("records", {})
    all_records = []
    record_to_cluster_map = {}
    master_data_dict = {}

    for cluster_id, records in records_by_cluster.items():
        for record in records:
            record_id = record["id"]
            all_records.append(record)
            record_to_cluster_map[record_id] = cluster_id
            master_data_dict[record_id] = format_record_data(
                record["data"], data_type
            )

    print("ポジティブペアとネガティブペアを生成しています...")
    positive_pairs = []
    for records in records_by_cluster.values():
        record_ids = [r["id"] for r in records]
        if len(record_ids) > 1:
            # 常にソートされたタプルとしてペアを追加し、順序の問題をなくす
            for id1, id2 in itertools.combinations(record_ids, 2):
                positive_pairs.append(tuple(sorted((id1, id2))))

    num_positive_pairs = len(positive_pairs)
    all_record_ids = list(record_to_cluster_map.keys())
    
    if total_candidates and total_candidates > num_positive_pairs:
        num_negative_to_generate = total_candidates - num_positive_pairs
    else:
        # デフォルトの挙動: ポジティブペアと同数を生成
        num_negative_to_generate = num_positive_pairs

    # 生成可能なネガティブペアの最大数を計算
    total_possible_pairs = len(all_record_ids) * (len(all_record_ids) - 1) // 2
    max_possible_negative_pairs = total_possible_pairs - num_positive_pairs

    if num_negative_to_generate > max_possible_negative_pairs:
        print(
            f"警告: 要求されたネガティブペア数 ({num_negative_to_generate}) は "
            f"生成可能な最大数 ({max_possible_negative_pairs}) を超えています。"
        )
        print("生成可能な全てのネガティブペアを使用します。")
        num_negative_to_generate = max_possible_negative_pairs

    negative_pairs_set = set()
    while len(negative_pairs_set) < num_negative_to_generate:
        id1, id2 = random.sample(all_record_ids, 2)
        if record_to_cluster_map[id1] != record_to_cluster_map[id2]:
            # ペアの順序を統一して重複を防ぐ
            pair = tuple(sorted((id1, id2)))
            negative_pairs_set.add(pair)
    
    negative_pairs = list(negative_pairs_set)

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    print(
        f"生成されたペア数: ポジティブ {len(positive_pairs)}, "
        f"ネガティブ {len(negative_pairs)}, 合計 {len(all_pairs)}"
    )
    
    # ポジティブペアのセットを返して、後でラベルを判定できるようにする
    return all_pairs, master_data_dict, set(positive_pairs)


def select_one_batch_with_llm(
    client, candidate_pairs, master_data_dict, num_to_select, data_type
):
    """
    LLMに一回の選択（1バッチ分）を依頼する
    """
    task_description = get_prompt_for_selection(data_type)
    prompt_header = f"""あなたは、機械学習モデルの訓練に使うためのデータを選択する専門家（アクティブラーナー）です。
{task_description}
モデルの学習効率が最大になるように、最も有益だと考えられるペアを {num_to_select} 個選択してください。
選択する際は、多様性、曖昧さ、代表性などを考慮してください。

思考のステップを一つずつ記述してください。

最終的な回答は、選択したペアのインデックスのみをカンマ区切りで、一行で出力してください。思考プロセスや他の説明は一切含めないでください。
例: 1, 5, 12, 28

以下がデータのペアです:
---
"""
    prompt_body_parts = []
    for i, (id1, id2) in enumerate(candidate_pairs):
        content1 = master_data_dict.get(id1, "[コンテンツ不明]")
        content2 = master_data_dict.get(id2, "[コンテンツ不明]")
        prompt_body_parts.append(
            f"{i}: (レコード1: '{content1}', レコード2: '{content2}')"
        )
    
    full_prompt = prompt_header + "\n".join(prompt_body_parts)

    print(
        f"{len(candidate_pairs)} 件の候補から {num_to_select} 件を"
        "選択するようLLMに依頼します..."
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # コストと速度のバランスが良いモデル
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in selecting data for "
                        "machine learning model training."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        print(f"エラー: OpenAI APIへの問い合わせ中にエラー: {e}", file=sys.stderr)
        return []

    try:
        # 思考プロセスが含まれる場合でも、最後の行に数値リストがあると仮定
        last_line = response_text.strip().split("\n")[-1]
        # 数字とカンマ、スペース以外の文字を削除してパースの頑健性を高める
        cleaned_line = "".join(filter(lambda char: char.isdigit() or char == ',' or char.isspace(), last_line))
        selected_indices = [int(i.strip()) for i in cleaned_line.split(",") if i.strip()]
        
        if any(i >= len(candidate_pairs) for i in selected_indices):
            raise ValueError("LLMが範囲外のインデックスを返しました。")
        
        selected_pairs = [candidate_pairs[i] for i in selected_indices]
        print(f"LLMが {len(selected_pairs)} 件のペアを選択しました。")
        return selected_pairs
    except (ValueError, IndexError):
        print(f"エラー: LLMの応答解析に失敗。応答: '{response_text}'", file=sys.stderr)
        return []


def save_as_jsonl(
    pairs, master_data_dict, positive_pairs_set, output_file, data_type
):
    """
    選択されたペアをファインチューニング用のJSONL形式で保存する
    """
    print(f"{output_file} にJSONL形式で保存しています...")

    system_prompt, user_prompt_template = get_prompt_for_finetuning(data_type)

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            # ペアの順序をソートしてタプルに変換し、セットでの検索を確実にする
            sorted_pair = tuple(sorted(pair))
            is_match = sorted_pair in positive_pairs_set
            
            assistant_response = (
                "はい\\n類似度スコア: 1.0" if is_match else "いいえ\\n類似度スコア: 0.0"
            )

            id1, id2 = pair
            content1 = master_data_dict.get(id1, "[コンテンツ不明]")
            content2 = master_data_dict.get(id2, "[コンテンツ不明]")

            user_content = user_prompt_template.format(
                content1=content1, content2=content2
            )
            json_line = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_response},
                ]
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print("JSONLファイルの保存が完了しました。")


def main():
    parser = argparse.ArgumentParser(
        description="LLMを使用してYAMLファイルからファインチューニング用のデータペアを選択します。"
    )
    parser.add_argument(
        "--record_yml", type=str, required=True,
        help="レコード情報が記載されたYAMLファイルのパス"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="出力ファイルを保存するディレクトリのパス"
    )
    parser.add_argument(
        "--total_samples", type=int, required=True,
        help="最終的に選択したいペアの総数"
    )
    parser.add_argument(
        "--num_samples_per_request", type=int, default=32,
        help="1回のLLMリクエストで選択するペアの数"
    )
    parser.add_argument(
        "--num_candidates", type=int, default=200,
        help="1回のLLMリクエストで提示する候補の数"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="選択されたペアを保存するCSVファイル名"
    )
    parser.add_argument(
        "--output_jsonl", type=str,
        help="ファインチューニング用に選択されたペアを保存するJSONLファイル名"
    )
    parser.add_argument(
        "--total_candidates", type=int, default=None,
        help="生成する候補ペアの総数。指定しない場合、ポジティブペア数の2倍になります。"
    )
    parser.add_argument(
        "--data_type", type=str, required=True,
        choices=[
            "bib",
            "music",
            "person",
            "wdc_product",
            "walmart_amazon_product",
        ],
        help="データの種類 (プロンプト生成に利用)"
    )
    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)

    if "OPENAI_API_KEY" not in os.environ:
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)
    client = OpenAI()

    all_pairs, master_data_dict, positive_pairs_set = (
        load_and_prepare_data_from_yml(
            args.record_yml, args.data_type, args.total_candidates
        )
    )
    
    unselected_pairs_set = {tuple(sorted(p)) for p in all_pairs}
    final_selected_pairs = []

    iteration = 1
    while len(final_selected_pairs) < args.total_samples:
        print(f"\n--- イテレーション {iteration} ---")
        print(f"目標: {args.total_samples}件 / 現在: {len(final_selected_pairs)}件")

        if not unselected_pairs_set:
            print("警告: 未選択のペアがなくなりました。")
            break
        
        num_candidates = min(args.num_candidates, len(unselected_pairs_set))
        candidate_pairs_tuples = random.sample(
            list(unselected_pairs_set), num_candidates
        )
        # LLMにはタプルではなくリストで渡す
        candidate_pairs = [list(p) for p in candidate_pairs_tuples]
        
        num_to_select = min(
            args.num_samples_per_request,
            args.total_samples - len(final_selected_pairs)
        )

        selected_batch = select_one_batch_with_llm(
            client, candidate_pairs, master_data_dict,
            num_to_select, args.data_type
        )
        
        if selected_batch:
            final_selected_pairs.extend(selected_batch)
            # selected_batchは[[id1, id2], ...]の形式なので、ソートしてタプルに変換
            for pair in selected_batch:
                unselected_pairs_set.discard(tuple(sorted(pair)))
        else:
            print("このイテレーションではペアが選択されませんでした。処理を続行します。")

        iteration += 1

    # CSV出力
    output_csv_path = os.path.join(args.output_dir, args.output_csv)
    output_df = pd.DataFrame(
        final_selected_pairs, columns=["record_id_1", "record_id_2"]
    )
    output_df.to_csv(output_csv_path, index=False)
    print(
        f"\n処理完了。合計 {len(final_selected_pairs)} 件のペアを "
        f"{output_csv_path} に保存しました。"
    )

    # JSONL出力 (引数が指定されている場合のみ)
    if args.output_jsonl:
        output_jsonl_path = os.path.join(args.output_dir, args.output_jsonl)
        save_as_jsonl(
            final_selected_pairs, master_data_dict,
            positive_pairs_set, output_jsonl_path, args.data_type
        )


if __name__ == "__main__":
    main()