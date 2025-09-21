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


def get_prompts(data_type):
    """
    データタイプに応じて、ファインチューニング用のsystemプロンプトを返す
    (他スクリプトと統一)
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


def get_user_prompt(record1_details, record2_details, data_type):
    """
    データタイプに応じて、ファインチューニング用のuserプロンプトを生成する
    """
    if data_type == "walmart_amazon_product" or data_type == "wdc_product":
        task = "product"
        entity = "product"
    elif data_type == "music":
        task = "music records"
        entity = "musical work"
    elif data_type == "person":
        task = "person records"
        entity = "individual"
    elif data_type == "bib":
        task = "bibliographic records"
        entity = "publication"
    else:
        task = "records"
        entity = "entity"

    return (
        f"Please determine whether the following two {task} refer to essentially the same {entity}.\\n\\n"
        f"Record 1:\\n{record1_details}\\n\\n"
        f"Record 2:\\n{record2_details}\\n\\n"
        f"Do these refer to the same {entity}?\\nAnswer:"
    )


def create_finetuning_message(record1_id, record2_id, is_truly_similar,
                              master_data_dict, data_type):
    """
    ファインチューニングメッセージを作成する (他スクリプトと統一)
    """
    system_prompt = get_prompts(data_type)
    
    content1 = master_data_dict.get(record1_id, "[コンテンツ不明]")
    content2 = master_data_dict.get(record2_id, "[コンテンツ不明]")
    user_prompt = get_user_prompt(content1, content2, data_type)
    
    if is_truly_similar:
        assistant_response = "Yes\\nConfidence Score: 1.0"
    else:
        assistant_response = "No\\nConfidence Score: 0.0"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "record_id_1": str(record1_id),
        "record_id_2": str(record2_id)
    }


def format_record_data(data_dict, data_type):
    """
    レコードのデータ辞書をLLMが読みやすい改行区切りの文字列にフォーマットする
    (get_record_details_for_finetuning_prompt と同じロジックに統一)
    """
    if data_type == "music":
        title = data_dict.get("title", "Unknown")
        artist = data_dict.get("artist", "Unknown")
        album = data_dict.get("album", "Unknown")
        release_date = data_dict.get("release_date", "Unknown")
        length = data_dict.get("length", "Unknown")
        return (f"Title: {title}\nArtist: {artist}\n"
                f"Album: {album}\nRelease Date: {release_date}\nLength: {length}")
    elif data_type == "person":
        givenname = data_dict.get("givenname", "Unknown")
        surname = data_dict.get("surname", "Unknown")
        postcode = data_dict.get("postcode", "Unknown")
        suburb = data_dict.get("suburb", "Unknown")
        return (f"Given Name: {givenname}\nSurname: {surname}\nPostcode: {postcode}\nSuburb: {suburb}")

    elif data_type == "walmart_amazon_product":
        name = data_dict.get("title", "Unknown")
        brand = data_dict.get("brand", "Unknown")
        modelno = data_dict.get("modelno", "Unknown")
        price = data_dict.get("price", "Unknown")
        return f"Product Name: {name}\nBrand: {brand}\nModel Number: {modelno}\nPrice: {price}"
    elif data_type == "wdc_product":
        name = data_dict.get("title", "Unknown")
        brand = data_dict.get("brand", "Unknown")
        description = data_dict.get("description", "Unknown")
        price = data_dict.get("price", "Unknown")
        return f"Product Name: {name}\nBrand: {brand}\nDescription: {description}\nPrice: {price}"
    else:  # bib or default
        title = data_dict.get("bib1_title", "Unknown")
        author = data_dict.get("bib1_author", "Unknown")
        publisher = data_dict.get("bib1_publisher", "Unknown")
        pubdate = data_dict.get("bib1_pubdate", "Unknown")
        return (f"Title: {title}\nAuthor: {author}\n"
                f"Publisher: {publisher}\nPublication Date: {pubdate}")


def load_and_prepare_data_from_yml(yml_path, data_type, labeled_pairs, total_candidates=None):
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
                pair = tuple(sorted((id1, id2)))
                if pair not in labeled_pairs:
                    positive_pairs.append(pair)

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
            if pair not in labeled_pairs:
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
選択する際は、多様性、曖昧さ、代表性、データのバランスなどを考慮してください。

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
            model="gpt-4o",  # コストと速度のバランスが良いモデル
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
        cleaned_line = "".join(filter(
            lambda char: char.isdigit() or char == ',' or char.isspace(), 
            last_line
        ))
        selected_indices = [
            int(i.strip()) for i in cleaned_line.split(",") if i.strip()
        ]
        
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
    選択されたペアをファインチューニング用のJSONL形式で保存する (形式を統一)
    """
    print(f"{output_file} にJSONL形式で保存しています...")

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            # ペアの順序をソートしてタプルに変換し、セットでの検索を確実にする
            sorted_pair = tuple(sorted(pair))
            is_match = sorted_pair in positive_pairs_set
            
            message = create_finetuning_message(
                pair[0], pair[1], is_match, master_data_dict, data_type
            )
            # OpenAI APIは 'messages' キーのみを要求するため、それ以外は除外
            openai_entry = {"messages": message["messages"]}
            f.write(json.dumps(openai_entry, ensure_ascii=False) + "\n")
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
    parser.add_argument(
        '--labeled_pairs_csv',
        type=str,
        default=None,
        help='過去にラベル付けされたペアのCSVファイルパス。これらのペアはサンプリングから除外されます。'
    )
    args = parser.parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)

    if "OPENAI_API_KEY" not in os.environ:
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)
    client = OpenAI()

    # ラベル済みペアをロード
    labeled_pairs = set()
    if args.labeled_pairs_csv and os.path.exists(args.labeled_pairs_csv):
        print(f"読み込み中: {args.labeled_pairs_csv}")
        try:
            labeled_df = pd.read_csv(args.labeled_pairs_csv)
            for _, row in labeled_df.iterrows():
                pair = tuple(sorted((str(row['record_id_1']), str(row['record_id_2']))))
                labeled_pairs.add(pair)
            print(f"{len(labeled_pairs)}件のラベル済みペアをロードしました。")
        except Exception as e:
            print(f"警告: ラベル済みペアファイルの読み込みに失敗: {e}")

    all_pairs, master_data_dict, positive_pairs_set = (
        load_and_prepare_data_from_yml(
            args.record_yml, args.data_type, labeled_pairs, args.total_candidates
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

    # (追加) 最終的なデータバランス調整 (50:50目標)
    print("\n最終的なデータバランス調整を行います...")
    positive_pairs_selected = [p for p in final_selected_pairs if tuple(sorted(p)) in positive_pairs_set]
    negative_pairs_selected = [p for p in final_selected_pairs if tuple(sorted(p)) not in positive_pairs_set]

    target_count = args.total_samples // 2
    
    # 過剰なサンプルをランダムに削除
    if len(positive_pairs_selected) > target_count:
        print(f"正例が多すぎるため、{len(positive_pairs_selected) - target_count}件をランダムに削除します。")
        positive_pairs_selected = random.sample(positive_pairs_selected, target_count)
    if len(negative_pairs_selected) > target_count:
        print(f"負例が多すぎるため、{len(negative_pairs_selected) - target_count}件をランダムに削除します。")
        negative_pairs_selected = random.sample(negative_pairs_selected, target_count)
        
    balanced_pairs = positive_pairs_selected + negative_pairs_selected

    # 不足分をランダムに追加
    needed_positive = target_count - len(positive_pairs_selected)
    needed_negative = (args.total_samples - target_count) - len(negative_pairs_selected)

    if needed_positive > 0 or needed_negative > 0:
        print(f"バランス調整のためさらにペアを追加: 正例+{needed_positive}, 負例+{needed_negative}")
        
        # 現在選択済みのペアを集合に
        current_selected_set = {tuple(sorted(p)) for p in balanced_pairs}
        
        # 候補となる未選択のペア
        positive_candidates = [p for p in positive_pairs_set if p not in current_selected_set]
        all_negative_pairs = unselected_pairs_set - positive_pairs_set
        negative_candidates = [p for p in all_negative_pairs if p not in current_selected_set]

        # 正例を追加
        if needed_positive > 0 and positive_candidates:
            num_to_add = min(needed_positive, len(positive_candidates))
            added_pos = random.sample(positive_candidates, num_to_add)
            balanced_pairs.extend(added_pos)

        # 負例を追加
        if needed_negative > 0 and negative_candidates:
            num_to_add = min(needed_negative, len(negative_candidates))
            added_neg = random.sample(negative_candidates, num_to_add)
            balanced_pairs.extend(added_neg)

    # CSV出力
    output_csv_path = os.path.join(args.output_dir, args.output_csv)
    output_df = pd.DataFrame(
        balanced_pairs, columns=["record_id_1", "record_id_2"]
    )
    output_df.to_csv(output_csv_path, index=False)
    print(
        f"\n処理完了。合計 {len(balanced_pairs)} 件のペアを "
        f"{output_csv_path} に保存しました。"
    )

    # JSONL出力 (引数が指定されている場合のみ)
    if args.output_jsonl:
        output_jsonl_path = os.path.join(args.output_dir, args.output_jsonl)
        save_as_jsonl(
            balanced_pairs, master_data_dict,
            positive_pairs_set, output_jsonl_path, args.data_type
        )


if __name__ == "__main__":
    main()