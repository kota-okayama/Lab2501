import json
import os
import yaml
import sys
import argparse
import pandas as pd

# --- グローバル設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_RESULTS_FILENAME = (
    "human_review_simulation_accuracy_sample2000_100.csv"
)
SIMULATION_RESULTS_PATH = os.path.join(BASE_DIR, SIMULATION_RESULTS_FILENAME)

PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024"
RECORD_YAML_FILENAME = "sampled_data_2000.yml"
RECORD_YAML_PATH = os.path.join(
    PROJECT_ROOT_ASSUMED,
    BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT,
    RECORD_YAML_FILENAME,
)

OUTPUT_JSONL_FILENAME = "finetuning_data_with_llm_score.jsonl"
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, OUTPUT_JSONL_FILENAME)

# グローバル変数として書誌データを保持
BIB_DATA = {}
RECORD_ID_TO_CLUSTER_ID = {}  # Add this global variable
BIB_DATA = {}
GROUND_TRUTH_CLUSTERS = {}


# --- 書誌データ読み込み関連関数 (evaluate_pairs_with_openai_async.py から拝借・調整) ---
def load_bib_data_for_finetuning(yaml_path):
    global BIB_DATA, RECORD_ID_TO_CLUSTER_ID
    BIB_DATA = {}
    RECORD_ID_TO_CLUSTER_ID = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        sys.exit(1)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            processed_record_ids_for_bib_data = set()
            processed_record_ids_for_cluster_map = set()

            if isinstance(possible_records_dict, dict):
                for (
                    key,
                    value_list,
                ) in possible_records_dict.items():
                    if (key in ["version", "type", "id", "summary", "inf_attr"]
                            and possible_records_dict is all_data):
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            cluster_id_val = None
                            actual_bib_data = {}

                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])
                                cluster_id_val = record.get("cluster_id")

                                if "data" in record and isinstance(
                                    record["data"], dict
                                ):
                                    actual_bib_data = record["data"]
                                else:
                                    actual_bib_data = {
                                        k_rec: v_rec
                                        for k_rec, v_rec in record.items()
                                        if k_rec not in ["id", "cluster_id"]
                                    }

                                if record_id_str and actual_bib_data:
                                    if (record_id_str not in
                                            processed_record_ids_for_bib_data):
                                        BIB_DATA[record_id_str] = actual_bib_data
                                        processed_record_ids_for_bib_data.add(
                                            record_id_str
                                        )

                                    if cluster_id_val is not None:
                                        RECORD_ID_TO_CLUSTER_ID[
                                            record_id_str
                                        ] = cluster_id_val
                                        processed_record_ids_for_cluster_map.add(
                                            record_id_str
                                        )

                                elif record_id_str and not actual_bib_data:
                                    print(
                                        f"警告: レコードID {record_id_str} に有効な"
                                        f"書誌データが見つかりませんでした。"
                                        f"BIB_DATAへの登録をスキップします。"
                                    )

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。"
                  "YAMLの構造を確認してください。")
            sys.exit(1)
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        print(
            f"{len(RECORD_ID_TO_CLUSTER_ID)} 件の record_id と "
            "cluster_id のマッピングをロードしました。"
        )
        if not RECORD_ID_TO_CLUSTER_ID:
            print(
                f"警告: {yaml_path} から cluster_id を含むレコードが見つからな"
                f"かったか、マッピングの作成に失敗しました。"
                f"ランダム非一致ペアの生成が困難または不可能になります。"
            )

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が"
              f"正しくありません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に"
              f"予期せぬエラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def load_bib_data_and_gt_clusters(yaml_path):
    global BIB_DATA, RECORD_ID_TO_CLUSTER_ID, GROUND_TRUTH_CLUSTERS
    BIB_DATA = {}
    RECORD_ID_TO_CLUSTER_ID = {}
    GROUND_TRUTH_CLUSTERS = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        sys.exit(1)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            processed_record_ids_for_bib_data = set()
            processed_record_ids_for_cluster_map = set()

            if isinstance(possible_records_dict, dict):
                for (
                    key,
                    value_list,
                ) in possible_records_dict.items():
                    if (key in ["version", "type", "id", "summary", "inf_attr"]
                            and possible_records_dict is all_data):
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            cluster_id_val = None
                            actual_bib_data = {}

                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])
                                cluster_id_val = record.get("cluster_id")

                                if "data" in record and isinstance(
                                    record["data"], dict
                                ):
                                    actual_bib_data = record["data"]
                                else:
                                    actual_bib_data = {
                                        k_rec: v_rec
                                        for k_rec, v_rec in record.items()
                                        if k_rec not in ["id", "cluster_id"]
                                    }

                                if record_id_str and actual_bib_data:
                                    if (record_id_str not in
                                            processed_record_ids_for_bib_data):
                                        BIB_DATA[record_id_str] = actual_bib_data
                                        processed_record_ids_for_bib_data.add(
                                            record_id_str
                                        )

                                    if cluster_id_val is not None:
                                        RECORD_ID_TO_CLUSTER_ID[
                                            record_id_str
                                        ] = cluster_id_val
                                        processed_record_ids_for_cluster_map.add(
                                            record_id_str
                                        )

                                elif record_id_str and not actual_bib_data:
                                    print(
                                        f"警告: レコードID {record_id_str} に有効な"
                                        f"書誌データが見つかりませんでした。"
                                        f"BIB_DATAへの登録をスキップします。"
                                    )

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。"
                  "YAMLの構造を確認してください。")
            sys.exit(1)
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        print(
            f"{len(RECORD_ID_TO_CLUSTER_ID)} 件の record_id と "
            "cluster_id のマッピングをロードしました。"
        )
        if not RECORD_ID_TO_CLUSTER_ID:
            print(
                f"警告: {yaml_path} から cluster_id を含むレコードが見つからな"
                f"かったか、マッピングの作成に失敗しました。"
                f"ランダム非一致ペアの生成が困難または不可能になります。"
            )

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が"
              f"正しくありません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に"
              f"予期せぬエラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def get_record_details_for_finetuning_prompt(record_id):
    if not BIB_DATA:
        print("エラー: 書誌データがロードされていません。"
              "(get_record_details_for_finetuning_prompt)")
        return "情報取得エラー: BIB_DATA未ロード"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    title = bib_details.get("bib1_title", "タイトル不明")
    authors_str = bib_details.get("bib1_author", "著者不明")
    publisher = bib_details.get("bib1_publisher", "出版社不明")
    pubdate = bib_details.get("bib1_pubdate", "出版日不明")
    return (f"タイトル: {title}\n著者: {authors_str}\n"
            f"出版社: {publisher}\n出版日: {pubdate}")


def get_prompts(data_type):
    prompt_map = {
        "bib": (
            "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\\n"
            "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
        "music": (
            "あなたは2つの音楽情報が実質的に同一の作品を指すかどうかを判断する専門家です。\\n"
            "まず、2つの音楽情報が同一の作品と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
        "person": (
            "あなたは2つの人物情報が実質的に同一の人物を指すかどうかを判断する専門家です。\\n"
            "まず、2つの人物情報が同一の人物と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
        "unknown": (
            "あなたは2つの情報が実質的に同一のものを指すかどうかを判断する専門家です。\\n"
            "まず、2つの情報が同一のものと思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
    }
    return prompt_map.get(data_type, prompt_map["unknown"])


def create_finetuning_message(record1_id, record2_id, is_truly_similar,
                              data_type, score=None):
    system_prompt = get_prompts(data_type)
    user_prompt = (
        f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\\n\\n"
        f"書誌情報1:\\n{get_record_details_for_finetuning_prompt(record1_id)}\\n\\n"
        f"書誌情報2:\\n{get_record_details_for_finetuning_prompt(record2_id)}\\n\\n"
        "これらは同一の文献ですか？\\n回答:"
    )
    if is_truly_similar:
        assistant_response = "はい\\n類似度スコア: 1.0"
    else:
        assistant_response = "いいえ\\n類似度スコア: 0.0"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


# --- メイン処理 ---
def main(args):
    """
    矛盾する三角形と判断が難しいペアを組み合わせて、
    ファインチューニング用のデータセットを生成する。
    """
    print("ファインチューニング用データ作成処理を開始します...")

    # 正解データのロード
    load_bib_data_and_gt_clusters(args.ground_truth_yaml)

    finetuning_samples = []
    seen_pairs = set()

    # 1. 矛盾する三角形のペアを追加
    try:
        inconsistent_df = pd.read_csv(args.inconsistent_triangles_csv)
        temp_samples = []
        for _, row in inconsistent_df.iterrows():
            # 3つのペアを処理: (node1,node2), (node2,node3), (node1,node3)
            pairs_data = [
                # ペア1: node1 - node2
                {
                    'id1': row['triangle_node1'],
                    'id2': row['triangle_node2'],
                    'is_similar': row['true_edge12'],
                    'score': row['p_edge12']
                },
                # ペア2: node2 - node3
                {
                    'id1': row['triangle_node2'],
                    'id2': row['triangle_node3'],
                    'is_similar': row['true_edge23'],
                    'score': row['p_edge23']
                },
                # ペア3: node1 - node3
                {
                    'id1': row['triangle_node1'],
                    'id2': row['triangle_node3'],
                    'is_similar': row['true_edge31'],
                    'score': row['p_edge31']
                }
            ]

            for pair_data in pairs_data:
                message = create_finetuning_message(
                    pair_data['id1'], pair_data['id2'],
                    pair_data['is_similar'], args.data_type,
                    pair_data['score']
                )
                temp_samples.append(message)

        # 矛盾ペアの重複除去
        for sample in temp_samples:
            user_content = sample['messages'][1]['content']
            if user_content not in seen_pairs:
                seen_pairs.add(user_content)
                finetuning_samples.append(sample)

        print(
            f"矛盾ペア処理: {len(inconsistent_df) * 3} 件の候補から "
            f"{len(finetuning_samples)} 件のユニークなペアを追加しました。"
        )

    except FileNotFoundError:
        print(f"警告: 矛盾ペアファイルが見つかりません: {args.inconsistent_triangles_csv}")
    except Exception as e:
        print(f"警告: 矛盾ペアファイルの処理中にエラー: {e}")
        if 'inconsistent_df' in locals():
            print(f"  利用可能なカラム: {list(inconsistent_df.columns)}")
        else:
            print("  データフレームを読み込めませんでした")

    # 2. Hard negative/positive ペアを追加してバランス調整
    # 現在の正例・負例を数える
    positive_count = sum(1 for sample in finetuning_samples
                         if 'はい' in sample['messages'][2]['content'])
    negative_count = len(finetuning_samples) - positive_count

    print(f"矛盾ペアの内訳: 正例={positive_count}件, 負例={negative_count}件")

    try:
        # バランスを取るために必要な数を計算
        target_count = max(positive_count, negative_count)
        needed_positive = target_count - positive_count
        needed_negative = target_count - negative_count

        if needed_positive > 0 or needed_negative > 0:
            print(f"バランス調整目標: 正例・負例を各{target_count}件に調整")
            print(f"必要な追加数: 正例={needed_positive}件, 負例={needed_negative}件")

            details_df = pd.read_csv(args.evaluation_details_csv)
            details_df['abs_score_dist'] = \
                (details_df[args.score_column] - 0.5).abs()
            hard_pairs_df = details_df.sort_values(by='abs_score_dist')

            hard_positive_df = hard_pairs_df[
                hard_pairs_df['ground_truth_similar']
            ]
            hard_negative_df = hard_pairs_df[
                ~hard_pairs_df['ground_truth_similar']
            ]

            added_positive = 0
            if needed_positive > 0:
                for _, row in hard_positive_df.iterrows():
                    if added_positive >= needed_positive:
                        break
                    message = create_finetuning_message(
                        row['record_id_1'], row['record_id_2'],
                        row['ground_truth_similar'], args.data_type,
                        row[args.score_column]
                    )
                    user_content = message['messages'][1]['content']
                    if user_content not in seen_pairs:
                        finetuning_samples.append(message)
                        seen_pairs.add(user_content)
                        added_positive += 1

            added_negative = 0
            if needed_negative > 0:
                for _, row in hard_negative_df.iterrows():
                    if added_negative >= needed_negative:
                        break
                    message = create_finetuning_message(
                        row['record_id_1'], row['record_id_2'],
                        row['ground_truth_similar'], args.data_type,
                        row[args.score_column]
                    )
                    user_content = message['messages'][1]['content']
                    if user_content not in seen_pairs:
                        finetuning_samples.append(message)
                        seen_pairs.add(user_content)
                        added_negative += 1

            print(f"Hard ペア追加完了: 正例={added_positive}件, "
                  f"負例={added_negative}件")
        else:
            print("矛盾ペアのバランスが取れているため、Hardペアの追加はスキップします。")

    except FileNotFoundError:
        print(f"警告: 評価詳細ファイルが見つかりません: "
              f"{args.evaluation_details_csv}。バランス調整は行われません。")
    except Exception as e:
        print(f"警告: 評価詳細ファイルの処理中にエラーが発生しました: {e}")

    # 最終確認
    final_positive = sum(1 for sample in finetuning_samples
                         if 'はい' in sample['messages'][2]['content'])
    final_negative = len(finetuning_samples) - final_positive

    print("-" * 20)
    print(f"最終的なサンプル数: {len(finetuning_samples)} 件")
    print(f"最終データバランス: 正例={final_positive}件, 負例={final_negative}件")
    if final_negative > 0:
        print(f"バランス比率: {final_positive/final_negative:.2f} (理想は1.00)")

    if abs(final_positive - final_negative) <= 5:
        print("✅ バランス良好（差異5件以内）")
    else:
        print(f"⚠️  バランス偏り（差異{abs(final_positive - final_negative)}件）")
    print("-" * 20)

    # 指定されたパスにファインチューニング用データを保存
    try:
        with open(args.output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in finetuning_samples:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"ファインチューニング用データを {args.output_jsonl_path} に保存しました。")
    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました - "
              f"{args.output_jsonl_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="矛盾する三角形と判断が難しいペアからファインチューニング用データを生成します。"
    )
    parser.add_argument(
        "--inconsistent_triangles_csv", required=True,
        help="矛盾ペア情報のCSVファイルパス"
    )
    parser.add_argument(
        "--evaluation_details_csv", required=True,
        help="モデル評価詳細情報のCSVファイルパス"
    )
    parser.add_argument(
        "--ground_truth_yaml", required=True,
        help="正解データのYAMLファイルパス"
    )
    parser.add_argument(
        "--output_jsonl_path", required=True,
        help="出力するJSONLファイルのパス"
    )
    parser.add_argument(
        "--data_type",
        required=True,
        choices=["bib", "music", "person"],
        help="データの種類 (プロンプト生成に利用)"
    )
    parser.add_argument(
        "--score_column",
        required=True,
        help="Hard negative/positiveマイニングに使用するスコア列名"
    )

    args = parser.parse_args()
    main(args)
