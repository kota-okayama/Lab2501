import csv
import json
import os
import yaml
import sys  # sys.exitのため追加
import random  # Add this import

# --- グローバル設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_RESULTS_FILENAME = "human_review_simulation_accuracy_sample2000_100.csv"
SIMULATION_RESULTS_PATH = os.path.join(BASE_DIR, SIMULATION_RESULTS_FILENAME)

PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024"
RECORD_YAML_FILENAME = "sampled_data_2000.yml"
RECORD_YAML_PATH = os.path.join(PROJECT_ROOT_ASSUMED, BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, RECORD_YAML_FILENAME)

OUTPUT_JSONL_FILENAME = "finetuning_data_with_llm_score.jsonl"
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, OUTPUT_JSONL_FILENAME)

# グローバル変数として書誌データを保持
BIB_DATA = {}
RECORD_ID_TO_CLUSTER_ID = {}  # Add this global variable


# --- 書誌データ読み込み関連関数 (evaluate_pairs_with_openai_async.py から拝借・調整) ---
def load_bib_data_for_finetuning(yaml_path):
    global BIB_DATA, RECORD_ID_TO_CLUSTER_ID  # Add RECORD_ID_TO_CLUSTER_ID to global
    BIB_DATA = {}
    RECORD_ID_TO_CLUSTER_ID = {}  # Initialize
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
                ) in (
                    possible_records_dict.items()
                ):  # value_listのtypo修正 value -> This comment seems to refer to a previous state; value_list is correct here.
                    if key in ["version", "type", "id", "summary", "inf_attr"] and possible_records_dict is all_data:
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            cluster_id_val = None
                            actual_bib_data = {}

                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])
                                cluster_id_val = record.get("cluster_id")

                                if "data" in record and isinstance(record["data"], dict):
                                    actual_bib_data = record["data"]
                                else:
                                    actual_bib_data = {
                                        k_rec: v_rec
                                        for k_rec, v_rec in record.items()
                                        if k_rec not in ["id", "cluster_id"]
                                    }

                                if record_id_str and actual_bib_data:
                                    if record_id_str not in processed_record_ids_for_bib_data:
                                        BIB_DATA[record_id_str] = actual_bib_data
                                        processed_record_ids_for_bib_data.add(record_id_str)

                                    if cluster_id_val is not None:  # cluster_id could be 0, so check for None
                                        # Add to cluster_id map. If ID appears multiple times with different cluster_ids, this will take the last one.
                                        # This assumes cluster_id is consistent if id appears multiple times in the YAML under different categories.
                                        RECORD_ID_TO_CLUSTER_ID[record_id_str] = cluster_id_val
                                        processed_record_ids_for_cluster_map.add(
                                            record_id_str
                                        )  # Keep track of which IDs had a cluster_id

                                elif record_id_str and not actual_bib_data:
                                    print(
                                        f"警告: レコードID {record_id_str} に有効な書誌データが見つかりませんでした。BIB_DATAへの登録をスキップします。"
                                    )

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。YAMLの構造を確認してください。")
            sys.exit(1)
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        print(f"{len(RECORD_ID_TO_CLUSTER_ID)} 件の record_id と cluster_id のマッピングをロードしました。")
        if not RECORD_ID_TO_CLUSTER_ID:
            print(
                f"警告: {yaml_path} から cluster_id を含むレコードが見つからなかったか、マッピングの作成に失敗しました。ランダム非一致ペアの生成が困難または不可能になります。"
            )

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def get_record_details_for_finetuning_prompt(record_id):
    if not BIB_DATA:
        print("エラー: 書誌データがロードされていません。(get_record_details_for_finetuning_prompt)")
        # この関数が呼ばれる時点ではBIB_DATAはロードされているはずなので、基本的にはここに来ない想定
        return "情報取得エラー: BIB_DATA未ロード"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    title = bib_details.get("bib1_title", "タイトル不明")
    authors_str = bib_details.get("bib1_author", "著者不明")
    publisher = bib_details.get("bib1_publisher", "出版社不明")
    pubdate = bib_details.get("bib1_pubdate", "出版日不明")
    return f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"


# --- メイン処理 ---
def main():
    print("ファインチューニング用データ作成処理を開始します...")

    load_bib_data_for_finetuning(RECORD_YAML_PATH)  # BIB_DATAをグローバルにロード
    # BIB_DATAがロード失敗した場合は load_bib_data_for_finetuning 内で exit する

    finetuning_samples = []
    existing_pairs_from_csv = set()  # For de-duplication with random pairs

    if not os.path.exists(SIMULATION_RESULTS_PATH):
        print(f"エラー: シミュレーション結果ファイルが見つかりません: {SIMULATION_RESULTS_PATH}")
        # return # If CSV is mandatory, return. If optional and we only add random pairs, then continue.
        # For now, let's assume CSV provides some samples.
    else:
        try:
            with open(SIMULATION_RESULTS_PATH, "r", newline="", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                if not reader.fieldnames:  # ヘッダーがない場合（空ファイルなど）
                    print(f"エラー: {SIMULATION_RESULTS_PATH} からヘッダーが読み取れませんでした。")
                    # return
                else:
                    for row in reader:
                        record_id_1 = row.get("record_id_1")
                        record_id_2 = row.get("record_id_2")
                        ground_truth_label_str = row.get("ground_truth_label", "").strip().lower()
                        # llm_score_str = row.get("original_llm_score") # original_llm_score is not used for assistant response generation

                        if not record_id_1 or not record_id_2:
                            print(f"警告: record_idが不足している行があります: {row}。スキップします。")
                            continue

                        # Add to existing_pairs_from_csv for de-duplication with random pairs later
                        pair_key = tuple(sorted((str(record_id_1), str(record_id_2))))
                        existing_pairs_from_csv.add(pair_key)

                        if ground_truth_label_str not in ["true", "false"]:
                            print(
                                f"警告: ペア ({record_id_1}, {record_id_2}) の Ground Truth が不正です ('{ground_truth_label_str}')。スキップします。"
                            )
                            continue

                        # original_llm_score processing seems to be for analysis, not for generating training data.
                        # We will generate assistant response based on ground_truth_label.
                        # llm_similarity_score = 0.0
                        # (The block for processing llm_score_str is removed as it's not directly used for the assistant's response here)

                        ground_truth_is_similar = ground_truth_label_str == "true"

                        bib_info_1 = get_record_details_for_finetuning_prompt(record_id_1)
                        bib_info_2 = get_record_details_for_finetuning_prompt(record_id_2)

                        if (
                            "情報取得エラー" in bib_info_1
                            or "書誌情報なし" in bib_info_1
                            or "情報取得エラー" in bib_info_2
                            or "書誌情報なし" in bib_info_2
                        ):
                            print(
                                f"警告: ペア ({record_id_1}, {record_id_2}) の書誌情報取得に失敗。スキップします。詳細1: {bib_info_1}, 詳細2: {bib_info_2}"
                            )
                            continue

                        system_prompt = (
                            "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\\n"
                            "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
                            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
                        )

                        user_prompt = (
                            f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\\n\\n"
                            f"書誌情報1:\\n{bib_info_1}\\n\\n"
                            f"書誌情報2:\\n{bib_info_2}\\n\\n"
                            "これらは同一の文献ですか？\\n回答:"
                        )

                        if ground_truth_is_similar:
                            assistant_response = "はい\\n類似度スコア: 1.0"
                        else:
                            assistant_response = "いいえ\\n類似度スコア: 0.0"

                        finetuning_samples.append(
                            {
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": assistant_response},
                                ]
                            }
                        )
        except Exception as e:
            print(f"シミュレーション結果ファイル ({SIMULATION_RESULTS_PATH}) の処理中にエラー: {e}")
            import traceback

            traceback.print_exc()
            # return # Decide if to proceed if CSV processing fails

    # --- 追加: ランダムな非一致ペアの生成 ---
    if RECORD_ID_TO_CLUSTER_ID and len(RECORD_ID_TO_CLUSTER_ID) >= 2:
        print("ランダムな非一致ペアの追加処理を開始します...")
        num_csv_samples = len(finetuning_samples)
        # Aim to add a similar number of random non-matching pairs, or a fixed cap
        num_target_random_pairs = (
            num_csv_samples if num_csv_samples > 0 else 1000
        )  # Add at least some if CSV was empty/failed

        all_record_ids_with_cluster = list(RECORD_ID_TO_CLUSTER_ID.keys())

        generated_random_pairs_set = set()  # Tracks (sorted_id1, sorted_id2) tuples for random pairs
        added_random_pairs_count = 0

        # Adjust max_attempts based on the number of available records and target pairs
        # If many records, collisions are less likely. If few, more attempts might be needed.
        max_attempts = num_target_random_pairs * 20  # Increased multiplier for more robust sampling

        print(f"目標とするランダム非一致ペア数: {num_target_random_pairs}")
        print(f"クラスタIDを持つレコード数: {len(all_record_ids_with_cluster)}")

        if len(all_record_ids_with_cluster) < 2:
            print("警告: ランダムペア生成のための十分なユニークレコード数（cluster_id持ち）がありません。")
        else:
            for attempt in range(max_attempts):
                if added_random_pairs_count >= num_target_random_pairs:
                    break

                record_id_1, record_id_2 = random.sample(all_record_ids_with_cluster, 2)

                # Canonical pair representation (sorted tuple of strings)
                current_pair_sorted = tuple(sorted((str(record_id_1), str(record_id_2))))

                # Check for various forms of duplication or invalidity
                if record_id_1 == record_id_2:  # Should not happen with random.sample(..., 2)
                    continue
                if current_pair_sorted in generated_random_pairs_set:  # Already generated this random pair
                    continue
                if current_pair_sorted in existing_pairs_from_csv:  # Pair was in the input CSV
                    continue

                cluster_1 = RECORD_ID_TO_CLUSTER_ID.get(record_id_1)
                cluster_2 = RECORD_ID_TO_CLUSTER_ID.get(record_id_2)

                # Ensure they are from different clusters and both cluster_ids are valid
                if cluster_1 is not None and cluster_2 is not None and cluster_1 != cluster_2:
                    bib_info_1 = get_record_details_for_finetuning_prompt(record_id_1)
                    bib_info_2 = get_record_details_for_finetuning_prompt(record_id_2)

                    if (
                        "情報取得エラー" in bib_info_1
                        or "書誌情報なし" in bib_info_1
                        or "情報取得エラー" in bib_info_2
                        or "書誌情報なし" in bib_info_2
                    ):
                        print(
                            f"警告: ランダムペア ({record_id_1}, {record_id_2}) の書誌情報取得に失敗。スキップします。"
                        )
                        continue

                    system_prompt = (
                        "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\\n"
                        "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
                        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
                    )
                    user_prompt = (
                        f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\\n\\n"
                        f"書誌情報1:\\n{bib_info_1}\\n\\n"
                        f"書誌情報2:\\n{bib_info_2}\\n\\n"
                        "これらは同一の文献ですか？\\n回答:"
                    )
                    assistant_response = "いいえ\\n類似度スコア: 0.0"  # For non-matching pairs

                    finetuning_samples.append(
                        {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": assistant_response},
                            ]
                        }
                    )
                    generated_random_pairs_set.add(current_pair_sorted)
                    added_random_pairs_count += 1

            if added_random_pairs_count < num_target_random_pairs and attempt == max_attempts - 1:
                print(
                    f"情報: ランダム非一致ペアの目標数 {num_target_random_pairs} に対し、{added_random_pairs_count} 件生成しました。試行回数の上限 ({max_attempts}) に達したか、利用可能な非重複・非一致ペアが少なかった可能性があります。"
                )
            else:
                print(
                    f"{added_random_pairs_count} 件のランダムな非一致ペアをファインチューニング用サンプルに追加しました。"
                )

    elif not RECORD_ID_TO_CLUSTER_ID or len(RECORD_ID_TO_CLUSTER_ID) < 2:
        print(
            "RECORD_ID_TO_CLUSTER_ID が空またはレコード数が2未満のため、ランダムな非一致ペアの追加はスキップされました。"
        )

    if not finetuning_samples:
        print("ファインチューニング対象のサンプルが0件でした。処理を終了します。")
        return

    print(f"{len(finetuning_samples)} 件のファインチューニング用サンプルを作成しました。")

    try:
        with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as outfile:
            for sample in finetuning_samples:
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"ファインチューニング用データを {OUTPUT_JSONL_PATH} に保存しました。")
    except Exception as e:
        print(f"エラー: JSONLファイル書き込み中にエラー: {e}")


if __name__ == "__main__":
    main()
