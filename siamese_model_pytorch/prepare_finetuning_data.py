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


def get_record_details_for_finetuning_prompt(record_id, data_type):
    if not BIB_DATA:
        print("エラー: 書誌データがロードされていません。"
              "(get_record_details_for_finetuning_prompt)")
        return "情報取得エラー: BIB_DATA未ロード"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    if data_type == "music":
        title = bib_details.get("title", "Unknown")
        authors_str = bib_details.get("artist", "Unknown")
        publisher = bib_details.get("album", "Unknown")
        pubdate = bib_details.get("release_date", "Unknown")
        length = bib_details.get("length", "Unknown")
        return (f"Title: {title}\nArtist: {authors_str}\n"
                f"Album: {publisher}\nRelease Date: {pubdate}\nLength: {length}")
    elif data_type == "person":
        givenname = bib_details.get("givenname", "Unknown")
        surname = bib_details.get("surname", "Unknown")
        postcode = bib_details.get("postcode", "Unknown")
        suburb = bib_details.get("suburb", "Unknown")
        return (f"Given Name: {givenname}\nSurname: {surname}\nPostcode: {postcode}\nSuburb: {suburb}")
    elif data_type == "walmart_amazon_product":
        name = bib_details.get("title", "Unknown")
        brand = bib_details.get("brand", "Unknown")
        modelno = bib_details.get("modelno", "Unknown")
        price = bib_details.get("price", "Unknown")
        return (f"Product Name: {name}\nBrand: {brand}\nModel Number: {modelno}\nPrice: {price}")
    elif data_type == "wdc_product":
        name = bib_details.get("title", "Unknown")
        brand = bib_details.get("brand", "Unknown")
        description = bib_details.get("description", "Unknown")
        price = bib_details.get("price", "Unknown")
        return (f"Product Name: {name}\nBrand: {brand}\nDescription: {description}\nPrice: {price}")
    else:  # bib or default
        title = bib_details.get("bib1_title", "Unknown")
        authors_str = bib_details.get("bib1_author", "Unknown")
        publisher = bib_details.get("bib1_publisher", "Unknown")
        pubdate = bib_details.get("bib1_pubdate", "Unknown")
        return (f"Title: {title}\nAuthor: {authors_str}\n"
                f"Publisher: {publisher}\nPublication Date: {pubdate}")


def get_prompts(data_type):
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


def create_finetuning_message(record1_id, record2_id, is_truly_similar,
                              data_type, score=None):
    system_prompt = get_prompts(data_type)
    if data_type == "walmart_amazon_product":
        user_prompt = (
            f"Please determine whether the following two product records refer to essentially the same product.\\n\\n"
            f"Product 1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"Product 2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "Do these refer to the same product?\\nAnswer:"
        )
    elif data_type == "wdc_product":
        user_prompt = (
            f"Please determine whether the following two product records refer to essentially the same product.\\n\\n"
            f"Product 1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"Product 2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "Do these refer to the same product?\\nAnswer:"
        )
    elif data_type == "music":
        user_prompt = (
            f"Please determine whether the following two music records refer to essentially the same musical work.\\n\\n"
            f"Record 1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"Record 2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "Do these refer to the same work?\\nAnswer:"
        )
    elif data_type == "person":
        user_prompt = (
            f"Please determine whether the following two person records refer to essentially the same individual.\\n\\n"
            f"Record 1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"Record 2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "Do these refer to the same person?\\nAnswer:"
        )
    elif data_type == "bib":
        user_prompt = (
            f"Please determine whether the following two bibliographic records refer to essentially the same publication.\\n\\n"
            f"Record 1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"Record 2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "Do these refer to the same publication?\\nAnswer:"
        )
    if is_truly_similar:
        assistant_response = "Yes\\nConfidence Score: 1.0"
    else:
        assistant_response = "No\\nConfidence Score: 0.0"
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

    # 1. 矛盾する三角形のペアを追加（num_samplesで制限）
    try:
        inconsistent_df = pd.read_csv(args.inconsistent_triangles_csv)
        
        temp_samples = []

        if args.sampling_strategy == "inconsistency":
            print("サンプリング戦略: inconsistency (矛盾度スコアが高い順)")
            # 矛盾度の高い順にソート
            inconsistent_df = inconsistent_df.sort_values('inconsistency_score', ascending=False)
            
            for _, row in inconsistent_df.iterrows():
                # 3つのペアを処理
                pairs_data = [
                    {'id1': row['triangle_node1'], 'id2': row['triangle_node2'], 'is_similar': row['true_edge12'], 'score': row['p_edge12'], 'inconsistency_score': row['inconsistency_score']},
                    {'id1': row['triangle_node2'], 'id2': row['triangle_node3'], 'is_similar': row['true_edge23'], 'score': row['p_edge23'], 'inconsistency_score': row['inconsistency_score']},
                    {'id1': row['triangle_node1'], 'id2': row['triangle_node3'], 'is_similar': row['true_edge31'], 'score': row['p_edge31'], 'inconsistency_score': row['inconsistency_score']}
                ]
                for pair_data in pairs_data:
                    message = create_finetuning_message(
                        pair_data['id1'], pair_data['id2'],
                        pair_data['is_similar'], args.data_type,
                        pair_data['score']
                    )
                    message['inconsistency_score'] = pair_data['inconsistency_score']
                    temp_samples.append(message)
            
            # 矛盾度でソート（高い順）
            temp_samples.sort(key=lambda x: x['inconsistency_score'], reverse=True)

        elif args.sampling_strategy == "lowest_score":
            print("サンプリング戦略: lowest_score (類似度スコアが低い順)")
            for _, row in inconsistent_df.iterrows():
                pairs_data = [
                    {'id1': row['triangle_node1'], 'id2': row['triangle_node2'], 'is_similar': row['true_edge12'], 'score': row['p_edge12']},
                    {'id1': row['triangle_node2'], 'id2': row['triangle_node3'], 'is_similar': row['true_edge23'], 'score': row['p_edge23']},
                    {'id1': row['triangle_node1'], 'id2': row['triangle_node3'], 'is_similar': row['true_edge31'], 'score': row['p_edge31']}
                ]
                # 最も類似度が低いペアを選択
                lowest_score_pair = min(pairs_data, key=lambda x: x['score'])
                
                message = create_finetuning_message(
                    lowest_score_pair['id1'], lowest_score_pair['id2'],
                    lowest_score_pair['is_similar'], args.data_type,
                    lowest_score_pair['score']
                )
                message['score'] = lowest_score_pair['score'] # ソート用にスコアを保持
                temp_samples.append(message)

            # 類似度スコアでソート（低い順）
            temp_samples.sort(key=lambda x: x['score'])

        # 重複除去しながらnum_samplesまで追加
        unique_samples = []
        seen_pairs_set = set()
        for sample in temp_samples:
            user_content = sample['messages'][1]['content']
            if user_content not in seen_pairs_set:
                seen_pairs_set.add(user_content)
                # 一時的なスコアキーを削除
                if 'inconsistency_score' in sample:
                    del sample['inconsistency_score']
                if 'score' in sample:
                    del sample['score']
                
                unique_samples.append(sample)
                
                if len(unique_samples) >= args.num_samples:
                    break
        
        finetuning_samples = unique_samples

        print(
            f"候補ペア処理: {len(temp_samples)} 件の候補から "
            f"{len(finetuning_samples)} 件のユニークなペアを追加しました（上限: {args.num_samples}）。"
        )

    except FileNotFoundError:
        print(f"警告: 矛盾ペアファイルが見つかりません: {args.inconsistent_triangles_csv}")
    except Exception as e:
        print(f"警告: 矛盾ペアファイルの処理中にエラー: {e}")
        if 'inconsistent_df' in locals():
            print(f"  利用可能なカラム: {list(inconsistent_df.columns)}")
        else:
            print("  データフレームを読み込めませんでした")

    # 指定数に達していない場合は Hard sampling で補完
    current_total = len(finetuning_samples)
    if current_total < args.num_samples:
        try:
            needed_samples = args.num_samples - current_total
            print(f"目標数に達していないため Hard sampling で {needed_samples} 件追加します")
            
            # バランスを数え直す
            positive_count = sum(1 for sample in finetuning_samples
                                 if 'Yes' in sample['messages'][2]['content'])
            negative_count = len(finetuning_samples) - positive_count

            # 70:30バランスを目標に少数クラスを優先補完
            if positive_count < negative_count:
                # 正例が少ない場合：50:50を目指す
                target_positive = int(args.num_samples * 0.3)
                target_negative = args.num_samples - target_positive
                
                needed_positive = max(0, target_positive - positive_count)
                needed_negative = max(0, min(needed_samples - needed_positive, target_negative - negative_count))
                print(f"正例が少数クラスのため、バランス改善: 正例+{needed_positive}, 負例+{needed_negative}")
            else:
                # 負例が少ない場合：50:50を目指す
                target_positive = int(args.num_samples * 0.5)
                target_negative = args.num_samples - target_positive
                
                needed_negative = max(0, target_negative - negative_count)
                needed_positive = max(0, min(needed_samples - needed_negative, target_positive - positive_count))
                print(f"負例が少数クラスのため、バランス改善: 正例+{needed_positive}, 負例+{needed_negative}")

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
                    # seen_pairsは全サンプルで共有するため、ここでは使わない
                    # if user_content not in seen_pairs:
                    finetuning_samples.append(message)
                    #     seen_pairs.add(user_content)
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
                    # if user_content not in seen_pairs:
                    finetuning_samples.append(message)
                    #     seen_pairs.add(user_content)
                    added_negative += 1

            print(f"Hard ペア追加完了: 正例={added_positive}件, "
                  f"負例={added_negative}件")

        except FileNotFoundError:
            print(f"警告: 評価詳細ファイルが見つかりません: "
                  f"{args.evaluation_details_csv}。Hard Samplingは行われません。")
        except Exception as e:
            print(f"警告: Hard Sampling 中にエラーが発生しました: {e}")

    # 2. バランス調整（必要に応じて削除・追加）
    # 現在の正例・負例を数える
    positive_count = sum(1 for sample in finetuning_samples
                         if 'Yes' in sample['messages'][2]['content'])
    negative_count = len(finetuning_samples) - positive_count

    print(f"初期サンプリングの内訳: 正例={positive_count}件, 負例={negative_count}件")

    # バランスが大きく偏っている場合は調整
    imbalance_threshold = args.num_samples * 0.3  # 30%以上偏っている場合
    current_total = len(finetuning_samples)
    
    if (positive_count - negative_count) > imbalance_threshold and current_total >= args.num_samples:
        print(f"バランスが偏っているため調整します（閾値: {imbalance_threshold}）")
        
        # 70:30程度のバランスに調整（情報損失を最小限に）
        if positive_count > negative_count:
            # 正例が多い場合：50:50を目標
            target_positive = int(args.num_samples * 0.5)
            target_negative = args.num_samples - target_positive
            
            positive_samples = [s for s in finetuning_samples if 'Yes' in s['messages'][2]['content']]
            negative_samples = [s for s in finetuning_samples if 'No' in s['messages'][2]['content']]
            
            # 削減は最小限に
            actual_positive = min(target_positive, positive_count)
            actual_negative = min(target_negative, negative_count)
            
            finetuning_samples = positive_samples[:actual_positive] + negative_samples[:actual_negative]
        else:
            # 負例が多い場合：30:70を目標
            target_positive = int(args.num_samples * 0.3)
            target_negative = args.num_samples - target_positive
            
            positive_samples = [s for s in finetuning_samples if 'Yes' in s['messages'][2]['content']]
            negative_samples = [s for s in finetuning_samples if 'No' in s['messages'][2]['content']]
            
            # 削減は最小限に
            actual_positive = min(target_positive, positive_count)
            actual_negative = min(target_negative, negative_count)
            
            finetuning_samples = positive_samples[:actual_positive] + negative_samples[:actual_negative]
        
        # 再カウント
        positive_count = sum(1 for sample in finetuning_samples
                           if 'Yes' in sample['messages'][2]['content'])
        negative_count = len(finetuning_samples) - positive_count
        print(f"削減後の内訳: 正例={positive_count}件, 負例={negative_count}件")

    try:
        # 指定数に達していない場合は Hard sampling で補完
        current_total = len(finetuning_samples)
        if current_total < args.num_samples:
            needed_samples = args.num_samples - current_total
            print(f"目標数に達していないため Hard sampling で {needed_samples} 件追加します")
            
            # 70:30バランスを目標に少数クラスを優先補完
            if positive_count < negative_count:
                # 正例が少ない場合：30:70 → より良いバランスを目指す
                target_positive = int(args.num_samples * 0.5)
                target_negative = args.num_samples - target_positive
                
                needed_positive = max(0, target_positive - positive_count)
                needed_negative = max(0, min(needed_samples - needed_positive, target_negative - negative_count))
                print(f"正例が少数クラスのため、バランス改善: 正例+{needed_positive}, 負例+{needed_negative}")
            else:
                # 負例が少ない場合：70:30 → より良いバランスを目指す
                target_positive = int(args.num_samples * 0.5)
                target_negative = args.num_samples - target_positive
                
                needed_negative = max(0, target_negative - negative_count)
                needed_positive = max(0, min(needed_samples - needed_negative, target_positive - positive_count))
                print(f"負例が少数クラスのため、バランス改善: 正例+{needed_positive}, 負例+{needed_negative}")

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

    # 最終的なサンプル数制限
    if len(finetuning_samples) > args.num_samples:
        print(f"サンプル数が上限（{args.num_samples}）を超過しているため切り詰めます")
        finetuning_samples = finetuning_samples[:args.num_samples]

    # 最終確認
    final_positive = sum(1 for sample in finetuning_samples
                         if 'Yes' in sample['messages'][2]['content'])
    final_negative = len(finetuning_samples) - final_positive

    print("-" * 20)
    print(f"最終的なサンプル数: {len(finetuning_samples)} 件（上限: {args.num_samples}）")
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
        choices=["bib", "music", "person", "walmart_amazon_product", "wdc_product"],
        help="データの種類 (プロンプト生成に利用)"
    )
    parser.add_argument(
        "--score_column",
        required=True,
        help="Hard negative/positiveマイニングに使用するスコア列名"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="生成するサンプル数の上限（デフォルト: 100）"
    )
    parser.add_argument(
        "--sampling_strategy", type=str, default="inconsistency",
        choices=["inconsistency", "lowest_score"],
        help="サンプリング戦略を選択 (デフォルト: inconsistency)"
    )

    args = parser.parse_args()
    main(args)
