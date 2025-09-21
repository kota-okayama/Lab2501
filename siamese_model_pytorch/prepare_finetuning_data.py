import json
import os
import yaml
import sys
import argparse
import pandas as pd
import random # Added for random sampling

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
        ],
        "record_id_1": str(record1_id),
        "record_id_2": str(record2_id)
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


    finetuning_samples = []
    # この実行内で追加されたペアを追跡 (IDベース)
    seen_id_pairs = set()

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
            print("サンプリング戦略: lowest_score (スコアが0.5に最も近い順)")
            for _, row in inconsistent_df.iterrows():
                pairs_data = [
                    {'id1': row['triangle_node1'], 'id2': row['triangle_node2'], 'is_similar': row['true_edge12'], 'score': row['p_edge12']},
                    {'id1': row['triangle_node2'], 'id2': row['triangle_node3'], 'is_similar': row['true_edge23'], 'score': row['p_edge23']},
                    {'id1': row['triangle_node1'], 'id2': row['triangle_node3'], 'is_similar': row['true_edge31'], 'score': row['p_edge31']}
                ]
                # スコアが0.5に最も近いペアを選択
                lowest_score_pair = min(pairs_data, key=lambda x: abs(x['score'] - 0.5))
                
                message = create_finetuning_message(
                    lowest_score_pair['id1'], lowest_score_pair['id2'],
                    lowest_score_pair['is_similar'], args.data_type,
                    lowest_score_pair['score']
                )
                # ソート用に不確実性スコアを保持 (0.5に近いほど小さい)
                message['uncertainty'] = abs(lowest_score_pair['score'] - 0.5)
                temp_samples.append(message)

            # 不確実性スコアでソート（小さい順）
            temp_samples.sort(key=lambda x: x['uncertainty'])

        # IDベースで重複除去しながらnum_samplesまで追加
        for sample in temp_samples:
            pair_tuple = tuple(sorted((sample['record_id_1'], sample['record_id_2'])))
            if pair_tuple in labeled_pairs or pair_tuple in seen_id_pairs:
                continue

            # 一時的なスコアキーを削除
            if 'inconsistency_score' in sample:
                del sample['inconsistency_score']
            if 'score' in sample:
                del sample['score']
            if 'uncertainty' in sample:
                del sample['uncertainty']
            
            finetuning_samples.append(sample)
            seen_id_pairs.add(pair_tuple)
            
            if len(finetuning_samples) >= args.num_samples:
                break
        
        print(
            f"候補ペア処理: {len(temp_samples)} 件の候補から "
            f"{len(finetuning_samples)} 件のユニークなペアを追加しました（上限: {args.num_samples}）。"
        )
        initial_positive = sum(1 for s in finetuning_samples if 'Yes' in s['messages'][2]['content'])
        initial_negative = len(finetuning_samples) - initial_positive
        print(f"初期サンプリング時のバランス: 正例={initial_positive}件, 負例={initial_negative}件")

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
            
            details_df = pd.read_csv(args.evaluation_details_csv)
            # Hard Samplingをスコア順ではなくランダムにする
            hard_pairs_df = details_df.sample(frac=1, random_state=42)

            # バランスを数え直す
            positive_count = sum(1 for sample in finetuning_samples
                                 if 'Yes' in sample['messages'][2]['content'])
            negative_count = len(finetuning_samples) - positive_count

            # 50:50バランスを目標に補完
            target_positive = args.num_samples // 2
            target_negative = args.num_samples - target_positive
            
            needed_positive = max(0, target_positive - positive_count)
            needed_negative = max(0, target_negative - negative_count)
            print(f"バランス目標: 正例={target_positive}, 負例={target_negative}")
            print(f"不足分をランダムに補完: 正例+{needed_positive}, 負例+{needed_negative}")

            hard_positive_df = hard_pairs_df[hard_pairs_df['ground_truth_similar']]
            hard_negative_df = hard_pairs_df[~hard_pairs_df['ground_truth_similar']]

            added_positive = 0
            if needed_positive > 0:
                for _, row in hard_positive_df.iterrows():
                    if added_positive >= needed_positive:
                        break

                    pair_tuple = tuple(sorted((str(row['record_id_1']), str(row['record_id_2']))))
                    if pair_tuple in labeled_pairs or pair_tuple in seen_id_pairs:
                        continue

                    message = create_finetuning_message(
                        row['record_id_1'], row['record_id_2'],
                        row['ground_truth_similar'], args.data_type,
                        row[args.score_column]
                    )
                    finetuning_samples.append(message)
                    seen_id_pairs.add(pair_tuple)
                    added_positive += 1

            added_negative = 0
            if needed_negative > 0:
                for _, row in hard_negative_df.iterrows():
                    if added_negative >= needed_negative:
                        break
                    
                    pair_tuple = tuple(sorted((str(row['record_id_1']), str(row['record_id_2']))))
                    if pair_tuple in labeled_pairs or pair_tuple in seen_id_pairs:
                        continue
                        
                    message = create_finetuning_message(
                        row['record_id_1'], row['record_id_2'],
                        row['ground_truth_similar'], args.data_type,
                        row[args.score_column]
                    )
                    finetuning_samples.append(message)
                    seen_id_pairs.add(pair_tuple)
                    added_negative += 1

            print(f"Hard ペア追加完了: 正例={added_positive}件, "
                  f"負例={added_negative}件")

        except FileNotFoundError:
            print(f"警告: 評価詳細ファイルが見つかりません: "
                  f"{args.evaluation_details_csv}。Hard Samplingは行われません。")
        except Exception as e:
            print(f"警告: Hard Sampling 中にエラーが発生しました: {e}")

    # (追加) バランス調整前のデータを保存
    if args.output_jsonl_path_unbalanced:
        print(f"\nバランス調整前のデータを {args.output_jsonl_path_unbalanced} に保存します...")
        try:
            with open(args.output_jsonl_path_unbalanced, 'w', encoding='utf-8') as f:
                for entry in finetuning_samples:
                    openai_entry = {"messages": entry["messages"]}
                    f.write(json.dumps(openai_entry, ensure_ascii=False) + '\n')
            print(f"保存完了: {len(finetuning_samples)} 件")
        except IOError as e:
            print(f"エラー: バランス調整前データの書き込みに失敗: {e}")


    # 2. 最終的なデータバランス調整 (50:50目標)
    print("\n最終的なデータバランス調整を行います...")
    positive_samples = [s for s in finetuning_samples if 'Yes' in s['messages'][2]['content']]
    negative_samples = [s for s in finetuning_samples if 'No' in s['messages'][2]['content']]
    
    target_count = args.num_samples // 2
    
    # 過剰なサンプルをランダムに削除
    if len(positive_samples) > target_count:
        positive_samples = random.sample(positive_samples, target_count)
    if len(negative_samples) > target_count:
        negative_samples = random.sample(negative_samples, target_count)
        
    finetuning_samples = positive_samples + negative_samples

    # それでも不足している場合は、さらにランダムに追加
    current_positive = len(positive_samples)
    current_negative = len(negative_samples)
    
    needed_positive = target_count - current_positive
    needed_negative = (args.num_samples - target_count) - current_negative

    if needed_positive > 0 or needed_negative > 0:
        print(f"バランス調整のためさらにペアを追加: 正例+{needed_positive}, 負例+{needed_negative}")
        try:
            if 'details_df' not in locals():
                details_df = pd.read_csv(args.evaluation_details_csv)
            
            # まだ使われていないペアを候補にする
            all_current_pairs = labeled_pairs.union(seen_id_pairs)
            details_df['pair_tuple'] = details_df.apply(lambda row: tuple(sorted((str(row['record_id_1']), str(row['record_id_2'])))), axis=1)
            candidate_df = details_df[~details_df['pair_tuple'].isin(all_current_pairs)]

            positive_candidates = candidate_df[candidate_df['ground_truth_similar']]
            negative_candidates = candidate_df[~candidate_df['ground_truth_similar']]

            # 正例を追加
            if needed_positive > 0 and not positive_candidates.empty:
                num_to_add = min(needed_positive, len(positive_candidates))
                added_pos_df = positive_candidates.sample(n=num_to_add)
                for _, row in added_pos_df.iterrows():
                    message = create_finetuning_message(row['record_id_1'], row['record_id_2'], True, args.data_type)
                    finetuning_samples.append(message)

            # 負例を追加
            if needed_negative > 0 and not negative_candidates.empty:
                num_to_add = min(needed_negative, len(negative_candidates))
                added_neg_df = negative_candidates.sample(n=num_to_add)
                for _, row in added_neg_df.iterrows():
                    message = create_finetuning_message(row['record_id_1'], row['record_id_2'], False, args.data_type)
                    finetuning_samples.append(message)

        except FileNotFoundError:
            print(f"警告: 評価詳細ファイルが見つかりません: {args.evaluation_details_csv}。追加のバランス調整はスキップされます。")
        except Exception as e:
            print(f"警告: バランス調整中の追加サンプリングでエラー: {e}")


    # 最終的なサンプル数制限
    if len(finetuning_samples) > args.num_samples:
        print(f"サンプル数が上限（{args.num_samples}）を超過しているためランダムに切り詰めます")
        finetuning_samples = random.sample(finetuning_samples, args.num_samples)

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
                # OpenAI APIは 'messages' キーのみを要求するため、それ以外は除外
                openai_entry = {"messages": entry["messages"]}
                f.write(json.dumps(openai_entry, ensure_ascii=False) + '\n')
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
        "--output_jsonl_path_unbalanced",
        default=None,
        help="[任意] バランス調整前のデータを出力するJSONLファイルのパス"
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
    parser.add_argument(
        '--labeled_pairs_csv',
        type=str,
        default=None,
        help='過去にラベル付けされたペアのCSVファイルパス。これらのペアはサンプリングから除外されます。'
    )

    args = parser.parse_args()
    main(args)
