"""
ファインチューニングデータ作成スクリプト

多様性サンプリング、不確実性サンプリング、ランダムサンプリングの
3つの戦略に基づいて、ファインチューニング用のデータセットを生成する。
"""
import argparse
import json
import os
import sys
import random
import itertools
import pandas as pd
import yaml

# --- グローバル変数 ---
BIB_DATA = {}
RECORD_ID_TO_CLUSTER_ID = {}


# --- データ読み込みとプロンプト生成関数 (prepare_finetuning_data.py から流用・調整) ---

def load_bib_data_and_gt_clusters(yaml_path):
    """
    正解データ(GT)のYAMLファイルから書誌情報とクラスタ情報をロードする。
    """
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
            # 'records' キーがある場合はそれを、なければ全体を辞書と見なす
            possible_records_dict = all_data.get("records", all_data)

            if isinstance(possible_records_dict, dict):
                for value_list in possible_records_dict.values():
                    if isinstance(value_list, list):
                        for record in value_list:
                            if isinstance(record, dict) and "id" in record:
                                record_id = str(record["id"])
                                cluster_id = record.get("cluster_id")
                                actual_data = record.get("data", record)

                                # 'id' と 'cluster_id' をデータ部から除く
                                actual_data.pop("id", None)
                                actual_data.pop("cluster_id", None)

                                BIB_DATA[record_id] = actual_data
                                if cluster_id is not None:
                                    RECORD_ID_TO_CLUSTER_ID[record_id] = cluster_id
        
        if not BIB_DATA:
            raise ValueError(f"{yaml_path} から書誌データがロードできませんでした。")
        
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        print(f"{len(RECORD_ID_TO_CLUSTER_ID)} 件の Ground Truth クラスタマッピングをロードしました。")

    except Exception as e:
        print(f"エラー: 正解データファイル({yaml_path})の処理中にエラー: {e}")
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


# --- サンプリング戦略ごとの関数 ---

def sample_uncertainty(args):
    """不確実性サンプリング（バランス調整付き）"""
    print("不確実性サンプリングを開始します...")
    try:
        df = pd.read_csv(args.evaluation_details_csv)
    except FileNotFoundError:
        print(f"エラー: 評価詳細ファイルが見つかりません: {args.evaluation_details_csv}")
        sys.exit(1)

    # 不確実性でソート（0.5に近いほど不確実）
    df['uncertainty'] = (df[args.score_column] - 0.5).abs()
    df_sorted = df.sort_values(by='uncertainty')
    
    # 初期サンプリング
    sampled_df = df_sorted.head(args.num_samples)
    
    # バランスをチェック
    positive_count = sum(sampled_df['ground_truth_similar'])
    negative_count = len(sampled_df) - positive_count
    
    print(f"初期サンプリング結果: 正例={positive_count}件, 負例={negative_count}件")
    
    # バランスが大きく偏っている場合は調整
    imbalance_threshold = args.num_samples * 0.3  # 30%以上偏っている場合
    
    if abs(positive_count - negative_count) > imbalance_threshold:
        print(f"バランスが偏っているため調整します（閾値: {imbalance_threshold}）")
        
        # 70:30程度のバランスに調整（情報損失を最小限に）
        if positive_count < negative_count:
            # 正例が少ない場合：30:70を目標
            positive_df = df_sorted[df_sorted['ground_truth_similar'] == True]
            negative_df = df_sorted[df_sorted['ground_truth_similar'] == False]
            
            target_positive = int(args.num_samples * 0.3)
            target_negative = args.num_samples - target_positive
            
            # 削減は最小限に
            actual_positive = min(target_positive, len(positive_df))
            actual_negative = min(target_negative, len(negative_df))
            
            sampled_df = pd.concat([
                positive_df.head(actual_positive),
                negative_df.head(actual_negative)
            ]).sort_values(by='uncertainty')
            
        else:
            # 負例が少ない場合：70:30を目標
            positive_df = df_sorted[df_sorted['ground_truth_similar'] == True]
            negative_df = df_sorted[df_sorted['ground_truth_similar'] == False]
            
            target_positive = int(args.num_samples * 0.7)
            target_negative = args.num_samples - target_positive
            
            # 削減は最小限に
            actual_positive = min(target_positive, len(positive_df))
            actual_negative = min(target_negative, len(negative_df))
            
            sampled_df = pd.concat([
                positive_df.head(actual_positive),
                negative_df.head(actual_negative)
            ]).sort_values(by='uncertainty')
        
        # 再カウント
        positive_count = sum(sampled_df['ground_truth_similar'])
        negative_count = len(sampled_df) - positive_count
        print(f"調整後の内訳: 正例={positive_count}件, 負例={negative_count}件")
    
    pairs = []
    for _, row in sampled_df.iterrows():
        pairs.append((str(row['record_id_1']), str(row['record_id_2'])))
        
    print(f"{len(pairs)} 件のペアをサンプリングしました。")
    return pairs

def sample_diversity(args):
    """多様性サンプリング"""
    print("多様性サンプリングを開始します...")
    try:
        with open(args.llm_clusters_json, 'r', encoding='utf-8') as f:
            llm_clusters = json.load(f)
    except FileNotFoundError:
        print(f"エラー: LLMクラスタファイルが見つかりません: {args.llm_clusters_json}")
        sys.exit(1)

    positive_pairs = []
    for cluster_id, records in llm_clusters.items():
        if len(records) > 1:
            record_ids = [str(r['record_id']) for r in records]
            for pair in itertools.combinations(record_ids, 2):
                positive_pairs.append(tuple(sorted(pair)))

    cluster_ids = list(llm_clusters.keys())
    negative_pairs = []
    if len(cluster_ids) > 1:
        # 異なるクラスタ間のペアを生成する
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster1_records = [str(r['record_id']) for r in llm_clusters[cluster_ids[i]]]
                cluster2_records = [str(r['record_id']) for r in llm_clusters[cluster_ids[j]]]
                for r1 in cluster1_records:
                    for r2 in cluster2_records:
                        negative_pairs.append(tuple(sorted((r1, r2))))

    # サンプル数の半分ずつを目標にランダムサンプリング
    num_positive = args.num_samples // 2
    num_negative = args.num_samples - num_positive

    sampled_pos = random.sample(positive_pairs, min(num_positive, len(positive_pairs)))
    sampled_neg = random.sample(negative_pairs, min(num_negative, len(negative_pairs)))
    
    pairs = list(set(sampled_pos + sampled_neg))
    print(f"正例候補: {len(positive_pairs)}件, 負例候補: {len(negative_pairs)}件")
    print(f"サンプリング結果: 正例 {len(sampled_pos)}件, 負例 {len(sampled_neg)}件 -> 合計 {len(pairs)} 件")
    return pairs

def sample_random(args):
    """ランダムサンプリング"""
    print("ランダムサンプリングを開始します...")
    try:
        df = pd.read_csv(args.evaluation_details_csv)
    except FileNotFoundError:
        print(f"エラー: 評価詳細ファイルが見つかりません: {args.evaluation_details_csv}")
        sys.exit(1)

    sampled_df = df.sample(n=args.num_samples, random_state=42)
    
    pairs = []
    for _, row in sampled_df.iterrows():
        pairs.append((str(row['record_id_1']), str(row['record_id_2'])))
        
    print(f"{len(pairs)} 件のペアをサンプリングしました。")
    return pairs


# --- メイン処理 ---

def main():
    parser = argparse.ArgumentParser(description="各種サンプリング戦略に基づき、ファインチューニングデータを生成します。")
    parser.add_argument("--strategy", required=True, choices=["uncertainty", "diversity", "random"], help="サンプリング戦略")
    parser.add_argument("--output_jsonl_path", required=True, help="出力するJSONLファイルのパス")
    parser.add_argument("--ground_truth_yaml", required=True, help="正解データのYAMLファイルパス")
    parser.add_argument("--num_samples", type=int, required=True, help="生成するサンプル（ペア）の総数")
    parser.add_argument("--data_type", required=True, help="データの種類 (例: bib)")
    
    # 戦略によって必須となる引数
    parser.add_argument("--evaluation_details_csv", help="[uncertainty, random] ペア候補のCSV")
    parser.add_argument("--score_column", help="[uncertainty] スコア列名")
    parser.add_argument("--llm_clusters_json", help="[diversity] LLMによるクラスタJSONファイル")

    args = parser.parse_args()

    # 引数のバリデーション
    if args.strategy in ["uncertainty", "random"] and not args.evaluation_details_csv:
        parser.error("--evaluation_details_csv is required for uncertainty and random strategies.")
    if args.strategy == "uncertainty" and not args.score_column:
        parser.error("--score_column is required for uncertainty strategy.")
    if args.strategy == "diversity" and not args.llm_clusters_json:
        parser.error("--llm_clusters_json is required for diversity strategy.")

    # 1. 正解データをロード
    load_bib_data_and_gt_clusters(args.ground_truth_yaml)

    # 2. 戦略に応じてペアをサンプリング
    strategy_func = {
        "uncertainty": sample_uncertainty,
        "diversity": sample_diversity,
        "random": sample_random,
    }[args.strategy]
    sampled_pairs = strategy_func(args)
    
    # 3. サンプルをJSONL形式に変換
    finetuning_samples = []
    for id1, id2 in sampled_pairs:
        # 正解ラベルを判定
        gt_cluster1 = RECORD_ID_TO_CLUSTER_ID.get(id1)
        gt_cluster2 = RECORD_ID_TO_CLUSTER_ID.get(id2)
        is_truly_similar = (gt_cluster1 is not None and gt_cluster1 == gt_cluster2)
        
        message = create_finetuning_message(id1, id2, is_truly_similar, args.data_type)
        finetuning_samples.append(message)
        
    # 4. ファイルに保存
    try:
        with open(args.output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in finetuning_samples:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print("-" * 20)
        print(f"合計 {len(finetuning_samples)} 件のファインチューニング用データを {args.output_jsonl_path} に保存しました。")
        
        # バランスの確認
        final_positive = sum(1 for s in finetuning_samples if 'はい' in s['messages'][2]['content'])
        final_negative = len(finetuning_samples) - final_positive
        print(f"最終データバランス: 正例={final_positive}件, 負例={final_negative}件")
        print("-" * 20)
        
    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました - {args.output_jsonl_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
