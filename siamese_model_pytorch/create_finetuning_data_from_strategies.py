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
        title = bib_details.get("title", "タイトル不明")
        authors_str = bib_details.get("artist", "アーティスト不明")
        publisher = bib_details.get("album", "アルバム不明")
        pubdate = bib_details.get("release_date", "リリース日不明")
        length = bib_details.get("length", "長さ不明")
        return (f"タイトル: {title}\nアーティスト: {authors_str}\n"
                f"アルバム: {publisher}\nリリース日: {pubdate}\n長さ: {length}")
    elif data_type == "person":
        givenname = bib_details.get("givenname", "名前不明")
        surname = bib_details.get("surname", "姓不明")
        postcode = bib_details.get("postcode", "郵便番号不明")
        suburb = bib_details.get("suburb", "地域不明")
        return (f"名前: {givenname}\n姓: {surname}\n郵便番号: {postcode}\n地域: {suburb}")
    elif data_type == "walmart_amazon_product":
        name = bib_details.get("title", "商品名不明")
        brand = bib_details.get("brand", "ブランド不明")
        modelno = bib_details.get("modelno", "モデル番号不明")
        price = bib_details.get("price", "価格不明")
        return (f"商品名: {name}\nブランド: {brand}\nモデル番号: {modelno}\n価格: {price}")
    elif data_type == "wdc_product":
        name = bib_details.get("title", "商品名不明")
        brand = bib_details.get("brand", "ブランド不明")
        description = bib_details.get("description", "説明不明")
        price = bib_details.get("price", "価格不明")
        return (f"商品名: {name}\nブランド: {brand}\n説明: {description}\n価格: {price}")
    else:  # bib or default
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
        "walmart_amazon_product": (
            "あなたは2つの商品情報が実質的に同一の商品を指すかどうかを判断する専門家です。\\n"
            "まず、2つの商品情報が同一の商品と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
        "wdc_product": (
            "あなたは2つの商品情報が実質的に同一の商品を指すかどうかを判断する専門家です。\\n"
            "まず、2つの商品情報が同一の商品と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
        "unknown": (
            "あなたは2つの情報が実質的に同一のものを指すかどうかを判断する専門家です。\\n"
            "まず、2つの情報が同一のものと思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
        ),
    }
    return prompt_map.get(data_type, prompt_map["bib"])


def create_finetuning_message(record1_id, record2_id, is_truly_similar, data_type):
    system_prompt = get_prompts(data_type)
    if data_type == "bib":
        user_prompt = (
            f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\\n\\n"
            f"書誌情報1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"書誌情報2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "これらは同一の文献ですか？\\n回答:"
        )
    elif data_type == "music":
        user_prompt = (
            f"以下の2つの音楽情報が、実質的に同一の作品を指しているかどうかを判断してください。\\n\\n"
            f"音楽情報1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"音楽情報2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "これらは同一の作品ですか？\\n回答:"
        )
    elif data_type == "person":
        user_prompt = (
            f"以下の2つの人物情報が、実質的に同一の人物を指しているかどうかを判断してください。\\n\\n"
            f"人物情報1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"人物情報2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "これらは同一の人物ですか？\\n回答:"
        )
    elif data_type == "walmart_amazon_product":
        user_prompt = (
            f"以下の2つの商品情報が、実質的に同一の商品を指しているかどうかを判断してください。\\n\\n"
            f"商品情報1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"商品情報2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "これらは同一の商品ですか？\\n回答:"
        )
    elif data_type == "wdc_product":
        user_prompt = (
            f"以下の2つの商品情報が、実質的に同一の商品を指しているかどうかを判断してください。\\n\\n"
            f"商品情報1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"商品情報2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "これらは同一の商品ですか？\\n回答:"
        )
    else:
        user_prompt = (
            f"以下の2つの実体の情報が、実質的に同一の物を指しているかどうかを判断してください。\\n\\n"
            f"情報1:\\n{get_record_details_for_finetuning_prompt(record1_id, data_type)}\\n\\n"
            f"情報2:\\n{get_record_details_for_finetuning_prompt(record2_id, data_type)}\\n\\n"
            "これらは同一の実体ですか？\\n回答:"
        )
    assistant_response = "はい\\n類似度スコア: 1.0" if is_truly_similar else "いいえ\\n類似度スコア: 0.0"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


# --- サンプリング戦略ごとの関数 ---

def sample_uncertainty(args):
    """不確実性サンプリング"""
    print("不確実性サンプリングを開始します...")
    try:
        df = pd.read_csv(args.evaluation_details_csv)
    except FileNotFoundError:
        print(f"エラー: 評価詳細ファイルが見つかりません: {args.evaluation_details_csv}")
        sys.exit(1)

    df['uncertainty'] = (df[args.score_column] - 0.5).abs()
    sampled_df = df.sort_values(by='uncertainty').head(args.num_samples)
    
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
