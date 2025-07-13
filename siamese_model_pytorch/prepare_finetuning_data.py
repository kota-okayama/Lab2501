import csv
import json
import os
import yaml
import sys
import random

# --- グローバル設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024"
RECORD_YAML_FILENAME = "sampled_data_2000.yml"
RECORD_YAML_PATH = os.path.join(
    PROJECT_ROOT_ASSUMED,
    BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT,
    RECORD_YAML_FILENAME,
)

INCONSISTENT_TRIANGLES_FILENAME = "inconsistent_triangles.csv"
INCONSISTENT_TRIANGLES_PATH = os.path.join(
    BASE_DIR, INCONSISTENT_TRIANGLES_FILENAME
)

# 不一致ペアを追加するためのソースファイル
EVALUATION_DETAILS_FILENAME = (
    "eval_async_candidate_pairs_from_sampled_data_2000_k15_before-gpt-4o-mini"
    "-2024-07-18_after-gpt-4o-mini-2024-07-18_details.csv"
)
EVALUATION_DETAILS_PATH = os.path.join(
    PROJECT_ROOT_ASSUMED, "results", "evaluation_results",
    EVALUATION_DETAILS_FILENAME
)

OUTPUT_JSONL_FILENAME = "finetuning_data_balanced_with_hard_negatives.jsonl"
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, OUTPUT_JSONL_FILENAME)

# グローバル変数として書誌データを保持
BIB_DATA = {}


# --- 書誌データ読み込み関連関数 ---
def load_bib_data_for_finetuning(yaml_path):
    global BIB_DATA
    BIB_DATA = {}
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

            if isinstance(possible_records_dict, dict):
                for key, value_list in possible_records_dict.items():
                    if (key in ["version", "type", "id", "summary", "inf_attr"] and
                            possible_records_dict is all_data):
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            actual_bib_data = {}

                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])

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

                                elif record_id_str and not actual_bib_data:
                                    print(
                                        f"警告: レコードID {record_id_str} に有効な"
                                        "書誌データが見つかりませんでした。スキップします。"
                                    )

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。")
            sys.exit(1)
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")

    except yaml.YAMLError as e:
        print(
            f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}"
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラー: {e}"
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_record_details_for_finetuning_prompt(record_id):
    if not BIB_DATA:
        print("エラー: 書誌データがロードされていません。")
        return "情報取得エラー: BIB_DATA未ロード"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    title = bib_details.get("bib1_title", "タイトル不明")
    authors_str = bib_details.get("bib1_author", "著者不明")
    publisher = bib_details.get("bib1_publisher", "出版社不明")
    pubdate = bib_details.get("bib1_pubdate", "出版日不明")
    return f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"


def create_finetuning_sample(record_id_1, record_id_2, is_similar):
    """指定されたペアIDとラベルから、単一のファインチューニングサンプルを作成する。"""
    bib_info_1 = get_record_details_for_finetuning_prompt(record_id_1)
    bib_info_2 = get_record_details_for_finetuning_prompt(record_id_2)

    if (
        "情報取得エラー" in bib_info_1
        or "書誌情報なし" in bib_info_1
        or "情報取得エラー" in bib_info_2
        or "書誌情報なし" in bib_info_2
    ):
        print(
            f"警告: ペア ({record_id_1}, {record_id_2}) の書誌情報取得に失敗。"
            "サンプル生成をスキップします。"
        )
        return None

    system_prompt = (
        "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\\n"
        "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、"
        "そうでない場合は「いいえ」で明確に回答してください。\\n"
        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から"
        "1.0（完全に同一）の範囲で提示してください。"
    )

    user_prompt = (
        f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを"
        f"判断してください。\\n\\n"
        f"書誌情報1:\\n{bib_info_1}\\n\\n"
        f"書誌情報2:\\n{bib_info_2}\\n\\n"
        "これらは同一の文献ですか？\\n回答:"
    )

    assistant_response = (
        "はい\\n類似度スコア: 1.0" if is_similar else "いいえ\\n類似度スコア: 0.0"
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def generate_data_from_all_inconsistent_triangles(filepath, existing_pairs):
    """
    矛盾三角形CSVのすべての行を読み込み、各三角形の3つの構成ペアすべてを
    学習データとして生成する。正解ラベルはCSV内のものを使用する。
    """
    if not os.path.exists(filepath):
        print(f"エラー: 矛盾三角形ファイルが見つかりません: {filepath}。")
        return []

    print(f"{filepath} から全ての矛盾三角形を読み込み、学習データを生成します...")
    finetuning_samples = []

    try:
        with open(filepath, "r", newline="", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                try:
                    id1 = row['triangle_node1']
                    id2 = row['triangle_node2']
                    id3 = row['triangle_node3']

                    is_similar_12 = row['true_edge12'] == 'True'
                    is_similar_23 = row['true_edge23'] == 'True'
                    is_similar_31 = row['true_edge31'] == 'True'

                    pairs_to_process = [
                        ((id1, id2), is_similar_12),
                        ((id2, id3), is_similar_23),
                        ((id3, id1), is_similar_31),
                    ]

                    for (r_id_1, r_id_2), is_similar in pairs_to_process:
                        pair_key = tuple(sorted((str(r_id_1), str(r_id_2))))
                        if pair_key in existing_pairs:
                            continue

                        sample = create_finetuning_sample(r_id_1, r_id_2, is_similar)
                        if sample:
                            finetuning_samples.append(sample)
                            existing_pairs.add(pair_key)
                except KeyError as e:
                    print(f"警告: CSVに必要なキーが見つかりません: {e} (行: {row})。スキップします。")
                    continue
    except Exception as e:
        print(f"矛盾三角形ファイル ({filepath}) の処理中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return []

    return finetuning_samples


def add_balancing_negative_pairs(
    filepath, num_to_add, existing_pairs, score_column_name='score_before'
):
    """評価結果ファイルから、バランス調整用の不一致ペアを追加する。"""
    if not os.path.exists(filepath):
        print(
            f"警告: バランス調整用の評価結果ファイルが見つかりません: {filepath}。"
            "スキップします。"
        )
        return []

    print(f"{filepath} からバランス調整用の不一致ペアを追加します...")
    
    negative_candidates = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                try:
                    if row.get('ground_truth_similar', 'True').lower() == 'false':
                        id1 = row['record_id_1']
                        id2 = row['record_id_2']
                        pair_key = tuple(sorted((id1, id2)))
                        if pair_key in existing_pairs:
                            continue

                        score_str = row.get(score_column_name, '0.0')
                        score = float(score_str) if score_str else 0.0
                        
                        negative_candidates.append({
                            'id1': id1, 'id2': id2, 'score': score
                        })
                except (ValueError, KeyError) as e:
                    print(
                        f"警告: 評価結果ファイルの行処理中にエラー: {e} "
                        f"(行: {row})。スキップします。"
                    )
                    continue
    except Exception as e:
        print(f"評価結果ファイル ({filepath}) の処理中にエラー: {e}")
        return []

    if not negative_candidates:
        print("警告: 追加できる不一致ペア候補がありません。")
        return []
    
    # スコア0.5に近い順にソート (Hard Negatives)
    hard_negatives = sorted(negative_candidates, key=lambda p: abs(p['score'] - 0.5))
    
    num_hard_to_add = num_to_add // 2
    num_random_to_add = num_to_add - num_hard_to_add

    print(
        f"目標追加数: {num_to_add} (困難なペア: {num_hard_to_add}, "
        f"ランダムペア: {num_random_to_add})"
    )
    
    new_negative_pairs = []
    
    # 困難な不一致ペアを追加
    added_keys = set()
    for pair_data in hard_negatives:
        if len(new_negative_pairs) >= num_hard_to_add:
            break
        key = tuple(sorted((pair_data['id1'], pair_data['id2'])))
        if key not in added_keys:
            new_negative_pairs.append((pair_data['id1'], pair_data['id2']))
            added_keys.add(key)
            
    # ランダムな不一致ペアを追加
    remaining_candidates = [
        p for p in negative_candidates
        if tuple(sorted((p['id1'], p['id2']))) not in added_keys
    ]
    
    if len(remaining_candidates) > num_random_to_add:
        selected_randoms = random.sample(
            remaining_candidates, num_random_to_add
        )
    else:
        print(
            f"警告: ランダム不一致ペアの候補が目標数 ({num_random_to_add}) "
            "より少ないため、あるだけ追加します。"
        )
        selected_randoms = remaining_candidates

    for pair_data in selected_randoms:
        new_negative_pairs.append((pair_data['id1'], pair_data['id2']))

    # ファインチューニングサンプルを作成
    added_samples = []
    for id1, id2 in new_negative_pairs:
        pair_key = tuple(sorted((id1, id2)))
        if pair_key in existing_pairs:
            continue
        sample = create_finetuning_sample(id1, id2, is_similar=False)
        if sample:
            added_samples.append(sample)
            existing_pairs.add(pair_key)
            
    print(f"{len(added_samples)} 件の不一致ペアを追加しました。")
    return added_samples


# --- メイン処理 ---
def main():
    print("ファインチューニング用データ作成処理を開始します...")

    load_bib_data_for_finetuning(RECORD_YAML_PATH)

    existing_pairs = set()

    # 1. 矛盾三角形ファイルからベースとなるサンプルを生成
    finetuning_samples = generate_data_from_all_inconsistent_triangles(
        INCONSISTENT_TRIANGLES_PATH, existing_pairs
    )

    # 2. 現在の正解・不正解の数をカウント
    pos_count = sum(1 for s in finetuning_samples if "はい" in s['messages'][-1]['content'])
    neg_count = sum(1 for s in finetuning_samples if "いいえ" in s['messages'][-1]['content'])
    print(f"矛盾ペアからの読み込み結果: 一致 {pos_count}件, 不一致 {neg_count}件")

    # 3. 不一致ペアが不足している場合、バランスを取るために追加
    if pos_count > neg_count:
        num_to_add = pos_count - neg_count
        balancing_samples = add_balancing_negative_pairs(
            EVALUATION_DETAILS_PATH, num_to_add, existing_pairs
        )
        finetuning_samples.extend(balancing_samples)

    if not finetuning_samples:
        print("ファインチューニング対象のサンプルが0件でした。処理を終了します。")
        return

    # 最終結果のカウントとシャッフル
    final_pos_count = sum(1 for s in finetuning_samples if "はい" in s['messages'][-1]['content'])
    final_neg_count = sum(1 for s in finetuning_samples if "いいえ" in s['messages'][-1]['content'])
    
    print(
        f"\n最終的な学習データ: 合計 {len(finetuning_samples)} 件 "
        f"(一致: {final_pos_count}, 不一致: {final_neg_count})"
    )

    random.shuffle(finetuning_samples)

    try:
        with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as outfile:
            for sample in finetuning_samples:
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"ファインチューニング用データを {OUTPUT_JSONL_PATH} に保存しました。")
    except Exception as e:
        print(f"エラー: JSONLファイル書き込み中にエラー: {e}")


if __name__ == "__main__":
    main()
