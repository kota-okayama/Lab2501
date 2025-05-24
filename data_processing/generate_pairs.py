import itertools
from collections import defaultdict
import random  # For potential sampling later
import argparse  # argparse をインポート
import os  # os をインポート
import json  # json をインポート

# data_processingディレクトリ内のload_yaml_data.pyから関数をインポート
from .load_yaml_data import load_bibliographic_data


def generate_training_pairs(records_list, max_negative_pairs_factor=1.0):
    """
    書誌レコードのリストから、学習用の類似ペアと非類似ペアを生成します。

    Args:
        records_list (list): load_bibliographic_dataから返されるレコードのリスト。
        max_negative_pairs_factor (float): 非類似ペアの数を類似ペアの数に対する倍数で指定するためのファクター。

    Returns:
        tuple: (positive_pairs, negative_pairs)
               各ペアは (record_id_1, record_id_2, label) のタプルです。
    """
    positive_pairs = []

    if not records_list:
        return positive_pairs, []

    clusters = defaultdict(list)
    for record in records_list:
        # record_idのみを保持してメモリを節約
        clusters[record["cluster_id"]].append(record["record_id"])

    for cluster_id, ids_in_cluster in clusters.items():
        if len(ids_in_cluster) > 1:
            for r1_id, r2_id in itertools.combinations(ids_in_cluster, 2):
                positive_pairs.append((r1_id, r2_id, 1))

    num_positive_pairs = len(positive_pairs)
    if num_positive_pairs == 0:
        print("  警告: 類似ペアが0件です。非類似ペアの生成はスキップします。")
        return positive_pairs, []

    target_num_negative_pairs = (
        int(num_positive_pairs * max_negative_pairs_factor) if max_negative_pairs_factor > 0 else num_positive_pairs
    )
    if target_num_negative_pairs == 0:
        print("  情報: 目標とする非類似ペア数が0件のため、非類似ペアの生成はスキップします。")
        return positive_pairs, []

    print(
        f"  目標とする非類似ペア数: {target_num_negative_pairs} (類似ペア数: {num_positive_pairs})慮したファクター: {max_negative_pairs_factor}) "
    )

    # record_id と cluster_id のマッピングを作成 (ペア生成時にクラスタIDを比較するため)
    record_id_to_cluster_id = {record["record_id"]: record["cluster_id"] for record in records_list}
    all_record_ids_for_negative_sampling = [record["record_id"] for record in records_list]
    # もしレコード数が非常に多い場合、ここですべてのrecord_idを保持するのもメモリ負荷になる可能性があるが、
    # YAMLからの読み込み時点でrecords_listとして保持しているので、ここでの追加負荷は限定的と仮定。

    sampled_negative_pairs = []
    count_processed_negative_candidates = 0

    # 全レコードのインデックスペアをイテレート
    # itertools.combinations はイテレータなので、一度に全ペアをメモリに展開しない
    all_indices_pairs = itertools.combinations(range(len(all_record_ids_for_negative_sampling)), 2)

    for i, j in all_indices_pairs:
        # records_listから直接IDとクラスタIDを取得するのではなく、
        # all_record_ids_for_negative_sampling と record_id_to_cluster_id を使う
        record_id1 = all_record_ids_for_negative_sampling[i]
        record_id2 = all_record_ids_for_negative_sampling[j]

        # cluster_id をマッピングから取得
        cluster_id1 = record_id_to_cluster_id.get(record_id1)
        cluster_id2 = record_id_to_cluster_id.get(record_id2)

        if cluster_id1 is not None and cluster_id2 is not None and cluster_id1 != cluster_id2:
            count_processed_negative_candidates += 1
            if len(sampled_negative_pairs) < target_num_negative_pairs:
                sampled_negative_pairs.append((record_id1, record_id2, 0))
            else:
                # Reservoirがいっぱいになったら、確率的に入れ替える (Reservoir Sampling)
                # m番目のアイテムについて、k/m の確率で採用し、reservoir内のランダムな要素と置き換える
                # (k = target_num_negative_pairs, m = count_processed_negative_candidates)
                if random.randint(1, count_processed_negative_candidates) <= target_num_negative_pairs:
                    idx_to_replace = random.randint(0, target_num_negative_pairs - 1)
                    sampled_negative_pairs[idx_to_replace] = (record_id1, record_id2, 0)

    print(f"  処理した非類似ペア候補数: {count_processed_negative_candidates}")
    if len(sampled_negative_pairs) < target_num_negative_pairs and count_processed_negative_candidates > 0:
        print(
            f"  警告: 目標とする非類似ペア数 ({target_num_negative_pairs}) に対して、収集できたのは {len(sampled_negative_pairs)} 件です。データが少ないか、クラスタ構造が偏っている可能性があります。"
        )
    elif count_processed_negative_candidates == 0 and target_num_negative_pairs > 0:
        print(
            f"  警告: 非類似ペア候補が1件も見つかりませんでした。全てのレコードが同じクラスタに属している可能性があります。"
        )

    return positive_pairs, sampled_negative_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training pairs from bibliographic data.")
    parser.add_argument(
        "--input_yaml", type=str, required=True, help="Path to the input YAML file containing bibliographic records."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated positive and negative pair files.",
    )
    # オプションとしてmax_negative_pairs_factorを追加
    parser.add_argument(
        "--neg_pair_factor",
        type=float,
        default=1.0,
        help="Factor to determine the number of negative pairs relative to positive pairs. E.g., 1.0 means same number, 2.0 means twice. Default is 1.0.",
    )

    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print(f"Loading bibliographic data from: {args.input_yaml}")
    bibliographic_records = load_bibliographic_data(args.input_yaml)

    if bibliographic_records:
        print(f"Successfully loaded {len(bibliographic_records)} records.")

        print("\nGenerating training pairs...")
        # コマンドライン引数からファクターを渡す
        positive_pairs, negative_pairs = generate_training_pairs(bibliographic_records, args.neg_pair_factor)

        num_positive = len(positive_pairs)
        num_negative = len(negative_pairs)

        print(f"\nGenerated {num_positive} positive pairs.")
        if positive_pairs:
            print("First 5 positive pairs:")
            for pair in positive_pairs[:5]:
                print(f"  {pair}")

        print(f"\nGenerated {num_negative} negative pairs.")
        if negative_pairs:
            print("First 5 negative pairs:")
            for pair in negative_pairs[:5]:
                print(f"  {pair}")

        # ファイルに保存
        positive_pairs_path = os.path.join(args.output_dir, "positive_pairs.json")
        negative_pairs_path = os.path.join(args.output_dir, "negative_pairs.json")

        try:
            with open(positive_pairs_path, "w", encoding="utf-8") as f_pos:
                json.dump(positive_pairs, f_pos, ensure_ascii=False, indent=4)
            print(f"Positive pairs saved to: {positive_pairs_path}")

            with open(negative_pairs_path, "w", encoding="utf-8") as f_neg:
                json.dump(negative_pairs, f_neg, ensure_ascii=False, indent=4)
            print(f"Negative pairs saved to: {negative_pairs_path}")

        except IOError as e:
            print(f"Error saving pair files: {e}")

    else:
        print(f"Failed to load records from {args.input_yaml}. Cannot generate pairs.")
