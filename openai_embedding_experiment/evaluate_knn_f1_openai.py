import json
import os
import yaml
from itertools import combinations
from collections import defaultdict
import networkx as nx
import argparse

# --- グローバル設定 (パス変数はmain関数で上書きされる可能性あり) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))

# デフォルトパス (コマンドライン引数で上書き可能にする)
DEFAULT_KNN_GRAPH_PATH = os.path.join(BASE_DIR, "knn_graphs/knn_graph_openai_k10.json")

# 正解データのパス (プロジェクトルートからの相対パス)
DEFAULT_BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024"  # 1k や 10k を含まない形に
DEFAULT_RECORD_YAML_FILENAME = "extract_subset_10k.yml"  # YAMLファイル名を指定
DEFAULT_RECORD_YAML_PATH = os.path.join(
    PROJECT_ROOT_ASSUMED, DEFAULT_BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, DEFAULT_RECORD_YAML_FILENAME
)


# --- ヘルパー関数群 ---
def get_initial_clusters_from_knn(knn_graph_path_arg):
    if not os.path.exists(knn_graph_path_arg):
        print(f"エラー: K近傍グラフファイルが見つかりません: {knn_graph_path_arg}")
        return [], None
    graph = nx.Graph()
    try:
        with open(knn_graph_path_arg, "r", encoding="utf-8") as f:
            knn_data = json.load(f)
        for record_id_1, neighbors in knn_data.items():
            node1_str = str(record_id_1)
            graph.add_node(node1_str)
            for record_id_2 in neighbors:
                node2_str = str(record_id_2)
                graph.add_node(node2_str)
                if node1_str != node2_str:
                    graph.add_edge(node1_str, node2_str)
        initial_clusters = [set(component) for component in nx.connected_components(graph)]
        print(f"K近傍グラフから {len(initial_clusters)} 個の初期クラスタを特定。 (from {knn_graph_path_arg})")
        return initial_clusters, graph
    except Exception as e:
        print(f"エラー: K近傍グラフからの初期クラスタ特定中にエラー: {e}")
        return [], None


def generate_pairs_from_clusters_list(clusters_list):
    all_pairs = set()
    for cluster in clusters_list:
        if len(cluster) >= 2:
            for pair_tuple in combinations(cluster, 2):
                normalized_pair = tuple(sorted(pair_tuple))
                all_pairs.add(normalized_pair)
    return all_pairs


def load_ground_truth_pairs(record_yaml_path_arg):
    if not os.path.exists(record_yaml_path_arg):
        print(f"エラー: 書誌データファイルが見つかりません: {record_yaml_path_arg}")
        return None
    gt_pairs = set()
    clusters_to_records = defaultdict(list)
    try:
        with open(record_yaml_path_arg, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)
        possible_records_dict = {}
        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]
        if isinstance(possible_records_dict, dict):
            for ck, rlv in possible_records_dict.items():
                if ck in ["version", "type", "id", "summary", "inf_attr"]:
                    continue
                if isinstance(rlv, list):
                    for r in rlv:
                        if isinstance(r, dict) and "id" in r and "cluster_id" in r:
                            rid, cid = str(r["id"]), str(r["cluster_id"])
                            if not cid.startswith("gt_orphan_"):
                                clusters_to_records[cid].append(rid)
        else:
            print(f"エラー: {record_yaml_path_arg} の構造が予期したものではありません。")
            return None
        if not clusters_to_records:
            print(f"エラー: {record_yaml_path_arg} からクラスタ情報をロードできませんでした。")
            return None
        for cid, rids_in_c in clusters_to_records.items():
            if len(rids_in_c) >= 2:
                for pt in combinations(rids_in_c, 2):
                    gt_pairs.add(tuple(sorted(pt)))
        print(f"{len(gt_pairs)} 組の正解類似ペアをロードしました。 (from {record_yaml_path_arg})")
        return gt_pairs
    except Exception as e:
        print(f"エラー: 正解ペアのロード中にエラー: {e}")
        return None


def calculate_and_print_metrics(scenario_name, proposed_pairs_set, ground_truth_pairs_set):
    print(f"\n--- シナリオ: {scenario_name} ---")
    if ground_truth_pairs_set is None:
        print("  正解ペアがロードされていません。評価をスキップします。")
        return

    num_proposed = len(proposed_pairs_set)
    num_gt = len(ground_truth_pairs_set)

    if num_gt == 0:
        print("  正解の類似ペアが0件です。Precision, Recall, F1スコアは計算できません。")
        if num_proposed > 0:
            print(f"  提案されたペアの総数: {num_proposed} (FPとしてカウント)")
            # 正解がない場合、提案されたものは全てFP
            tp, fp, fn = 0, num_proposed, 0
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            print(f"  提案されたペアの総数: {num_proposed}")
            tp, fp, fn = 0, 0, 0
            precision, recall, f1 = 0.0, 0.0, 0.0  # TNしかない状態に近いが、ここでは0とする
    else:
        common_pairs = proposed_pairs_set.intersection(ground_truth_pairs_set)
        tp = len(common_pairs)
        fp = num_proposed - tp
        fn = num_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # num_gt が (tp + fn) と同じはず
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  提案されたペアの総数: {num_proposed}")
    print(f"  正解の類似ペアの総数: {num_gt}")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  Recall:               {recall:.4f}")
    print(f"  F1 Score:             {f1:.4f}")


# --- メイン処理 ---
def main():
    parser = argparse.ArgumentParser(
        description="K近傍グラフの連結成分に基づくブロッキング性能 (F1スコア) を評価します。"
    )
    parser.add_argument(
        "--knn_graph",
        type=str,
        default=DEFAULT_KNN_GRAPH_PATH,
        help=f"K近傍グラフJSONファイルのパス (デフォルト: {DEFAULT_KNN_GRAPH_PATH})",
    )
    parser.add_argument(
        "--ground_truth_yaml",
        type=str,
        default=DEFAULT_RECORD_YAML_PATH,
        help=f"正解データYAMLファイルのパス (デフォルト: {DEFAULT_RECORD_YAML_PATH})",
    )

    args = parser.parse_args()

    knn_graph_file_path = args.knn_graph
    ground_truth_yaml_file_path = args.ground_truth_yaml

    print("K近傍グラフに基づくブロッキング性能 (F1スコア) の評価を開始します...")
    print("\n使用するファイルパス:")
    print(f"  K近傍グラフ     : {knn_graph_file_path}")
    print(f"  正解データ      : {ground_truth_yaml_file_path}\n")

    ground_truth_pairs = load_ground_truth_pairs(ground_truth_yaml_file_path)
    if ground_truth_pairs is None:
        print("正解ペアのロードに失敗したため、処理を中断します。")
        return

    initial_clusters, initial_graph = get_initial_clusters_from_knn(knn_graph_file_path)
    if not initial_clusters or initial_graph is None:
        print("エラー: 初期クラスタまたは初期グラフの取得に失敗しました。処理を中断します。")
        return

    print(f"初期グラフのノード数: {initial_graph.number_of_nodes()}, エッジ数: {initial_graph.number_of_edges()}")

    # K近傍グラフの連結成分をクラスタとみなし、そこから生成される全ペアを評価対象とする
    # これは「もしLLMが連結成分内のペアを完璧に判定できたら」という仮定に基づいた性能評価
    potential_pairs_from_knn_components = generate_pairs_from_clusters_list(initial_clusters)

    calculate_and_print_metrics(
        "K近傍グラフ連結成分 (潜在的ペア)", potential_pairs_from_knn_components, ground_truth_pairs
    )

    print("\n評価処理が完了しました。")


if __name__ == "__main__":
    main()
