import json
import os
import yaml
from itertools import combinations
from collections import defaultdict
import networkx as nx
import numpy as np  # NumPy をインポート
import pickle  # pickle をインポート
import argparse  # argparse をインポート

# --- グローバル設定 (パス変数はmain関数で上書きされる可能性あり) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))

# デフォルトパス (コマンドライン引数で上書き可能にする)
DEFAULT_KNN_GRAPH_PATH = os.path.join(BASE_DIR, "knn_graph/knn_graph_k10.json")
DEFAULT_VECTOR_DATA_DIR = os.path.join(BASE_DIR, "vectorized_data")
DEFAULT_EMBEDDINGS_PATH = os.path.join(DEFAULT_VECTOR_DATA_DIR, "record_embeddings.npy")
DEFAULT_RECORD_IDS_PATH = os.path.join(DEFAULT_VECTOR_DATA_DIR, "record_ids.pkl")
DEFAULT_BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = (
    "benchmark/bib_japan_20241024/1k"  # これはrecord.ymlのデフォルト場所の計算に使用
)
DEFAULT_RECORD_YAML_PATH = os.path.join(
    PROJECT_ROOT_ASSUMED, DEFAULT_BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, "record.yml"
)

# シナリオCのグラフ出力パスのデフォルト値
DEFAULT_OUTPUT_GRAPH_C_PATH = os.path.join(BASE_DIR, "knn_graph/knn_graph_scenarioC_k10.json")

# # 実行時にmain関数で設定されるグローバルパス変数 (初期値はデフォルト)
# # これらをグローバルスコープで宣言しておくことで、各関数が参照可能になる
# # → 関数に直接パスを渡すアプローチに変更するため、これらのグローバル変数は不要に
# KNN_GRAPH_PATH = DEFAULT_KNN_GRAPH_PATH
# EMBEDDINGS_PATH = DEFAULT_EMBEDDINGS_PATH
# RECORD_IDS_PATH = DEFAULT_RECORD_IDS_PATH
# RECORD_YAML_PATH = DEFAULT_RECORD_YAML_PATH


# --- ヘルパー関数群 ---
def load_record_vectors(embeddings_path_arg, ids_path_arg):
    if not os.path.exists(embeddings_path_arg) or not os.path.exists(ids_path_arg):
        print(f"エラー: ベクトルデータファイルまたはIDファイルが見つかりません。")
        print(f"  Embeddings: {embeddings_path_arg}")
        print(f"  Record IDs: {ids_path_arg}")
        return None, None
    try:
        embeddings = np.load(embeddings_path_arg)
        with open(ids_path_arg, "rb") as f:
            record_ids_list = pickle.load(f)
        if len(record_ids_list) != embeddings.shape[0]:
            print("エラー: レコードIDの数とベクトルの数が一致しません。")
            return None, None
        record_vectors_dict = {str(record_ids_list[i]): embeddings[i] for i in range(len(record_ids_list))}
        print(
            f"{len(record_vectors_dict)} 件のレコードベクトルをロードしました。 (from {embeddings_path_arg}, {ids_path_arg})"
        )
        return record_vectors_dict, record_ids_list
    except Exception as e:
        print(f"エラー: レコードベクトルのロード中にエラー: {e}")
        return None, None


def get_initial_clusters_from_knn(knn_graph_path_arg):
    if not os.path.exists(knn_graph_path_arg):
        print(f"エラー: K近傍グラフファイルが見つかりません: {knn_graph_path_arg}")
        return [], None  # グラフオブジェクトも返すように変更
    graph = nx.Graph()
    try:
        with open(knn_graph_path_arg, "r", encoding="utf-8") as f:
            knn_data = json.load(f)
        for record_id_1, neighbors in knn_data.items():
            node1_str = str(record_id_1)  # ノードを文字列として追加
            graph.add_node(node1_str)
            for record_id_2 in neighbors:
                node2_str = str(record_id_2)  # ノードを文字列として追加
                graph.add_node(node2_str)
                if node1_str != node2_str:
                    graph.add_edge(node1_str, node2_str)
        initial_clusters = [set(component) for component in nx.connected_components(graph)]
        print(f"K近傍グラフから {len(initial_clusters)} 個の初期クラスタを特定。 (from {knn_graph_path_arg})")
        return initial_clusters, graph  # グラフオブジェクトも返す
    except Exception as e:
        print(f"エラー: K近傍グラフからの初期クラスタ特定中にエラー: {e}")
        return [], None  # エラー時もNoneを返す


def calculate_cluster_centroid(cluster_record_ids, record_vectors_dict):
    vectors = [record_vectors_dict[rid] for rid in cluster_record_ids if rid in record_vectors_dict]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def distance_between_centroids(centroid1, centroid2):
    if centroid1 is None or centroid2 is None:
        return float("inf")
    return np.linalg.norm(centroid1 - centroid2)


def shortest_distance_between_clusters(cluster1_ids, cluster2_ids, record_vectors_dict):
    min_dist = float("inf")
    cluster1_vectors = [record_vectors_dict[rid] for rid in cluster1_ids if rid in record_vectors_dict]
    cluster2_vectors = [record_vectors_dict[rid] for rid in cluster2_ids if rid in record_vectors_dict]
    if not cluster1_vectors or not cluster2_vectors:
        return float("inf")
    for vec1 in cluster1_vectors:
        for vec2 in cluster2_vectors:
            dist = np.linalg.norm(vec1 - vec2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def find_closest_cluster_pair(clusters, record_vectors_dict, distance_metric="centroid"):
    if len(clusters) < 2:
        return None, None, None, float("inf")
    min_dist_overall = float("inf")
    closest_pair_indices = None
    # closest_c1_ids, closest_c2_ids = None, None # これらは関数のスコープ内で決定されるべき
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster1_ids_local = clusters[i]  # ローカル変数を使用
            cluster2_ids_local = clusters[j]  # ローカル変数を使用
            current_dist = float("inf")
            if distance_metric == "centroid":
                centroid1 = calculate_cluster_centroid(cluster1_ids_local, record_vectors_dict)
                centroid2 = calculate_cluster_centroid(cluster2_ids_local, record_vectors_dict)
                current_dist = distance_between_centroids(centroid1, centroid2)
            elif distance_metric == "shortest":
                current_dist = shortest_distance_between_clusters(
                    cluster1_ids_local, cluster2_ids_local, record_vectors_dict
                )  # 引数順修正
            else:
                raise ValueError("Unknown distance metric")
            if current_dist < min_dist_overall:
                min_dist_overall = current_dist
                closest_pair_indices = (i, j)
                # closest_c1_ids, closest_c2_ids = cluster1_ids_local, cluster2_ids_local # ここでセットするのは冗長、返すときにインデックスで取得
    if closest_pair_indices:
        # 返すときにインデックスを使って実際のクラスタIDセットを取得
        return (
            closest_pair_indices,
            clusters[closest_pair_indices[0]],
            clusters[closest_pair_indices[1]],
            min_dist_overall,
        )
    return None, None, None, float("inf")


def merge_clusters_by_indices(clusters, indices_to_merge):
    if not indices_to_merge or len(indices_to_merge) != 2:
        return clusters
    idx1, idx2 = sorted(indices_to_merge, reverse=True)
    merged_cluster = clusters[idx1].union(clusters[idx2])
    new_clusters = [c for i, c in enumerate(clusters) if i != idx1 and i != idx2]
    new_clusters.append(merged_cluster)
    return new_clusters


def generate_pairs_from_clusters_list(clusters_list):
    all_pairs = set()
    for cluster in clusters_list:
        if len(cluster) >= 2:
            for pair_tuple in combinations(cluster, 2):
                normalized_pair = tuple(sorted(pair_tuple))
                all_pairs.add(normalized_pair)
    return all_pairs


def generate_actual_edge_pairs_from_graph(graph_obj):
    actual_pairs = set()
    if graph_obj:
        for u, v in graph_obj.edges():
            # ノードが文字列であることを期待。get_initial_clusters_from_knnで文字列化済み
            normalized_pair = tuple(sorted((str(u), str(v))))
            actual_pairs.add(normalized_pair)
    return actual_pairs


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


def calculate_and_print_recall(scenario_name, proposed_pairs_set, ground_truth_pairs_set):
    print(f"\n--- シナリオ: {scenario_name} ---")
    if ground_truth_pairs_set is None:
        print("  正解ペア未ロード")
        return
    num_proposed, num_gt = len(proposed_pairs_set), len(ground_truth_pairs_set)
    if num_gt == 0:
        print("  正解ペア0件")
        recall, num_common_pairs = 0.0, 0
    else:
        common_pairs = proposed_pairs_set.intersection(ground_truth_pairs_set)
        num_common_pairs = len(common_pairs)
        recall = num_common_pairs / num_gt if num_gt > 0 else 0.0
    print(f"  提案されたペアの総数: {num_proposed}")
    print(f"  正解の類似ペアの総数: {num_gt}")
    print(f"  共通ペアの数:         {num_common_pairs}")
    print(f"  再現率 (Recall):      {recall:.4f}")


def get_target_connecting_records(cluster_A_ids, cluster_B_ids, record_vectors_dict, num_top_connections=2):
    if not cluster_A_ids or not cluster_B_ids or not record_vectors_dict:
        return [], float("inf")
    min_dist_AB = float("inf")
    a_initial_bridge = None
    b_target_for_A = None
    vecs_A = {rid: record_vectors_dict[rid] for rid in cluster_A_ids if rid in record_vectors_dict}
    vecs_B = {rid: record_vectors_dict[rid] for rid in cluster_B_ids if rid in record_vectors_dict}
    if not vecs_A or not vecs_B:
        return [], float("inf")
    for rid_a, vec_a in vecs_A.items():
        for rid_b, vec_b in vecs_B.items():
            dist = np.linalg.norm(vec_a - vec_b)
            if dist < min_dist_AB:
                min_dist_AB = dist
                a_initial_bridge = rid_a
                b_target_for_A = rid_b
    if a_initial_bridge is None or b_target_for_A is None:
        return [], float("inf")
    distances_to_b_target = []
    b_target_vec = record_vectors_dict.get(b_target_for_A)
    if b_target_vec is None:
        return [], min_dist_AB
    for rid_a_candidate, vec_a_candidate in vecs_A.items():
        dist_to_target = np.linalg.norm(vec_a_candidate - b_target_vec)
        distances_to_b_target.append((rid_a_candidate, dist_to_target))
    distances_to_b_target.sort(key=lambda x: x[1])
    bridge_edges = []
    for i in range(min(num_top_connections, len(distances_to_b_target))):
        a_connector = distances_to_b_target[i][0]
        bridge_edges.append((str(a_connector), str(b_target_for_A)))
    return bridge_edges, min_dist_AB


# --- メイン処理 (改修) ---
def main():
    # global KNN_GRAPH_PATH, EMBEDDINGS_PATH, RECORD_IDS_PATH, RECORD_YAML_PATH # global宣言は不要に

    parser = argparse.ArgumentParser(description="K近傍グラフに基づく多様なブロッキング戦略の再現率評価")
    parser.add_argument(
        "--knn_graph",
        type=str,
        default=DEFAULT_KNN_GRAPH_PATH,
        help=f"K近傍グラフJSONファイルのパス (デフォルト: {DEFAULT_KNN_GRAPH_PATH})",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=DEFAULT_EMBEDDINGS_PATH,
        help=f"レコード埋め込みNPYファイルのパス (デフォルト: {DEFAULT_EMBEDDINGS_PATH})",
    )
    parser.add_argument(
        "--record_ids",
        type=str,
        default=DEFAULT_RECORD_IDS_PATH,
        help=f"レコードID PKLファイルのパス (デフォルト: {DEFAULT_RECORD_IDS_PATH})",
    )
    parser.add_argument(
        "--ground_truth_yaml",
        type=str,
        default=DEFAULT_RECORD_YAML_PATH,
        help=f"正解データYAMLファイルのパス (デフォルト: {DEFAULT_RECORD_YAML_PATH})",
    )
    parser.add_argument(
        "--output_graph_c_path",
        type=str,
        default=DEFAULT_OUTPUT_GRAPH_C_PATH,
        help=f"シナリオCのグラフを保存するJSONファイルのパス (デフォルト: {DEFAULT_OUTPUT_GRAPH_C_PATH})",
    )
    parser.add_argument(
        "--centroid_distance_threshold",
        type=float,
        default=0.5,  # デフォルト値を0.5に設定 (調整が必要な場合があります)
        help="シナリオCでクラスタペアを接続するときのセントロイド間距離の最大閾値 (デフォルト: 0.5)",
    )
    parser.add_argument(
        "--num_bridge_connections",
        type=int,
        default=2,  # デフォルト値を2に設定
        help="シナリオCで接続するクラスタペアごとに作成するブリッジエッジの数 (デフォルト: 2)",
    )

    args = parser.parse_args()

    # コマンドライン引数から取得したパスを変数に格納
    knn_graph_file_path = args.knn_graph
    embeddings_file_path = args.embeddings
    record_ids_file_path = args.record_ids
    ground_truth_yaml_file_path = args.ground_truth_yaml
    output_graph_c_file_path = args.output_graph_c_path
    centroid_distance_threshold = args.centroid_distance_threshold  # 追加
    num_bridge_connections = args.num_bridge_connections  # 追加

    print("K近傍グラフに基づく多様なブロッキング戦略の再現率評価を開始します...")
    print("\n使用するファイルパスとパラメータ:")
    print(f"  K近傍グラフ     : {knn_graph_file_path}")
    print(f"  埋め込みベクトル: {embeddings_file_path}")
    print(f"  レコードIDリスト: {record_ids_file_path}")
    print(f"  正解データ      : {ground_truth_yaml_file_path}")
    print(f"  シナリオCグラフ出力: {output_graph_c_file_path}")
    print(f"  シナリオC セントロイド距離閾値: {centroid_distance_threshold}")  # 追加
    print(f"  シナリオC ブリッジ接続数: {num_bridge_connections}\n")  # 追加

    ground_truth_pairs = load_ground_truth_pairs(ground_truth_yaml_file_path)
    if ground_truth_pairs is None:
        return

    record_vectors, _ = load_record_vectors(embeddings_file_path, record_ids_file_path)
    if record_vectors is None:
        return

    initial_clusters, initial_graph = get_initial_clusters_from_knn(knn_graph_file_path)
    if not initial_clusters or initial_graph is None:
        print("エラー: 初期クラスタまたは初期グラフの取得に失敗しました。処理を中断します。")
        return

    print(f"初期クラスタ数: {len(initial_clusters)}")
    print(f"初期グラフのノード数: {initial_graph.number_of_nodes()}, エッジ数: {initial_graph.number_of_edges()}")

    # シナリオ0: K近傍連結成分 (マージなし)
    # 潜在的再現率
    pairs_scenario0_potential = generate_pairs_from_clusters_list(initial_clusters)
    calculate_and_print_recall(
        "K近傍連結成分 (マージなし) - 潜在的再現率", pairs_scenario0_potential, ground_truth_pairs
    )
    # (グラフエッジ)再現率
    pairs_scenario0_actual_edges = generate_actual_edge_pairs_from_graph(initial_graph)
    calculate_and_print_recall(
        "K近傍連結成分 (マージなし) - グラフエッジ再現率", pairs_scenario0_actual_edges, ground_truth_pairs
    )

    if len(initial_clusters) < 2:
        print("\n初期クラスタが1つ以下のため、マージ/接続処理はスキップします。")
    else:
        # --- シナリオA: セントロイド間距離で最も近い1組のクラスタをマージ ---
        print("\n[シナリオA] セントロイド間距離で最も近い1組のクラスタをマージします...")
        indices_A, c1_A_ids_actual, c2_A_ids_actual, dist_A = find_closest_cluster_pair(
            initial_clusters, record_vectors, distance_metric="centroid"
        )
        if indices_A:
            print(f"  -> マージ対象インデックス (セントロイド間距離 {dist_A:.4f}): {indices_A}")
            clusters_scenarioA = merge_clusters_by_indices(initial_clusters, indices_A)
            # 潜在的再現率
            pairs_scenarioA_potential = generate_pairs_from_clusters_list(clusters_scenarioA)
            calculate_and_print_recall(
                "セントロイド間距離で1組マージ - 潜在的再現率", pairs_scenarioA_potential, ground_truth_pairs
            )
            # (グラフエッジ)再現率 - マージは概念的なので、エッジはinitial_graphのものをそのまま使う
            calculate_and_print_recall(
                "セントロイド間距離で1組マージ - グラフエッジ再現率", pairs_scenario0_actual_edges, ground_truth_pairs
            )
        else:
            print("  セントロイド間距離でマージ可能なペアが見つかりませんでした。")

        # --- シナリオB: 最短レコード間距離で最も近い1組のクラスタをマージ ---
        print("\n[シナリオB] 最短レコード間距離で最も近い1組のクラスタをマージします...")
        indices_B, c1_B_ids_actual, c2_B_ids_actual, dist_B = find_closest_cluster_pair(
            initial_clusters, record_vectors, distance_metric="shortest"
        )
        if indices_B:
            print(f"  -> マージ対象インデックス (最短レコード間距離 {dist_B:.4f}): {indices_B}")
            clusters_scenarioB = merge_clusters_by_indices(initial_clusters, indices_B)
            # 潜在的再現率
            pairs_scenarioB_potential = generate_pairs_from_clusters_list(clusters_scenarioB)
            calculate_and_print_recall(
                "最短レコード間距離で1組マージ - 潜在的再現率", pairs_scenarioB_potential, ground_truth_pairs
            )
            # (グラフエッジ)再現率 - マージは概念的なので、エッジはinitial_graphのものをそのまま使う
            calculate_and_print_recall(
                "最短レコード間距離で1組マージ - グラフエッジ再現率", pairs_scenario0_actual_edges, ground_truth_pairs
            )
        else:
            print("  最短レコード間距離でマージ可能なペアが見つかりませんでした。")

        # --- シナリオC: ターゲット接続戦略 (改修: 閾値ベースで複数ペア接続) ---
        print(
            f"\n[シナリオC] セントロイド間距離が閾値 ({centroid_distance_threshold}) 以下の複数クラスタペア間で接続 (各{num_bridge_connections}本)..."
        )

        graph_C = initial_graph.copy()
        num_connected_pairs = 0

        if len(initial_clusters) >= 2:
            # 最初に全クラスタのセントロイドを計算
            cluster_centroids = []
            valid_initial_clusters_for_C = []  # セントロイドが計算できたクラスタのみを対象とする
            for i, cluster_ids in enumerate(initial_clusters):
                centroid = calculate_cluster_centroid(cluster_ids, record_vectors)
                if centroid is not None:
                    cluster_centroids.append(centroid)
                    valid_initial_clusters_for_C.append(cluster_ids)
                else:
                    print(
                        f"  警告: クラスタ {i} (サイズ {len(cluster_ids)}) のセントロイドが計算できませんでした。スキップします。"
                    )

            print(f"  セントロイド計算対象の有効な初期クラスタ数: {len(valid_initial_clusters_for_C)}")

            if len(valid_initial_clusters_for_C) >= 2:
                for i in range(len(valid_initial_clusters_for_C)):
                    for j in range(i + 1, len(valid_initial_clusters_for_C)):
                        cluster_A_ids = valid_initial_clusters_for_C[i]
                        cluster_B_ids = valid_initial_clusters_for_C[j]
                        centroid_A = cluster_centroids[i]
                        centroid_B = cluster_centroids[j]

                        dist_centroids = distance_between_centroids(centroid_A, centroid_B)

                        if dist_centroids <= centroid_distance_threshold:
                            num_connected_pairs += 1
                            print(
                                f"  -> 接続候補: クラスタ {i} と クラスタ {j} (セントロイド間距離: {dist_centroids:.4f} <= {centroid_distance_threshold})"
                            )

                            # get_target_connecting_records を使用してブリッジエッジを構築
                            # この関数は内部で最短レコード間距離も計算するが、ここでは接続のトリガーとしてセントロイド間距離を使用
                            bridge_edges_C, min_dist_records = get_target_connecting_records(
                                cluster_A_ids, cluster_B_ids, record_vectors, num_top_connections=num_bridge_connections
                            )
                            print(
                                f"    構築されるブリッジエッジ: {bridge_edges_C} (レコード間最短距離: {min_dist_records:.4f})"
                            )

                            for u_bridge, v_bridge in bridge_edges_C:
                                graph_C.add_edge(str(u_bridge), str(v_bridge))
            else:
                print("  有効な初期クラスタが2つ未満のため、シナリオCの接続処理はスキップします。")

        if (
            num_connected_pairs == 0 and len(initial_clusters) >= 2 and len(valid_initial_clusters_for_C) >= 2
        ):  # len(initial_clusters) >=2 and len(valid_initial_clusters_for_C) >=2 を追加
            print(
                "  閾値以下のセントロイド間距離を持つクラスタペアが見つからなかったため、ブリッジエッジは追加されませんでした。"
            )

        print(f"シナリオCグラフのノード数: {graph_C.number_of_nodes()}, エッジ数: {graph_C.number_of_edges()}")
        final_clusters_C = [set(component) for component in nx.connected_components(graph_C)]

        # 潜在的再現率
        pairs_scenarioC_potential = generate_pairs_from_clusters_list(final_clusters_C)
        calculate_and_print_recall(
            f"ターゲット接続戦略 (閾値{centroid_distance_threshold}, {num_bridge_connections}本) - 潜在的再現率",
            pairs_scenarioC_potential,
            ground_truth_pairs,
        )
        # (グラフエッジ)再現率
        pairs_scenarioC_actual_edges = generate_actual_edge_pairs_from_graph(graph_C)
        calculate_and_print_recall(
            f"ターゲット接続戦略 (閾値{centroid_distance_threshold}, {num_bridge_connections}本) - グラフエッジ再現率",
            pairs_scenarioC_actual_edges,
            ground_truth_pairs,
        )

        # シナリオCのグラフをJSONファイルとして保存
        if graph_C:
            try:
                graph_c_data = nx.to_dict_of_lists(graph_C)
                # 出力ディレクトリが存在しない場合は作成
                output_dir = os.path.dirname(output_graph_c_file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"作成されたディレクトリ: {output_dir}")

                with open(output_graph_c_file_path, "w", encoding="utf-8") as f:
                    json.dump(graph_c_data, f, indent=2, ensure_ascii=False)
                print(f"シナリオCのグラフを {output_graph_c_file_path} に保存しました。")
            except Exception as e:
                print(f"エラー: シナリオCのグラフ保存中にエラーが発生しました: {e}")

    print("\n評価処理が完了しました。")


if __name__ == "__main__":
    main()
