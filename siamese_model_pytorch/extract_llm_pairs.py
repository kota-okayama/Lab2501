import json
import os
import csv
import time
import sys
import argparse

# プロジェクトルートをPythonパスに追加 (他モジュールをインポートするため)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- デフォルト値設定 ---
DEFAULT_KNN_GRAPH_FILENAME = "scenarioC_graph_output.json"
DEFAULT_OUTPUT_PAIRS_FILENAME = "evaluation_candidate_pairs_scenarios.csv"
DEFAULT_KNN_GRAPH_DIR_NAME = "knn_graph"
DEFAULT_OUTPUT_CSV_DIR_NAME = "llm_evaluation_pairs"


def extract_unique_pairs_from_knn_graph(knn_graph_path):
    """
    K近傍グラフからユニークなレコードIDのペアを抽出する。

    Args:
        knn_graph_path (str): K近傍グラフのJSONファイルパス。

    Returns:
        set: ユニークなレコードIDのペアのセット。各ペアは (id1, id2) のタプルで、id1 < id2 となっている。
    """
    unique_pairs = set()
    try:
        with open(knn_graph_path, "r", encoding="utf-8") as f:
            knn_graph = json.load(f)
    except FileNotFoundError:
        print(f"エラー: K近傍グラフファイルが見つかりません: {knn_graph_path}")
        return unique_pairs
    except json.JSONDecodeError:
        print(f"エラー: K近傍グラフファイルのJSON形式が正しくありません: {knn_graph_path}")
        return unique_pairs

    for record_id, neighbors in knn_graph.items():
        for neighbor_id in neighbors:
            # 自分自身とのペアは除外
            if record_id == neighbor_id:
                continue
            # ペアを正規化 (id1, id2) where id1 < id2
            pair = tuple(sorted((record_id, neighbor_id)))
            unique_pairs.add(pair)

    return unique_pairs


def save_pairs_to_csv(pairs, output_csv_path):
    """
    レコードIDのペアをCSVファイルに保存する。

    Args:
        pairs (set): 保存するレコードIDのペアのセット。
        output_csv_path (str): 出力するCSVファイルのパス。
    """
    count = 0
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id_1", "record_id_2"])  # ヘッダー
            for pair in sorted(list(pairs)):  # 出力順序を安定させるためにソート
                writer.writerow(pair)
                count += 1
        print(f"{count} ペアを {output_csv_path} に保存しました。")
    except IOError:
        print(f"エラー: CSVファイルへの書き込みに失敗しました: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K近傍グラフからユニークなペアを抽出しCSVに保存するスクリプト")
    parser.add_argument(
        "--input_directory",
        type=str,
        default=None,
        help=f"K近傍グラフファイルが格納されているディレクトリのパス。指定されない場合、スクリプトの場所の '{DEFAULT_KNN_GRAPH_DIR_NAME}' ディレクトリが使われます。",
    )
    parser.add_argument(
        "--knn_graph_filename",
        type=str,
        default=DEFAULT_KNN_GRAPH_FILENAME,
        help=f"K近傍グラフのファイル名 (デフォルト: {DEFAULT_KNN_GRAPH_FILENAME})",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=None,
        help=f"出力CSVファイルを保存するディレクトリのパス。指定されない場合、スクリプトの場所の '{DEFAULT_OUTPUT_CSV_DIR_NAME}' ディレクトリが使われます。",
    )
    parser.add_argument(
        "--output_pairs_filename",
        type=str,
        default=DEFAULT_OUTPUT_PAIRS_FILENAME,
        help=f"出力するCSVファイル名 (デフォルト: {DEFAULT_OUTPUT_PAIRS_FILENAME})",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 入力ディレクトリの決定
    if args.input_directory:
        input_dir = args.input_directory
    else:
        input_dir = os.path.join(script_dir, DEFAULT_KNN_GRAPH_DIR_NAME)

    # 出力ディレクトリの決定
    if args.output_directory:
        output_dir = args.output_directory
    else:
        output_dir = os.path.join(script_dir, DEFAULT_OUTPUT_CSV_DIR_NAME)

    # ファイルパスの構築
    knn_graph_path = os.path.join(input_dir, args.knn_graph_filename)
    output_csv_path = os.path.join(output_dir, args.output_pairs_filename)

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")

    print(f"K近傍グラフファイル: {knn_graph_path}")
    print(f"出力CSVファイル: {output_csv_path}")

    # ペアの抽出
    candidate_pairs = extract_unique_pairs_from_knn_graph(knn_graph_path)

    if candidate_pairs:
        # CSVに保存
        save_pairs_to_csv(candidate_pairs, output_csv_path)
    else:
        print("処理対象のペアが見つかりませんでした。")

    print("処理完了。")
