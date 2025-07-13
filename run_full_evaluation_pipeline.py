#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding生成からLLM評価、ファインチューニングデータ生成までの一連のパイプラインを実行するスクリプト

このスクリプトは以下の5つの主要なステップを自動的に実行します:
1. Embeddingとグラフ生成:
   - `openai_embedding_experiment/run_multi_embedding_pipeline.py` を使用
   - YAML形式の書誌データから複数フィールドのEmbeddingを生成します。
   - 生成したEmbeddingを元にK近傍グラフを構築し、それらを一つに統合します。
2. 評価ペア抽出:
   - 統合されたK近傍グラフから、LLMによる評価対象となるユニークなレコードペアを抽出します。
3. モデル評価:
   - `siamese_model_pytorch/evaluate_finetuning_performance_async.py` を使用
   - 抽出されたペアを用いて、ファインチューニング前後のLLMの性能を非同期で評価し、レポートを生成します。
4. 矛盾する三角形の検出:
   - `siamese_model_pytorch/detect_inconsistent_triangles.py` を使用
   - モデル評価の結果から、推移律に反するペアを検出します。
5. ファインチューニング用データの準備:
   - `siamese_model_pytorch/prepare_finetuning_data.py` を使用
   - 検出した矛盾ペアと判断が難しいペアを組み合わせて、次回のファインチューニング用のデータを生成します。

各ステップはコマンドライン引数でスキップすることが可能です。
"""
import os
import sys
import argparse
import subprocess
import json
import csv
from tqdm import tqdm

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def run_command(command, description):
    """コマンドを実行して結果を表示"""
    print(f"\n{'='*80}")
    print(f"実行中: {description}")
    command_str_list = [str(c) for c in command]
    print(f"コマンド: {' '.join(command_str_list)}")
    print(f"{'='*80}")

    try:
        process = subprocess.Popen(
            command_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"\nエラー: コマンドが失敗しました (終了コード: {process.returncode})")
            return False

        return True

    except FileNotFoundError as e:
        print(f"エラー: コマンド '{e.filename}' が見つかりません。PATHが通っているか確認してください。")
        return False
    except Exception as e:
        print(f"コマンド実行中に予期せぬエラーが発生しました: {e}")
        return False


def run_embedding_and_graph_pipeline(args):
    """STEP 1: 複数フィールドエンベディング生成からK近傍グラフ統合までを実行"""
    print("\n\n===== STEP 1: Embedding生成とK近傍グラフ構築 =====")

    script_path = os.path.join(
        "openai_embedding_experiment", "run_multi_embedding_pipeline.py"
    )

    command = [
        "python3", "-u", script_path,
        "--record_yaml_path", args.record_yaml_path,
        "--output_base_dir", args.output_base_dir,
        "--openai_model", args.openai_embedding_model,
        "--api_batch_size", str(args.api_batch_size),
        "--k_neighbors", str(args.k_neighbors),
    ]

    if args.selected_combinations:
        command.extend(["--selected_combinations", args.selected_combinations])

    if not run_command(command, "Embedding生成とグラフ構築パイプライン"):
        print("STEP 1 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 1完了 =====")


def extract_unique_pairs_from_knn_graph(knn_graph_path):
    """K近傍グラフからユニークなレコードIDのペアを抽出する"""
    unique_pairs = set()
    try:
        with open(knn_graph_path, "r", encoding="utf-8") as f:
            knn_graph = json.load(f)
    except FileNotFoundError:
        print(f"エラー: K近傍グラフファイルが見つかりません: {knn_graph_path}")
        return None
    except json.JSONDecodeError:
        print(f"エラー: K近傍グラフファイルのJSON形式が正しくありません: {knn_graph_path}")
        return None

    desc = "  ペア抽出中"
    for record_id, neighbors in tqdm(
        knn_graph.items(), desc=desc, unit="nodes"
    ):
        for neighbor_id in neighbors:
            if record_id == neighbor_id:
                continue
            pair = tuple(sorted((record_id, neighbor_id)))
            unique_pairs.add(pair)
    return unique_pairs


def save_pairs_to_csv(pairs, output_csv_path):
    """レコードIDのペアをCSVファイルに保存する"""
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id_1", "record_id_2"])
            writer.writerows(sorted(list(pairs)))
        print(f"{len(pairs)} ペアを {output_csv_path} に保存しました。")
        return True
    except IOError as e:
        print(f"エラー: CSVファイルへの書き込みに失敗しました: {output_csv_path} ({e})")
        return False


def get_evaluation_pairs_path(args):
    """評価用ペアCSVのパスを生成する"""
    output_pairs_dir = os.path.join(
        args.output_base_dir, "llm_evaluation_pairs"
    )
    pairs_file_basename = os.path.basename(
        args.record_yaml_path
    ).replace('.yaml', '').replace('.yml', '')
    output_pairs_filename = (
        f"candidate_pairs_from_{pairs_file_basename}_k{args.k_neighbors}.csv"
    )
    return os.path.join(output_pairs_dir, output_pairs_filename)


def run_pair_extraction(args):
    """STEP 2: K近傍グラフから評価用ペアを抽出"""
    print("\n\n===== STEP 2: 評価ペア抽出 =====")

    graphs_dir = os.path.join(args.output_base_dir, "graphs")
    knn_graph_filename = f"merged_knn_graph_k{args.k_neighbors}.json"
    knn_graph_path = os.path.join(graphs_dir, knn_graph_filename)
    output_csv_path = get_evaluation_pairs_path(args)

    print(f"入力K近傍グラフ: {knn_graph_path}")
    print(f"出力ペアCSV: {output_csv_path}")

    candidate_pairs = extract_unique_pairs_from_knn_graph(knn_graph_path)

    if candidate_pairs is None:
        print("ペアの抽出に失敗しました。処理を中断します。")
        sys.exit(1)

    if not candidate_pairs:
        print("処理対象のペアが見つかりませんでした。パイプラインを終了します。")
        sys.exit(0)

    if not save_pairs_to_csv(candidate_pairs, output_csv_path):
        print("ペアの保存に失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 2完了 =====")
    return output_csv_path


def get_evaluation_details_path(args, pairs_csv_path):
    """評価詳細CSVファイルのパスを生成する"""
    base_name = os.path.basename(pairs_csv_path).replace(".csv", "")
    before_model_name = args.model_before_ft.split('/')[-1]
    after_model_name = args.model_after_ft.split('/')[-1]
    
    # 評価スクリプト内の命名規則に合わせる
    output_filename = (
        f"eval_async_{base_name}_before-{before_model_name}_"
        f"after-{after_model_name}_details.csv"
    )
    
    # evaluate_finetuning_performance_async.py は
    # evaluation_results ディレクトリを生成するため、それを考慮
    return os.path.join(
        os.path.dirname(pairs_csv_path),
        "..",
        "evaluation_results",
        output_filename
    )


def run_evaluation(args, pairs_csv_path):
    """STEP 3: ファインチューニング前後のモデル性能を評価"""
    print("\n\n===== STEP 3: モデル評価 =====")

    script_path = os.path.join(
        "siamese_model_pytorch", "evaluate_finetuning_performance_async.py"
    )

    command = [
        "python3", "-u", script_path,
        "--pairs_csv", pairs_csv_path,
        "--ground_truth_yaml", args.record_yaml_path,
        "--model_before_ft", args.model_before_ft,
        "--model_after_ft", args.model_after_ft,
        "--max_concurrent", str(args.max_concurrent),
        "--requests_per_minute", str(args.requests_per_minute),
    ]

    if not run_command(command, "モデル性能評価"):
        print("STEP 3 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 3完了 =====")
    # 次のステップで使うため、生成された評価詳細ファイルのパスを返す
    return get_evaluation_details_path(args, pairs_csv_path)


def run_inconsistency_detection(args, details_csv_path):
    """STEP 4: 矛盾する三角形を検出"""
    print("\n\n===== STEP 4: 矛盾する三角形の検出 =====")

    script_path = os.path.join(
        "siamese_model_pytorch", "detect_inconsistent_triangles.py"
    )

    output_dir = os.path.dirname(details_csv_path)
    base_name = os.path.basename(details_csv_path).replace("_details.csv", "")
    output_csv_path = os.path.join(
        output_dir, f"{base_name}_inconsistent_triangles.csv"
    )

    print(f"入力評価詳細ファイル: {details_csv_path}")
    print(f"出力矛盾三角形CSV: {output_csv_path}")

    command = [
        "python3", "-u", script_path,
        "--details_csv_path", details_csv_path,
        "--output_csv_path", output_csv_path,
        "--score_column", "score_after",
        "--threshold", str(args.inconsistency_threshold),
        "--top_n", str(args.inconsistency_top_n)
    ]

    if not run_command(command, "矛盾する三角形の検出"):
        print("STEP 4 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 4完了 =====")
    return output_csv_path


def run_finetuning_data_preparation(
    args, inconsistent_triangles_csv, details_csv_path
):
    """STEP 5: ファインチューニング用データの準備"""
    print("\n\n===== STEP 5: ファインチューニング用データの準備 =====")

    script_path = os.path.join(
        "siamese_model_pytorch", "prepare_finetuning_data.py"
    )

    base_name = os.path.basename(
        inconsistent_triangles_csv
    ).replace(".csv", "")
    output_jsonl_path = os.path.join(
        os.path.dirname(inconsistent_triangles_csv),
        f"finetuning_data_from_{base_name}.jsonl"
    )

    print(f"入力矛盾ペアCSV: {inconsistent_triangles_csv}")
    print(f"入力評価詳細CSV: {details_csv_path}")
    print(f"出力Finetuning JSONL: {output_jsonl_path}")

    command = [
        "python3", "-u", script_path,
        "--inconsistent_triangles_csv", inconsistent_triangles_csv,
        "--evaluation_details_csv", details_csv_path,
        "--output_jsonl_path", output_jsonl_path,
        "--ground_truth_yaml", args.record_yaml_path
    ]

    if not run_command(command, "ファインチューニング用データの準備"):
        print("STEP 5 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 5完了 =====")
    return output_jsonl_path


def main():
    """メインのパイプライン処理"""
    parser = argparse.ArgumentParser(
        description="Embedding生成からLLM評価、FTデータ生成までの一貫パイプライン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 必須引数 ---
    req_group = parser.add_argument_group('必須引数')
    req_group.add_argument(
        "--record_yaml_path", required=True,
        help="入力レコードと正解クラスタのYAMLファイルパス"
    )
    req_group.add_argument(
        "--output_base_dir", required=True,
        help="全ての出力のベースディレクトリ"
    )
    req_group.add_argument(
        "--model_before_ft", required=True,
        help="ファインチューニング前のモデルID"
    )
    req_group.add_argument(
        "--model_after_ft", required=True,
        help="ファインチューニング後のモデルID"
    )

    # --- Step 1: Embedding & Graphing ---
    step1_group = parser.add_argument_group('Step 1: Embedding & Graphing')
    step1_group.add_argument(
        "--openai_embedding_model", default="text-embedding-ada-002",
        help="OpenAIエンベディングモデル"
    )
    step1_group.add_argument(
        "--api_batch_size", type=int, default=50,
        help="エンベディング生成時のAPIバッチサイズ"
    )
    step1_group.add_argument(
        "--selected_combinations", default="",
        help="使用するフィールド組み合わせ (例: 'full,title_only')"
    )
    step1_group.add_argument(
        "--k_neighbors", type=int, default=15, help="K近傍のK値"
    )

    # --- Step 3: Evaluation ---
    step3_group = parser.add_argument_group('Step 3: Model Evaluation')
    step3_group.add_argument(
        "--max_concurrent", type=int, default=20,
        help="評価時の最大同時リクエスト数"
    )
    step3_group.add_argument(
        "--requests_per_minute", type=int, default=3000,
        help="評価時の1分間の最大リクエスト数"
    )

    # --- Step 4: Inconsistency Detection ---
    step4_group = parser.add_argument_group('Step 4: Inconsistency Detection')
    step4_group.add_argument(
        "--inconsistency_threshold", type=float, default=0.8,
        help="矛盾検出におけるスコアの閾値"
    )
    step4_group.add_argument(
        "--inconsistency_top_n", type=int, default=1000,
        help="検出する矛盾ペアの上位N件"
    )

    # --- Step 5: Finetuning Data Preparation ---
    # このステップには専用の引数はありません。
    # 前のステップの出力を使用します。

    # --- 実行制御 ---
    control_group = parser.add_argument_group('実行制御')
    control_group.add_argument(
        "--skip_embedding_and_graphing", action="store_true",
        help="Step 1 (Embeddingとグラフ生成) をスキップ"
    )
    control_group.add_argument(
        "--skip_pair_extraction", action="store_true",
        help="Step 2 (評価ペア抽出) をスキップ"
    )
    control_group.add_argument(
        "--skip_evaluation", action="store_true",
        help="Step 3 (モデル評価) をスキップ"
    )
    control_group.add_argument(
        "--skip_inconsistency_detection", action="store_true",
        help="Step 4 (矛盾検出) をスキップ"
    )
    control_group.add_argument(
        "--skip_finetuning_data_preparation", action="store_true",
        help="Step 5 (FTデータ準備) をスキップ"
    )

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    # --- パイプライン実行 ---
    # Step 1 & 2
    if not args.skip_embedding_and_graphing:
        run_embedding_and_graph_pipeline(args)
    else:
        print("\n\n===== STEP 1 をスキップしました =====")

    if not args.skip_pair_extraction:
        pairs_csv_path = run_pair_extraction(args)
    else:
        print("\n\n===== STEP 2 をスキップしました =====")
        pairs_csv_path = get_evaluation_pairs_path(args)
        if not os.path.exists(pairs_csv_path):
            print(f"\nエラー: 評価ペアファイルが見つかりません: {pairs_csv_path}")
            sys.exit(1)
        print(f"既存の評価ペアファイルを使用します: {pairs_csv_path}")

    # Step 3
    if not args.skip_evaluation:
        details_csv_path = run_evaluation(args, pairs_csv_path)
    else:
        print("\n\n===== STEP 3 をスキップしました =====")
        details_csv_path = get_evaluation_details_path(args, pairs_csv_path)
        if not os.path.exists(details_csv_path):
            print(f"\nエラー: 評価詳細ファイルが見つかりません: {details_csv_path}")
            sys.exit(1)
        print(f"既存の評価詳細ファイルを使用します: {details_csv_path}")

    # Step 4
    if not args.skip_inconsistency_detection:
        inconsistent_triangles_csv = run_inconsistency_detection(
            args, details_csv_path
        )
    else:
        print("\n\n===== STEP 4 をスキップしました =====")
        output_dir = os.path.dirname(details_csv_path)
        base_name = os.path.basename(
            details_csv_path
        ).replace("_details.csv", "")
        inconsistent_triangles_csv = os.path.join(
            output_dir, f"{base_name}_inconsistent_triangles.csv"
        )
        if not os.path.exists(inconsistent_triangles_csv):
            print(f"エラー: 矛盾三角形ファイルが見つかりません: {inconsistent_triangles_csv}")
            sys.exit(1)
        print(f"既存の矛盾三角形ファイルを使用します: {inconsistent_triangles_csv}")

    # Step 5
    if not args.skip_finetuning_data_preparation:
        run_finetuning_data_preparation(
            args, inconsistent_triangles_csv, details_csv_path
        )
    else:
        print("\n\n===== STEP 5 をスキップしました =====")

    print("\n\nパイプラインの全工程が正常に完了しました。")


if __name__ == "__main__":
    main() 