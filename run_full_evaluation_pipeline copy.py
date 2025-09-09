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
   - `prepare_finetuning_data.py` (inconsistency) と
     `create_finetuning_data_from_strategies.py` (diversity, etc.) を使用
   - 複数の戦略に基づき、次回のファインチューニング用のデータを生成します。

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
        print(f"エラー: コマンド '{e.filename}' が見つかりません。"
              "PATHが通っているか確認してください。")
        return False
    except Exception as e:
        print(f"コマンド実行中に予期せぬエラーが発生しました: {e}")
        return False


def sanitize_model_name_for_filename(model_name):
    """ファイル名に使用できるようにモデル名をサニタイズする"""
    return model_name.replace('/', '_').replace(':', '_').replace(' ', '_')


def get_num_lines_in_jsonl(file_path):
    """JSONLファイルの行数を数える"""
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


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
        "--embedding_combinations", args.embedding_combinations
    ]

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


def get_legacy_evaluation_details_path(args, pairs_csv_path):
    """【旧バージョン互換】古い命名規則で評価詳細CSVファイルのパスを生成する"""
    if len(args.model_ids) < 2:
        return None # 古い形式は少なくとも2つのモデルIDを前提としていた
    # 互換性のために、最初の2つのモデルIDを旧引数とみなす
    return get_evaluation_details_path(
        args, pairs_csv_path, args.model_ids[0], args.model_ids[1]
    )

def get_evaluation_details_path(args, pairs_csv_path, model_before_ft, model_after_ft):
    """評価詳細CSVファイルのパスを生成する"""
    base_name = os.path.basename(pairs_csv_path).replace(".csv", "")
    before_model_name = sanitize_model_name_for_filename(model_before_ft)
    after_model_name = sanitize_model_name_for_filename(model_after_ft)

    output_filename = (
        f"eval_async_{base_name}_before-{before_model_name}_"
        f"after-{after_model_name}_details.csv"
    )

    return os.path.join(
        os.path.dirname(pairs_csv_path),
        "..",
        "evaluation_results",
        output_filename
    )


def run_evaluation(args, pairs_csv_path, model_before, model_after):
    """STEP 3: ファインチューニング前後のモデル性能を評価"""
    print(f"\n\n===== STEP 3: モデル評価 (Before: {model_before}, After: {model_after}) =====")

    script_path = os.path.join(
        "siamese_model_pytorch", "evaluate_finetuning_performance_async.py"
    )

    command = [
        "python3", "-u", script_path,
        "--pairs_csv", pairs_csv_path,
        "--ground_truth_yaml", args.record_yaml_path,
        "--data_type", args.data_type,
        "--model_before_ft", model_before,
        "--model_after_ft", model_after,
        "--max_concurrent", str(args.max_concurrent),
        "--requests_per_minute", str(args.requests_per_minute),
    ]

    if not run_command(command, "モデル性能評価"):
        print("STEP 3 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 3完了 =====")
    return get_evaluation_details_path(args, pairs_csv_path, model_before, model_after)


def run_inconsistency_detection(args, details_csv_path):
    """STEP 4: 矛盾する三角形を検出"""
    print("\n\n===== STEP 4: 矛盾する三角形の検出 =====")

    script_path = os.path.join(
        "siamese_model_pytorch", "detect_inconsistent_triangles.py"
    )

    output_dir = os.path.dirname(details_csv_path)

    command = [
        "python3", "-u", script_path,
        "--input-csv", details_csv_path,
        "--ground-truth-yaml", args.record_yaml_path,
        "--output-dir", output_dir,
        "--score-column", "score_after",
        "--num-triangles", str(args.inconsistency_top_n)
    ]

    if not run_command(command, "矛盾する三角形の検出"):
        print("STEP 4 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 4完了 =====")
    base_name = os.path.basename(details_csv_path).replace("_details.csv", "")
    return os.path.join(
        output_dir, f"{base_name}_score_after_inconsistent_triangles.csv"
    )


def run_finetuning_data_preparation(
    args, inconsistent_triangles_csv, details_csv_path, llm_clusters_json_path
):
    """STEP 5: ファインチューニング用データの準備（複数戦略）"""
    print("\n\n===== STEP 5: ファインチューニング用データの準備 =====")

    strategies = args.ft_strategies.split(',')
    base_name_for_ft = os.path.basename(
        details_csv_path
    ).replace("_details.csv", "")
    output_dir = os.path.dirname(details_csv_path)
    num_samples = 0

    # 1. Inconsistency-based strategy (if requested)
    if 'inconsistency' in strategies:
        print("\n--- 戦略: inconsistency ---")
        script_path = os.path.join(
            "siamese_model_pytorch", "prepare_finetuning_data.py"
        )
        output_jsonl_path = os.path.join(
            output_dir, f"ft_data_{base_name_for_ft}_inconsistency.jsonl"
        )
        command = [
            "python3", "-u", script_path,
            "--inconsistent_triangles_csv", inconsistent_triangles_csv,
            "--evaluation_details_csv", details_csv_path,
            "--ground_truth_yaml", args.record_yaml_path,
            "--output_jsonl_path", output_jsonl_path,
            "--data_type", args.data_type,
            "--score_column", "score_before"
        ]
        if not run_command(command, "FTデータ準備 (inconsistency)"):
            print("inconsistency 戦略が失敗しました。")
        else:
            num_samples = get_num_lines_in_jsonl(output_jsonl_path)
            print(f"inconsistency 戦略で {num_samples} 件のデータを生成しました。")

    if num_samples == 0 and len(strategies) > 1:
        print("警告: inconsistency戦略のサンプル数が0です。"
              "他の戦略のサンプル数を決定できません。スキップします。")
        return

    # 2. Other strategies
    other_strategies = [s for s in strategies if s != 'inconsistency']
    for strategy in other_strategies:
        print(f"\n--- 戦略: {strategy} ---")
        script_path = os.path.join(
            "siamese_model_pytorch",
            "create_finetuning_data_from_strategies.py"
        )
        output_jsonl_path = os.path.join(
            output_dir, f"ft_data_{base_name_for_ft}_{strategy}.jsonl"
        )
        command = [
            "python3", "-u", script_path,
            "--strategy", strategy,
            "--output_jsonl_path", output_jsonl_path,
            "--ground_truth_yaml", args.record_yaml_path,
            "--num_samples", str(num_samples),
            "--data_type", args.data_type,
        ]
        if strategy in ["uncertainty", "random"]:
            command.extend(["--evaluation_details_csv", details_csv_path])
        if strategy == "uncertainty":
            command.extend(["--score_column", "score_after"])
        if strategy == "diversity":
            command.extend(["--llm_clusters_json", llm_clusters_json_path])

        if not run_command(command, f"FTデータ準備 ({strategy})"):
            print(f"{strategy} 戦略が失敗しました。")

    print("\n===== STEP 5完了 =====")


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
        "--data_type",
        required=True,
        choices=[
            "bib", "music", "person",
            "walmart_amazon_product", "wdc_product"
        ],
        help="評価対象データの種類"
    )
    req_group.add_argument(
        "--model_ids",
        nargs='+',
        required=True,
        help="評価対象のモデルIDを順番に指定します。ペアで評価され、"
             "奇数個の場合は最後のモデルが最初のモデルとペアになります。"
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
        "--embedding_combinations", type=str, default="full",
        help="生成するエンベディングの組み合わせ (例: 'full;title')"
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
        "--inconsistency_top_n", type=int, default=100,
        help="検出する矛盾ペアの上位N件"
    )

    # --- Step 5: Finetuning Data Preparation ---
    step5_group = parser.add_argument_group('Step 5: Finetuning Data Prep')
    step5_group.add_argument(
        "--ft-strategies", type=str,
        default="inconsistency,diversity,uncertainty,random",
        help="実行するFTデータ生成戦略（カンマ区切り）"
    )

    # --- 実行制御 ---
    control_group = parser.add_argument_group('実行制御')
    control_group.add_argument(
        "--skip_step_1", action="store_true",
        help="Step 1 をスキップ"
    )
    control_group.add_argument(
        "--skip_step_2", action="store_true", help="Step 2 をスキップ"
    )
    control_group.add_argument(
        "--skip_step_3", action="store_true", help="Step 3 をスキップ"
    )
    control_group.add_argument(
        "--skip_step_4", action="store_true",
        help="Step 4 をスキップ"
    )
    control_group.add_argument(
        "--skip_step_5", action="store_true",
        help="Step 5 をスキップ"
    )

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    # --- パイプライン実行 ---
    if not args.skip_step_1:
        run_embedding_and_graph_pipeline(args)
    else:
        print("\n\n===== STEP 1 をスキップしました =====")

    if not args.skip_step_2:
        pairs_csv_path = run_pair_extraction(args)
    else:
        print("\n\n===== STEP 2 をスキップしました =====")
        pairs_csv_path = get_evaluation_pairs_path(args)
        if not os.path.exists(pairs_csv_path):
            print(f"エラー: 評価ペアファイルが見つかりません: {pairs_csv_path}")
            sys.exit(1)
        print(f"既存の評価ペアファイルを使用します: {pairs_csv_path}")

    # --- Create Model Pairs ---
    model_ids = args.model_ids
    if not model_ids:
        print("エラー: --model_ids には少なくとも1つのモデルIDを指定してください。")
        sys.exit(1)

    model_pairs = []
    # 偶数個のモデルIDをペアにする
    for i in range(0, len(model_ids) - (len(model_ids) % 2), 2):
        model_pairs.append((model_ids[i], model_ids[i + 1]))

    # モデルIDが奇数個の場合、最後のものを最初のものとペアにする
    if len(model_ids) % 2 != 0:
        model_pairs.append((model_ids[0], model_ids[-1]))

    # --- Loop for Steps 3, 4, 5 for each model pair ---
    if args.skip_step_3 and args.skip_step_4 and args.skip_step_5:
        print("\n\n===== STEP 3, 4, 5 をスキップしました =====")
    else:
        for i, (model_before, model_after) in enumerate(model_pairs):
            print(f"\n\n{'#'*80}")
            print(f"#### モデルペア {i+1}/{len(model_pairs)} の処理を開始... ####")
            print(f"#### BEFORE: {model_before}")
            print(f"#### AFTER:  {model_after}")
            print(f"{'#'*80}")

            # --- Step 3 ---
            if not args.skip_step_3:
                details_csv_path = run_evaluation(
                    args, pairs_csv_path, model_before, model_after
                )
            else:
                print("\n\n===== STEP 3 をスキップしました =====")
                # 新しい命名規則でパスを取得
                details_csv_path = get_evaluation_details_path(
                    args, pairs_csv_path, model_before, model_after
                )
                if not os.path.exists(details_csv_path):
                    # 見つからない場合、古い命名規則でフォールバック
                    print(f"  -> 新しい命名規則のファイルが見つかりません。古い命名規則で再試行します...")
                    legacy_path = get_legacy_evaluation_details_path(args, pairs_csv_path)
                    if legacy_path and os.path.exists(legacy_path):
                        details_csv_path = legacy_path
                        print(f"  -> 古い命名規則のファイルを使用します: {details_csv_path}")
                    else:
                        print(f"エラー: 評価詳細ファイルが見つかりません: {details_csv_path}")
                        print("Step 3をスキップするには、このファイルが事前に存在している必要があります。")
                        continue  # 次のペアへ
                print(f"既存の評価詳細ファイルを使用します: {details_csv_path}")

            # --- Step 4 ---
            if not args.skip_step_4:
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
                    output_dir,
                    f"{base_name}_score_after_inconsistent_triangles.csv"
                )
                if not os.path.exists(inconsistent_triangles_csv):
                     print(f"  -> 新しい命名規則のファイルが見つかりません。")
                     # 古いパスはdetails_csv_pathに依存するため、details_csv_pathが古いパスなら自動で古いパスを探すことになる
                     print(f"エラー: 矛盾三角形ファイルが見つかりません: {inconsistent_triangles_csv}")
                     print("Step 4をスキップするには、このファイルが事前に存在している必要があります。")
                     continue # 次のペアへ
                print(f"既存の矛盾三角形ファイルを使用します: {inconsistent_triangles_csv}")

            # --- Step 5 ---
            if not args.skip_step_5:
                # Step5で必要なクラスタファイルのパスを決定
                base_name = os.path.basename(
                    details_csv_path
                ).replace("_details.csv", "")
                clusters_path = os.path.join(
                    os.path.dirname(details_csv_path),
                    f"{base_name}_clusters_after.json"
                )
                run_finetuning_data_preparation(
                    args, inconsistent_triangles_csv, details_csv_path, clusters_path
                )
            else:
                print("\n\n===== STEP 5 をスキップしました =====")

    print("\n\nパイプラインの全工程が正常に完了しました。")


if __name__ == "__main__":
    main()
