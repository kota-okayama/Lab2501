#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding生成からLLM評価までの一連のパイプラインを実行するスクリプト

このスクリプトは以下の3つの主要なステップを自動的に実行します:
1. Embeddingとグラフ生成:
   - `openai_embedding_experiment/run_multi_embedding_pipeline.py` を使用
   - YAML形式の書誌データから複数フィールドのEmbeddingを生成します。
   - 生成したEmbeddingを元にK近傍グラフを構築し、それらを一つに統合します。
2. 評価ペア抽出:
   - `siamese_model_pytorch/extract_llm_pairs.py` のロジックを使用
   - 統合されたK近傍グラフから、LLMによる評価対象となるユニークなレコードペアを抽出します。
3. モデル評価:
   - `siamese_model_pytorch/evaluate_finetuning_performance_async.py` を使用
   - 抽出されたペアを用いて、ファインチューニング前後のLLMの性能を非同期で評価し、レポートを生成します。

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
    """コマンドをWSL経由で実行して結果を表示"""
    print(f"\n{'='*80}")
    print(f"実行中: {description}")
    # コマンドの各要素を文字列に変換
    command_str_list = [str(c) for c in command]
    print(f"コマンド: {' '.join(command_str_list)}")
    print(f"{'='*80}")

    try:
        # Popenを使用して出力をリアルタイムでストリーミングする
        process = subprocess.Popen(
            command_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1  # 行単位のバッファリング
        )

        # リアルタイムで出力を表示
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
        "python3",
        "-u",  # 出力バッファリングを無効にする
        script_path,
        "--record_yaml_path", args.record_yaml_path,
        "--output_base_dir", args.output_base_dir,
        "--openai_model", args.openai_embedding_model,
        "--api_batch_size", args.api_batch_size,
        "--k_neighbors", args.k_neighbors,
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

    for record_id, neighbors in tqdm(knn_graph.items(), desc="  ペア抽出中", unit=" nodes"):
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


def run_pair_extraction(args):
    """STEP 2: K近傍グラフから評価用ペアを抽出"""
    print("\n\n===== STEP 2: 評価ペア抽出 =====")

    graphs_dir = os.path.join(args.output_base_dir, "graphs")
    knn_graph_filename = f"merged_knn_graph_k{args.k_neighbors}.json"
    knn_graph_path = os.path.join(graphs_dir, knn_graph_filename)

    output_pairs_dir = os.path.join(args.output_base_dir, "llm_evaluation_pairs")
    pairs_file_basename = os.path.basename(args.record_yaml_path).replace('.yaml', '').replace('.yml', '')
    output_pairs_filename = f"candidate_pairs_from_{pairs_file_basename}_k{args.k_neighbors}.csv"
    output_csv_path = os.path.join(output_pairs_dir, output_pairs_filename)

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


def run_evaluation(args, pairs_csv_path):
    """STEP 3: ファインチューニング前後のモデル性能を評価"""
    print("\n\n===== STEP 3: モデル評価 =====")

    script_path = os.path.join(
        "siamese_model_pytorch", "evaluate_finetuning_performance_async.py"
    )

    command = [
        "python3",
        "-u",  # 出力バッファリングを無効にする
        script_path,
        "--pairs_csv", pairs_csv_path,
        "--ground_truth_yaml", args.record_yaml_path,
        "--model_before_ft", args.model_before_ft,
        "--model_after_ft", args.model_after_ft,
        "--max_concurrent", args.max_concurrent,
        "--requests_per_minute", args.requests_per_minute,
    ]

    # evaluate_finetuning_performance_async.pyは、入力された
    # pairs_csvのパスに基づいて出力先を決定するため、ここではコマンドを呼び出すだけ。
    if not run_command(command, "モデル性能評価"):
        print("STEP 3 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 3完了 =====")


def main():
    """メインのパイプライン処理"""
    parser = argparse.ArgumentParser(
        description="Embedding生成からLLM評価までの一貫パイプライン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 必須引数 ---
    parser.add_argument("--record_yaml_path", required=True, help="入力レコードと正解クラスタのYAMLファイルパス")
    parser.add_argument("--output_base_dir", required=True, help="全ての出力のベースディレクトリ")
    parser.add_argument("--model_before_ft", required=True, help="ファインチューニング前のモデルID")
    parser.add_argument("--model_after_ft", required=True, help="ファインチューニング後のモデルID")

    # --- Step 1: Embedding & Graphing ---
    step1_group = parser.add_argument_group('Step 1: Embedding & Graphing オプション')
    step1_group.add_argument("--openai_embedding_model", default="text-embedding-ada-002", help="OpenAIエンベディングモデル")
    step1_group.add_argument("--api_batch_size", type=int, default=50, help="エンベディング生成時のAPIバッチサイズ")
    step1_group.add_argument("--selected_combinations", default="", help="使用するフィールド組み合わせ (例: 'full,title_only')")
    step1_group.add_argument("--k_neighbors", type=int, default=15, help="K近傍のK値")

    # --- Step 3: Evaluation ---
    step3_group = parser.add_argument_group('Step 3: Model Evaluation オプション')
    step3_group.add_argument("--max_concurrent", type=int, default=20, help="評価時の最大同時リクエスト数")
    step3_group.add_argument("--requests_per_minute", type=int, default=3000, help="評価時の1分間の最大リクエスト数")

    # --- 実行制御 ---
    control_group = parser.add_argument_group('実行制御')
    control_group.add_argument("--skip_embedding_and_graphing", action="store_true", help="Step 1 (Embeddingとグラフ生成) をスキップ")
    control_group.add_argument("--skip_pair_extraction", action="store_true", help="Step 2 (評価ペア抽出) をスキップ")
    control_group.add_argument("--skip_evaluation", action="store_true", help="Step 3 (モデル評価) をスキップ")

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    # --- パイプライン実行 ---
    if not args.skip_embedding_and_graphing:
        run_embedding_and_graph_pipeline(args)
    else:
        print("\n\n===== STEP 1 (Embedding生成とK近傍グラフ構築) をスキップしました =====")

    if not args.skip_pair_extraction:
        pairs_csv_path = run_pair_extraction(args)
    else:
        print("\n\n===== STEP 2 (評価ペア抽出) をスキップしました =====")
        # スキップした場合、評価用のペアCSVパスを生成ルールから推測する
        output_pairs_dir = os.path.join(args.output_base_dir, "llm_evaluation_pairs")
        pairs_file_basename = os.path.basename(args.record_yaml_path).replace('.yaml', '').replace('.yml', '')
        output_pairs_filename = f"candidate_pairs_from_{pairs_file_basename}_k{args.k_neighbors}.csv"
        pairs_csv_path = os.path.join(output_pairs_dir, output_pairs_filename)
        
        if not os.path.exists(pairs_csv_path):
            print(f"\nエラー: 評価ペアファイルが見つかりません: {pairs_csv_path}")
            print("ペア抽出をスキップしましたが、必要なファイルが存在しませんでした。")
            print("`--skip_pair_extraction` を付けずに実行してファイルを生成してください。")
            sys.exit(1)
        print(f"既存の評価ペアファイルを使用します: {pairs_csv_path}")

    if not args.skip_evaluation:
        if 'pairs_csv_path' not in locals():
            print("\nエラー: 評価ペアのCSVパスが定義されていません。これは通常、ステップ2をスキップした際に発生します。")
            sys.exit(1)
        run_evaluation(args, pairs_csv_path)
    else:
        print("\n\n===== STEP 3 (モデル評価) をスキップしました =====")

    print("\n\nパイプラインの全工程が正常に完了しました。")


if __name__ == "__main__":
    main() 