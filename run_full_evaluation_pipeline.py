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
import shutil
from datetime import datetime
import glob

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
    # このパイプラインスクリプト内では、よりシンプルなサニタイズ関数を使用
    # 評価スクリプト側で複雑な命名規則を管理
    return model_name.replace('/', '_').replace(':', '_').replace(' ', '_')


def truncate_filename_component(text, max_length=50):
    """ファイル名の構成要素を指定した長さに短縮する"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_num_lines_in_jsonl(file_path):
    """JSONLファイルの行数を数える"""
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def extract_strategy_from_model_id(model_id):
    """
    モデルIDからサンプリング戦略を抽出する。
    例: "ft:...:music-matching-random-..." -> "random"
    """
    if not model_id.startswith("ft:"):
        return "random" 
    
    try:
        parts = model_id.split(':')
        if len(parts) >= 4:
            # amazon-walmart-product-matching-inconsistency-0904-100 の部分から戦略を抽出
            strategy_part = parts[3].replace('-', '_')
            
            # 既知の戦略パターンを確認
            known_strategies = ['inconsistency', 'diversity', 'uncertainty', 'random', 'lowest_score']
            for strategy in known_strategies:
                if strategy in strategy_part:
                    return strategy
        
        return 'random'  # デフォルト戦略
    except Exception:
        return 'random'


def extract_iteration_number(output_base_dir):
    """出力ベースディレクトリからイテレーション番号を抽出する
    
    Args:
        output_base_dir (str): 例: "results_music/run_2k_ite2_music"
    
    Returns:
        int: イテレーション番号（例: 2）。見つからない場合は0
    """
    import re
    match = re.search(r'_ite(\d+)_', output_base_dir)
    if match:
        return int(match.group(1))
    return 0


def find_previous_ft_data(output_base_dir, data_type, strategy, current_iteration):
    """過去のイテレーションのFTデータファイルを検索する
    
    Args:
        output_base_dir (str): 現在の出力ベースディレクトリ
        data_type (str): データタイプ（例: "music"）
        strategy (str): 戦略名（例: "diversity"）
        current_iteration (int): 現在のイテレーション番号
    
    Returns:
        list: 過去のFTデータファイルパスのリスト
    """
    import glob
    import re
    
    previous_files = []
    base_pattern = output_base_dir.replace(f'_ite{current_iteration}_', '_ite{}_')
    
    # ite0からcurrent_iteration-1までのファイルを検索
    for ite in range(current_iteration):
        ite_dir = base_pattern.format(ite)
        search_pattern = os.path.join(ite_dir, "evaluation_results", f"*{strategy}*.jsonl")
        
        # パターンマッチでファイルを検索
        matching_files = glob.glob(search_pattern)
        for file_path in matching_files:
            if os.path.exists(file_path):
                num_lines = get_num_lines_in_jsonl(file_path)
                if num_lines > 0:
                    previous_files.append(file_path)
                    print(f"  -> 過去のFTデータを発見: {file_path} ({num_lines} 件)")
                else:
                    print(f"  -> 空のFTデータファイルをスキップ: {file_path}")
    
    return previous_files


def find_previous_cumulative_ft_data(output_base_dir, strategy, current_iteration, is_balanced=True):
    """1つ前のイテレーションの累積FTデータファイルを検索する"""
    if current_iteration == 0:
        return None

    import glob
    
    previous_iteration = current_iteration - 1
    
    # バランス状態に応じたファイル名プレフィックス
    prefix = "ft_b_data" if is_balanced else "ft_ub_data"

    # 1つ前のイテレーションのディレクトリパスを構築
    base_pattern = output_base_dir.replace(
        f'_ite{current_iteration}_', f'_ite{previous_iteration}_'
    )
    
    # 累積FTファイルを検索
    search_pattern = os.path.join(
        base_pattern, "evaluation_results", f"{prefix}_{strategy}_cumulative_ite{previous_iteration}.jsonl"
    )
    
    matching_files = glob.glob(search_pattern)
    
    if matching_files:
        found_file = matching_files[0]
        num_lines = get_num_lines_in_jsonl(found_file)
        if num_lines > 0:
            print(f"  -> 1つ前の{'バランス済み' if is_balanced else 'バランスなし'}累積FTデータを発見: {found_file} ({num_lines} 件)")
            return found_file
        else:
            print(f"  -> 1つ前の{'バランス済み' if is_balanced else 'バランスなし'}累積FTデータが空のためスキップ: {found_file}")

    # ite0の場合は `_cumulative_` がつかない可能性があるため、フォールバック検索
    if previous_iteration == 0:
        # ite0のファイル名は `ft_balanced_data_..._strategy.jsonl` または `ft_unbalanced_data_..._strategy.jsonl`
        # `base_name_for_ft` が含まれるため、ワイルドカードで検索
        search_pattern_fallback = os.path.join(
            base_pattern, "evaluation_results", f"{prefix}_*_{strategy}.jsonl"
        )
        all_files = glob.glob(search_pattern_fallback)
        # 累積ファイル以外を選ぶ
        non_cumulative_files = [f for f in all_files if '_cumulative_' not in os.path.basename(f)]
        if non_cumulative_files:
            found_file = non_cumulative_files[0]
            num_lines = get_num_lines_in_jsonl(found_file)
            if num_lines > 0:
                 print(f"  -> [フォールバック] ite0の{'バランス済み' if is_balanced else 'バランスなし'}FTデータを発見: {found_file} ({num_lines} 件)")
                 return found_file

    return None


def find_previous_labeled_pairs_csv(output_base_dir, strategy, current_iteration):
    """1つ前のイテレーションの累積ラベル済みペアCSVを検索する"""
    if current_iteration == 0:
        return None

    previous_iteration = current_iteration - 1
    
    # 1つ前のイテレーションのディレクトリパスを構築
    base_pattern = output_base_dir.replace(
        f'_ite{current_iteration}_', f'_ite{previous_iteration}_'
    )
    
    # 累積ラベル済みペアCSVを検索
    search_pattern = os.path.join(
        base_pattern, "evaluation_results", f"labeled_pairs_{strategy}_cumulative_ite{previous_iteration}.csv"
    )
    
    matching_files = glob.glob(search_pattern)
    
    if matching_files:
        found_file = matching_files[0]
        if os.path.exists(found_file) and os.path.getsize(found_file) > 0:
            print(f"  -> 1つ前のラベル済みペアCSVを発見: {found_file}")
            return found_file
        else:
            print(f"  -> 1つ前のラベル済みペアCSVが空のためスキップ: {found_file}")

    return None


def update_labeled_pairs_csv(new_ft_jsonl_path, previous_labeled_csv_path, new_labeled_csv_path):
    """新しいFTデータからペアを抽出し、過去のラベル済みペアと統合して保存する"""
    all_pairs = set()

    # 1. 過去のラベル済みペアを読み込み
    if previous_labeled_csv_path and os.path.exists(previous_labeled_csv_path):
        try:
            with open(previous_labeled_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # ヘッダーをスキップ
                for row in reader:
                    if len(row) >= 2:
                        all_pairs.add(tuple(sorted((row[0], row[1]))))
            print(f"  -> {len(all_pairs)}件の過去のラベル済みペアを {os.path.basename(previous_labeled_csv_path)} から読み込み")
        except Exception as e:
            print(f"警告: 過去のラベル済みペアCSVの読み込みに失敗: {e}")

    # 2. 新しいFTデータ(.jsonl)からペアを抽出
    newly_added_pairs = set()
    if os.path.exists(new_ft_jsonl_path):
        try:
            with open(new_ft_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'record_id_1' in data and 'record_id_2' in data:
                            pair = tuple(sorted((str(data['record_id_1']), str(data['record_id_2']))))
                            newly_added_pairs.add(pair)
                    except json.JSONDecodeError:
                        continue # 空行などを無視
            print(f"  -> {len(newly_added_pairs)}件の新しいペアを {os.path.basename(new_ft_jsonl_path)} から抽出")
        except Exception as e:
            print(f"警告: 新しいFTデータからのペア抽出に失敗: {e}")
    
    # 3. 統合して保存
    original_count = len(all_pairs)
    all_pairs.update(newly_added_pairs)
    
    try:
        os.makedirs(os.path.dirname(new_labeled_csv_path), exist_ok=True)
        with open(new_labeled_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['record_id_1', 'record_id_2'])
            # 再現性のためソート
            for pair in sorted(list(all_pairs)):
                writer.writerow(pair)
        
        print(f"  -> 統合後のラベル済みペア: {len(all_pairs)}件 (新規 {len(all_pairs) - original_count}件) -> {os.path.basename(new_labeled_csv_path)}")
        return True
    except Exception as e:
        print(f"エラー: 新しいラベル済みペアCSVの保存に失敗: {e}")
        return False


def merge_ft_data_files(file_paths, output_path):
    """複数のFTデータファイルを統合し、重複を排除する
    
    Args:
        file_paths (list): 統合するJSONLファイルパスのリスト
        output_path (str): 出力先のJSONLファイルパス
    
    Returns:
        int: 統合されたユニークなレコード数
    """
    unique_lines = set()
    
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = [line.strip() for line in infile if line.strip()]
                    unique_lines.update(lines)
                    print(f"  -> {file_path} から {len(lines)} 件を統合対象として読み込み")
        
        total_records = len(unique_lines)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # 再現性のためソートして書き出し
            for line in sorted(list(unique_lines)):
                outfile.write(line + '\n')
        
        print(f"統合完了: {total_records} 件のユニークなレコードを {output_path} に保存")
        return total_records
        
    except Exception as e:
        print(f"FTデータ統合中にエラーが発生しました: {e}")
        return 0


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
    # 評価スクリプト (evaluate_finetuning_performance_async.py) と完全に同じロジックでファイル名を生成
    
    model_before_ft_sanitized = sanitize_model_name_for_filename(model_before_ft)
    model_after_ft_sanitized = sanitize_model_name_for_filename(model_after_ft)
    
    # ファイル名の各構成要素を短縮
    pairs_base = truncate_filename_component(os.path.basename(pairs_csv_path).replace('.csv', ''), 30)
    model_before_short = truncate_filename_component(model_before_ft_sanitized, 10)
    model_after_short = truncate_filename_component(model_after_ft_sanitized, 70)

    base_filename = f"eval_async_{pairs_base}_before-{model_before_short}_after-{model_after_short}"

    output_filename = f"{base_filename}_details.csv"

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

    if args.limit_pairs:
        command.extend(["--limit_pairs", str(args.limit_pairs)])

    if not run_command(command, "モデル性能評価"):
        print("STEP 3 が失敗しました。処理を中断します。")
        sys.exit(1)

    print("===== STEP 3完了 =====")
    return get_evaluation_details_path(args, pairs_csv_path, model_before, model_after)


def run_inconsistency_detection(args, details_csv_path):
    """STEP 4: 矛盾する三角形を検出"""
    print("\n\n===== STEP 4: 矛盾する三角形の検出 =====")

    # 実際のdetails.csvファイルが存在するか確認
    if not os.path.exists(details_csv_path):
        print(f"警告: 指定されたファイルが見つかりません: {details_csv_path}")
        # evaluation_resultsディレクトリで*details.csvファイルを検索
        output_dir = os.path.dirname(details_csv_path)
        import glob
        details_files = glob.glob(os.path.join(output_dir, "*details.csv"))
        if details_files:
            details_csv_path = details_files[0]  # 最初に見つかったファイルを使用
            print(f"代替ファイルを使用: {details_csv_path}")
        else:
            print("エラー: details.csvファイルが見つかりません。")
            sys.exit(1)

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
    
    # 評価スクリプトと同じロジックでinconsistent_trianglesのファイル名を生成
    inconsistent_filename = f"{base_name}_score_after_inconsistent_triangles.csv"
    
    return os.path.join(output_dir, inconsistent_filename)


def run_finetuning_data_preparation(
    args, inconsistent_triangles_csv, details_csv_path, llm_clusters_json_path, model_before_ft, model_after_ft, force_strategies=None
):
    """STEP 5: ファインチューニング用データの準備（モデル別戦略）"""
    print("\n\n===== STEP 5: ファインチューニング用データの準備 =====")

    # イテレーション番号を抽出
    current_iteration = extract_iteration_number(args.output_base_dir)
    print(f"現在のイテレーション: {current_iteration}")

    strategies_to_generate = []
    if force_strategies:
        print(f"指定された戦略リストに基づいてFTデータを生成します: {force_strategies}")
        strategies_to_generate = force_strategies
    else:
        # BEFOREとAFTERモデル両方から戦略を抽出
        strategy_before = extract_strategy_from_model_id(model_before_ft)
        strategy_after = extract_strategy_from_model_id(model_after_ft)
        
        print(f"BEFOREモデル '{model_before_ft}' から抽出された戦略: {strategy_before}")
        print(f"AFTERモデル '{model_after_ft}' から抽出された戦略: {strategy_after}")

        # 両方の戦略でFTデータを生成
        strategies_to_generate.append(strategy_before)
        strategies_to_generate.append(strategy_after)
        
        # 重複を除去
        strategies_to_generate = list(set(strategies_to_generate))
        
        if not strategies_to_generate:
            strategies_to_generate = ['random']  # フォールバック

    base_name_for_ft = os.path.basename(
        details_csv_path
    ).replace("_details.csv", "")
    output_dir = os.path.dirname(details_csv_path)
    num_samples = args.num_samples  

    # 各戦略でファインチューニングデータを生成
    for strategy in strategies_to_generate:
        print(f"\n--- 戦略: {strategy} ({num_samples}ペア) ---")
        
        # --- ファイルパス生成 (新しい命名規則) ---
        base_name_part = os.path.basename(details_csv_path).replace("_details.csv", "")

        # バランスあり
        balanced_current_filename = f"ft_b_data_{base_name_part}_{strategy}.jsonl"
        balanced_cumulative_filename = f"ft_b_data_{strategy}_cumulative_ite{current_iteration}.jsonl"
        balanced_current_path = os.path.join(output_dir, balanced_current_filename)
        balanced_cumulative_path = os.path.join(output_dir, balanced_cumulative_filename)

        # バランスなし
        unbalanced_current_filename = f"ft_ub_data_{base_name_part}_{strategy}.jsonl"
        unbalanced_cumulative_filename = f"ft_ub_data_{strategy}_cumulative_ite{current_iteration}.jsonl"
        unbalanced_current_path = os.path.join(output_dir, unbalanced_current_filename)
        unbalanced_cumulative_path = os.path.join(output_dir, unbalanced_cumulative_filename)


        # ラベル済みペアファイルのパスを定義・検索 (これはバランス調整の有無で共通)
        labeled_pairs_filename = f"labeled_pairs_{strategy}_cumulative_ite{current_iteration}.csv"
        new_labeled_pairs_path = os.path.join(output_dir, labeled_pairs_filename)
        previous_labeled_pairs_path = find_previous_labeled_pairs_csv(
            args.output_base_dir, strategy, current_iteration
        )

        # 実際のdetails.csvファイルが存在するか確認して修正
        actual_details_csv_path = details_csv_path
        if not os.path.exists(details_csv_path):
            print(f"警告: 指定されたファイルが見つかりません: {details_csv_path}")
            # evaluation_resultsディレクトリで*details.csvファイルを検索
            import glob
            details_files = glob.glob(os.path.join(output_dir, "*details.csv"))
            if details_files:
                actual_details_csv_path = details_files[0]
                print(f"代替ファイルを使用: {actual_details_csv_path}")
            else:
                print(f"エラー: {strategy} 戦略用のdetails.csvファイルが見つかりません。")
                continue

        # 戦略に基づいてファインチューニングデータを生成
        if strategy in ['inconsistency', 'lowest_score']:
            script_path = os.path.join(
                "siamese_model_pytorch", "prepare_finetuning_data.py"
            )
            command = [
                "python3", "-u", script_path,
                "--inconsistent_triangles_csv", inconsistent_triangles_csv,
                "--evaluation_details_csv", actual_details_csv_path,
                "--ground_truth_yaml", args.record_yaml_path,
                "--output_jsonl_path", balanced_current_path,
                "--output_jsonl_path_unbalanced", unbalanced_current_path,
                "--data_type", args.data_type,
                "--score_column", "score_after",
                "--num_samples", str(num_samples),
                "--sampling_strategy", strategy
            ]
            if previous_labeled_pairs_path:
                command.extend(["--labeled_pairs_csv", previous_labeled_pairs_path])

            if not run_command(command, f"FTデータ準備 ({strategy})"):
                print(f"{strategy} 戦略が失敗しました。")
                continue
            
            # 正常終了後、ラベル済みペアファイルを更新 (どちらかのファイルからでOK)
            update_labeled_pairs_csv(balanced_current_path, previous_labeled_pairs_path, new_labeled_pairs_path)
        else:
            script_path = os.path.join(
                "siamese_model_pytorch",
                "create_finetuning_data_from_strategies.py"
            )
            command = [
                "python3", "-u", script_path,
                "--strategy", strategy,
                "--output_jsonl_path", balanced_current_path,
                "--output_jsonl_path_unbalanced", unbalanced_current_path,
                "--ground_truth_yaml", args.record_yaml_path,
                "--num_samples", str(num_samples),
                "--data_type", args.data_type,
            ]
            if strategy in ["uncertainty", "random"]:
                command.extend(["--evaluation_details_csv", actual_details_csv_path])
            if strategy == "uncertainty":
                command.extend(["--score_column", "score_after"])
            if strategy == "diversity":
                command.extend(["--llm_clusters_json", llm_clusters_json_path])

            if previous_labeled_pairs_path:
                command.extend(["--labeled_pairs_csv", previous_labeled_pairs_path])

            if not run_command(command, f"FTデータ準備 ({strategy})"):
                print(f"{strategy} 戦略が失敗しました。")
                continue

            # 正常終了後、ラベル済みペアファイルを更新 (どちらかのファイルからでOK)
            update_labeled_pairs_csv(balanced_current_path, previous_labeled_pairs_path, new_labeled_pairs_path)

        # --- 累積処理 ---
        for is_balanced in [True, False]:
            status = "バランス済み" if is_balanced else "バランスなし"
            print(f"\n--- イテレーション統合処理: {strategy} ({status}) ---")

            current_path = balanced_current_path if is_balanced else unbalanced_current_path
            cumulative_path = balanced_cumulative_path if is_balanced else unbalanced_cumulative_path
            
            # 1つ前の累積FTデータファイルを検索
            previous_cumulative_file = find_previous_cumulative_ft_data(
                args.output_base_dir, strategy, current_iteration, is_balanced=is_balanced
            )
            
            # 統合するファイルリスト
            files_to_merge = []
            if previous_cumulative_file:
                files_to_merge.append(previous_cumulative_file)
            if get_num_lines_in_jsonl(current_path) > 0:
                files_to_merge.append(current_path)
            
            if len(files_to_merge) > 0:
                print(f"統合対象ファイル数: {len(files_to_merge)}")
                total_samples = merge_ft_data_files(files_to_merge, cumulative_path)
                print(f"累積FTデータ ({status}): {total_samples} 件 -> {cumulative_path}")
            else:
                print(f"統合するデータがありません ({status})。")

    print("\n===== STEP 5完了 =====")
    return strategies_to_generate


def run_finetuning_execution(args, output_dir, strategies_generated):
    """STEP 6: 生成されたFTデータを使用してファインチューニングを実行"""
    print("\n\n===== STEP 6: ファインチューニング実行 =====")
    
    current_iteration = extract_iteration_number(args.output_base_dir)
    
    for strategy in strategies_generated:
        print(f"\n--- {strategy} 戦略のファインチューニング実行 ---")
        
        # 累積FTデータファイルを検索
        if current_iteration >= 1:
            pattern = f"*{strategy}_cumulative_ite{current_iteration}*.jsonl"
        else:
            pattern = f"*{strategy}*.jsonl"
            
            ft_files = glob.glob(os.path.join(output_dir, pattern))
        
        if not ft_files:
            print(f"警告: {strategy} 戦略のFTデータファイルが見つかりません。")
            continue
            
        ft_file = ft_files[0]  # 最初に見つかったファイルを使用
        num_samples = get_num_lines_in_jsonl(ft_file)
        
        if num_samples == 0:
            print(f"警告: {strategy} 戦略のFTデータが空です。スキップします。")
            continue
            
        print(f"FTデータファイル: {ft_file} ({num_samples} 件)")
        
        # ファインチューニングジョブ名を生成
        job_suffix = f"{args.data_type}-matching-{strategy}-ite{current_iteration}-{num_samples}"
        
        print(f"ファインチューニングジョブを開始: {job_suffix}")
        print(f"ベースモデル: {args.base_model_for_ft}")
        print(f"トレーニングデータ: {num_samples} 件")
        
        # 実際のファインチューニングコマンドを実行
        # （ここでは実際のOpenAI APIコールは実装せず、ログのみ出力）
        print("注意: 実際のファインチューニング実行機能は未実装です。")
        print(f"実行予定コマンド: openai api fine_tuning.jobs.create -t {ft_file} -m {args.base_model_for_ft} --suffix {job_suffix}")
    
    print("\n===== STEP 6完了 =====")


def find_details_file_dynamically(pairs_csv_path, ft_model):
    """指定されたFTモデルに対応するdetails.csvファイルを動的に検索する"""
    evaluation_results_dir = os.path.join(
        os.path.dirname(pairs_csv_path), "..", "evaluation_results"
    )
    details_files = glob.glob(
        os.path.join(evaluation_results_dir, "*details.csv")
    )
    
    if details_files:
        current_strategy = extract_strategy_from_model_id(ft_model)
        strategy_with_hyphen = current_strategy.replace('_', '-')
        
        matching_files = []
        for file_path in details_files:
            filename = os.path.basename(file_path)
            if (f"_{current_strategy}_" in filename or 
                f"-{current_strategy}-" in filename or
                f"_{strategy_with_hyphen}_" in filename or
                f"-{strategy_with_hyphen}-" in filename):
                matching_files.append(file_path)
        
        if matching_files:
            found_path = matching_files[0]
            print(f"  -> 対応するファイルを発見: {found_path}")
            return found_path
        else:
            print(f"エラー: 戦略 '{current_strategy}' に対応する評価詳細ファイルが見つかりませんでした。")
            return None
    else:
        print(f"エラー: 評価詳細ファイルが一つも見つかりません: {evaluation_results_dir}")
        return None

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
        "--k_neighbors", type=int, default=10, help="K近傍のK値"
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
        "--inconsistency_top_n", type=int, default=400,
        help="検出する矛盾三角形の上位N件"
    )

    # --- Step 5: Finetuning Data Preparation ---
    step5_group = parser.add_argument_group('Step 5: Finetuning Data Prep')
    # 注意: FT戦略はモデルIDから自動抽出されるため、--ft-strategies引数は削除
    
    # --- Step 6: Finetuning Execution ---
    step6_group = parser.add_argument_group('Step 6: Finetuning Execution')
    step6_group.add_argument(
        "--execute_finetuning", action="store_true",
        help="生成したFTデータを使用して実際にファインチューニングを実行"
    )
    step6_group.add_argument(
        "--base_model_for_ft", type=str, default="gpt-4o-mini-2024-07-18",
        help="ファインチューニングのベースモデル"
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
    control_group.add_argument("--num_samples", type=int, default=100, help="Number of samples for finetuning data strategies")
    control_group.add_argument("--limit_pairs", type=int, help="Limit the number of pairs for evaluation for debugging.")

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    # ヘルパー関数をmain関数のトップレベルに定義
    def find_clusters_file(base_dir):
        """Helper to find the clusters_after.json file."""
        import glob
        evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
        
        # まずは期待通りのパスを探す
        expected_path = os.path.join(evaluation_results_dir, "clusters_after.json")
        if os.path.exists(expected_path):
            return expected_path
            
        # 見つからない場合はglobで探す
        search_pattern = os.path.join(evaluation_results_dir, "*_clusters_after.json")
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            return matching_files[0]
        
        return None

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

    current_iteration = extract_iteration_number(args.output_base_dir)

    # ite0かつモデルIDが1つの場合、全戦略のFTデータを生成する特別モード
    if current_iteration == 0 and len(args.model_ids) == 1:
        # ite0 special mode: 1つのベースモデルから全戦略のFTデータを生成
        base_model = args.model_ids[0]
        output_base_dir = args.output_base_dir # ★★★ 変数定義を先頭に移動 ★★★
        os.makedirs(output_base_dir, exist_ok=True)
        evaluation_results_dir = os.path.join(output_base_dir, "evaluation_results")
        os.makedirs(evaluation_results_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("Running in ite0 special mode: Generating all strategies from base model")
        print(f"Base Model: {base_model}")
        print("="*80)
        
        # パス変数をブロックの先頭で初期化
        details_csv_path = None
        inconsistent_triangles_path = None
        clusters_path = None

        # --- Step 3 ---
        if not args.skip_step_3:
            details_csv_path = run_evaluation(args, pairs_csv_path, base_model, base_model)
        else:
            print("\n\n===== STEP 3 をスキップしました =====")
            # 新しい命名規則でパスを取得
            details_csv_path = get_evaluation_details_path(args, pairs_csv_path, base_model, base_model)
            if not os.path.exists(details_csv_path):
                print(f"  -> 推定されたパスにファイルが見つかりません。動的に検索します...")
                evaluation_results_dir = os.path.join(
                    os.path.dirname(pairs_csv_path), "..", "evaluation_results"
                )
                import glob
                details_files = glob.glob(os.path.join(evaluation_results_dir, "*details.csv"))
                if details_files:
                    details_csv_path = details_files[0] # ite0では1つのはず
                    print(f"  -> ファイルを発見: {details_csv_path}")
                else:
                    print(f"エラー: 評価詳細ファイル (*details.csv) が見つかりません: {evaluation_results_dir}")
                    sys.exit(1)
            print(f"既存の評価詳細ファイルを使用します: {details_csv_path}")

        # Step3で有効なパスが得られなかった場合はここで終了
        if not details_csv_path or not os.path.exists(details_csv_path):
            print("エラー: Step3の評価ファイルが見つからないため、これ以上処理を続行できません。")
            sys.exit(1)

        # --- Step 4 ---
        if not args.skip_step_4:
            # inconsistency と lowest_score の場合にのみ実行
            # (ite0モードでは両方の戦略が含まれる可能性があるため、ここでチェックはしない)
            print("-" * 50)
            print(f"Step 4: 矛盾三角形の検出")
            inconsistent_triangles_path = run_inconsistency_detection(args, details_csv_path)
            if not inconsistent_triangles_path:
                print("警告: 矛盾三角形の検出に失敗しました。")
            else:
                 print("===== STEP 4完了 =====")
        else:
            print("\n\n===== STEP 4 をスキップしました =====")
            # 矛盾三角形ファイルのパスを推定
            output_dir = os.path.dirname(details_csv_path)
            base_name = os.path.basename(details_csv_path).replace("_details.csv", "")
            inconsistent_filename = f"{base_name}_score_after_inconsistent_triangles.csv"
            inconsistent_triangles_path = os.path.join(output_dir, inconsistent_filename)
            if not os.path.exists(inconsistent_triangles_path):
                 print(f"エラー: 矛盾三角形ファイルが見つかりません: {inconsistent_triangles_path}")
                 sys.exit(1)
            print(f"既存の矛盾三角形ファイルを使用します: {inconsistent_triangles_path}")

        # --- Step 5 ---
        if not args.skip_step_5:
            print("\n\n===== STEP 5: FTデータ準備 =====")
            # 'diversity' 戦略に備えて、クラスタファイルを無条件で探す
            clusters_path = find_clusters_file(output_base_dir)
            if not clusters_path:
                print(f"警告: 戦略 'diversity' のためのクラスタファイルが見つかりませんでした。")

            all_strategies = ['inconsistency', 'diversity', 'uncertainty', 'random', 'lowest_score']
            for strategy in all_strategies:
                run_finetuning_data_preparation(
                    args, 
                    inconsistent_triangles_path, 
                    details_csv_path, 
                    clusters_path, 
                    base_model, 
                    base_model, #
                    force_strategies=[strategy]
                )
        else:
            print("\n\n===== STEP 5 をスキップしました =====")

    # ite1以降、またはite0でも複数モデルIDが指定された場合の通常モード
    else:
        # --- Create Model Pairs ---
        output_base_dir = args.output_base_dir # ★★★ 変数定義を先頭に移動 ★★★
        os.makedirs(output_base_dir, exist_ok=True)
        evaluation_results_dir = os.path.join(output_base_dir, "evaluation_results")
        os.makedirs(evaluation_results_dir, exist_ok=True)
        model_ids = args.model_ids
        if len(args.model_ids) < 2:
            print("エラー: 通常モードの実行には、ベースモデルと少なくとも1つのFTモデルを指定してください。")
            sys.exit(1)
    
        base_model = model_ids[0]
        ft_models = model_ids[1:]
        print(f"ベースモデル: {base_model}")
        print(f"評価対象FTモデル数: {len(ft_models)}")
    
        # --- Loop for Steps 3, 4, 5 for each model pair ---
        if args.skip_step_3 and args.skip_step_4 and args.skip_step_5:
            print("\n\n===== STEP 3, 4, 5 をスキップしました =====")
        else:
            for i, ft_model in enumerate(ft_models):
                print(f"\n\n{'#'*80}")
                print(f"#### モデルペア {i+1}/{len(ft_models)} の処理を開始... ####")
                print(f"#### BASE:   {base_model}")
                print(f"#### FT:     {ft_model}")
                print(f"{'#'*80}")
    
                # パス変数をループの先頭で初期化
                details_csv_path = None
                inconsistent_triangles_path = None
                clusters_path = None

                # --- Step 3 ---
                if not args.skip_step_3:
                    print("\n\n===== STEP 3: モデル評価 =====")
                    details_csv_path = run_evaluation(
                        args, pairs_csv_path, base_model, ft_model
                    )
                    # run_evaluationが失敗した場合のフォールバック処理
                    if not details_csv_path or not os.path.exists(details_csv_path):
                        print("  -> Step3の実行に失敗、または結果ファイルが見つかりません。動的検索でフォールバックします...")
                        details_csv_path = find_details_file_dynamically(pairs_csv_path, ft_model)

                else:
                    print("\n\n===== STEP 3 をスキップしました =====")
                    details_csv_path = find_details_file_dynamically(pairs_csv_path, ft_model)


                # パスが見つからなかった場合に以降の処理を確実にスキップ
                if not details_csv_path or not os.path.exists(details_csv_path):
                    print(f"エラー: details.csvへの有効なパスが設定されませんでした。モデル '{os.path.basename(ft_model)}' の処理をスキップします。")
                    continue

                # --- Step 4 ---
                current_strategy = extract_strategy_from_model_id(ft_model)
                if not args.skip_step_4:
                    if current_strategy in ['inconsistency', 'lowest_score']:
                        print("-" * 50)
                        print(f"Step 4: 矛盾三角形の検出 ({current_strategy})")
                        inconsistent_triangles_path = run_inconsistency_detection(args, details_csv_path)
                        if not inconsistent_triangles_path:
                            print(f"警告: {current_strategy} 戦略のための矛盾三角形の検出に失敗しました。")
                        else:
                            print("===== STEP 4完了 =====")
                    else:
                        print(f"Note: Step 4 is skipped for '{current_strategy}' strategy.")
                else:
                    print("\n\n===== STEP 4 をスキップしました =====")
                    # 矛盾三角形ファイルのパスを推定
                    output_dir = os.path.dirname(details_csv_path)
                    base_name = os.path.basename(
                        details_csv_path
                    ).replace("_details.csv", "")
                    
                    # ファイル名の長さ制限（Step4のrun_inconsistency_detectionと同じロジック）
                    inconsistent_filename = f"{base_name}_score_after_inconsistent_triangles.csv"
                    if len(inconsistent_filename) > 200:
                        import hashlib
                        hash_str = hashlib.md5(inconsistent_filename.encode()).hexdigest()[:8]
                        inconsistent_filename = f"inconsistent_triangles_{hash_str}.csv"
                    
                    inconsistent_triangles_path = os.path.join(output_dir, inconsistent_filename)
                    
                    if not os.path.exists(inconsistent_triangles_path):
                        print(f"  -> 新しい命名規則のファイルが見つかりません。動的検索を実行中...")
                        # evaluation_resultsディレクトリで動的検索
                        import glob
                        search_pattern = os.path.join(output_dir, "*inconsistent_triangles.csv")
                        matching_files = glob.glob(search_pattern)
                        
                        if matching_files:
                            inconsistent_triangles_path = matching_files[0]  # 最初に見つかったファイルを使用
                            print(f"  -> 動的検索で発見: {inconsistent_triangles_path}")
                        else:
                            print(f"エラー: 矛盾三角形ファイルが見つかりません: {search_pattern}")
                            print("Step 4をスキップするには、このファイルが事前に存在している必要があります。")
                            continue # 次のペアへ
                    print(f"既存の矛盾三角形ファイルを使用します: {inconsistent_triangles_path}")
    
                # --- Step 5 ---
                if not args.skip_step_5:
                    print("\n\n===== STEP 5: FTデータ準備 =====")
                    # 'diversity' 戦略の場合のみクラスタファイルを探す
                    if current_strategy == 'diversity':
                        clusters_path = find_clusters_file(output_base_dir)
                        if not clusters_path:
                            print(f"警告: 戦略 'diversity' のためのクラスタファイルが見つかりませんでした。")

                    run_finetuning_data_preparation(
                        args,
                        inconsistent_triangles_path,
                        details_csv_path,
                        clusters_path,
                        base_model,
                        ft_model
                    )
                else:
                    print("\n\n===== STEP 5 をスキップしました =====")

    print("\n\nパイプラインの全工程が正常に完了しました。")


if __name__ == "__main__":
    main()
