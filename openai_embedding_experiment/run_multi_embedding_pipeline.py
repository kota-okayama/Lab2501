#!/usr/bin/env python3
"""
複数フィールドエンベディング生成からK近傍グラフ統合までの全パイプライン
"""

import os
import sys
import argparse
import subprocess
import json

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


def run_pipeline():
    """エンベディング生成とグラフ構築のパイプラインを実行するメイン関数"""
    parser = argparse.ArgumentParser(
        description="複数フィールドのエンベディング生成とK近傍グラフ構築・統合を行うパイプライン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必須引数
    parser.add_argument("--record_yaml_path", required=True, help="入力レコードのYAMLファイルパス")
    parser.add_argument("--output_base_dir", required=True, help="全ての出力ファイルのベースディレクトリ")
    
    # オプション引数
    parser.add_argument("--openai_model", default="text-embedding-ada-002", help="OpenAIエンベディングモデル")
    parser.add_argument("--api_batch_size", type=int, default=50, help="エンベディング生成時のAPIバッチサイズ")
    parser.add_argument(
        "--embedding_combinations",
        type=str,
        default="full",
        help="生成するエンベディングの組み合わせをセミコロンで区切って指定 (例: 'full;title;[title,author]')。デフォルトは 'full'。"
    )
    parser.add_argument("--k_neighbors", type=int, default=15, help="K近傍のK値")
    
    # 実行制御
    parser.add_argument("--skip_embedding", action="store_true", help="エンベディング生成をスキップ")
    parser.add_argument("--skip_graph_building", action="store_true", help="グラフ構築と統合をスキップ")

    args = parser.parse_args()

    # 出力ディレクトリの準備
    embeddings_dir = os.path.join(args.output_base_dir, "embeddings")
    graphs_dir = os.path.join(args.output_base_dir, "graphs")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    print(f"出力ベースディレクトリ: {args.output_base_dir}")
    print(f"エンベディング出力: {embeddings_dir}")
    print(f"グラフ出力: {graphs_dir}")

    # --- Step 1: 複数フィールドのエンベディングを生成 ---
    if not args.skip_embedding:
        print("\n" + "="*60)
        print("実行中: 複数フィールドエンベディング生成")
        vectorize_script = os.path.join(PROJECT_ROOT, "openai_embedding_experiment", "vectorize_multi_field_openai.py")
        vectorize_command = [
            sys.executable, "-u", vectorize_script,
            "--record_yaml_path", args.record_yaml_path,
            "--output_base_dir", embeddings_dir,
            "--openai_model", args.openai_model,
            "--api_batch_size", str(args.api_batch_size),
            "--embedding_combinations", args.embedding_combinations
        ]
        print(f"コマンド: {' '.join(vectorize_command)}")
        print("="*60)
        
        result = subprocess.run(vectorize_command)
        if result.returncode != 0:
            print("エンベディング生成に失敗しました。処理を中断します。")
            sys.exit(1)
    else:
        print("\nエンベディング生成をスキップしました。")


    # --- Step 2: K近傍グラフの構築と統合 ---
    embedding_summary_path = os.path.join(embeddings_dir, "embedding_summary.json")
    if not os.path.exists(embedding_summary_path):
        print(f"エラー: エンベディングサマリーファイルが見つかりません: {embedding_summary_path}")
        print("エンベディング生成ステップを先に実行してください。")
        sys.exit(1)

    if not args.skip_graph_building:
        print("\n" + "="*60)
        print("実行中: K近傍グラフ構築と統合")
        graph_script = os.path.join(PROJECT_ROOT, "openai_embedding_experiment", "build_multi_knn_graph_openai.py")
        graph_command = [
            sys.executable, "-u", graph_script,
            "--embedding_summary_path", embedding_summary_path,
            "--k_neighbors", str(args.k_neighbors),
            "--output_dir", graphs_dir
        ]
        print(f"コマンド: {' '.join(graph_command)}")
        print("="*60)

        result = subprocess.run(graph_command)
        if result.returncode != 0:
            print("グラフ構築と統合に失敗しました。処理を中断します。")
            sys.exit(1)
    else:
        print("\nグラフ構築と統合をスキップしました。")


    # --- 完了 ---
    print("\n" + "="*60)
    print("パイプライン実行完了")
    print("="*60)
    
    summary = json.load(open(embedding_summary_path))
    
    print(f"出力ディレクトリ: {args.output_base_dir}")
    print(f"生成されたエンベディングファイル: {len(summary)}個")
    print("エンベディング詳細:")
    for s in summary:
        print(f"  - {s['name']}: {s['record_count']} records")

    print(f"個別グラフファイル: {len(summary)}個 (in {graphs_dir})")
    
    merged_graph_path = os.path.join(graphs_dir, f"merged_knn_graph_k{args.k_neighbors}.json")
    print(f"ブロッキング用統合グラフ: {merged_graph_path}")


if __name__ == "__main__":
    run_pipeline()