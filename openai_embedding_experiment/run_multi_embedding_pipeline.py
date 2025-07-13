#!/usr/bin/env python3
"""
複数フィールドエンベディング生成からK近傍グラフ統合までの全パイプライン
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


def run_command(command, description):
    """コマンドを実行して結果をリアルタイムで表示"""
    print(f"\n{'='*60}")
    print(f"実行中: {description}")
    print(f"コマンド: {' '.join(command)}")
    print('='*60)
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1  # 行単位のバッファリング
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


def main():
    parser = argparse.ArgumentParser(
        description="多フィールドエンベディング→K近傍グラフ統合の全パイプライン実行"
    )
    
    # 入力ファイル
    parser.add_argument(
        "--record_yaml_path", type=str, required=True,
        help="入力YAMLファイルのパス"
    )
    
    # 出力ディレクトリ
    parser.add_argument(
        "--output_base_dir", type=str, required=True,
        help="出力ベースディレクトリ"
    )
    
    # エンベディング設定
    parser.add_argument(
        "--openai_model", type=str, default="text-embedding-ada-002",
        help="OpenAI エンベディングモデル"
    )
    parser.add_argument(
        "--api_batch_size", type=int, default=50,
        help="API バッチサイズ"
    )
    
    # フィールド組み合わせ選択
    parser.add_argument(
        "--selected_combinations", type=str, default="",
        help="使用するフィールド組み合わせ (例: 'full,title_only,author_only')"
    )
    
    # K近傍設定
    parser.add_argument(
        "--k_neighbors", type=int, default=15,
        help="K近傍のK値"
    )
    
    # 実行制御
    parser.add_argument(
        "--skip_embedding", action="store_true",
        help="エンベディング生成をスキップ (既存ファイルを使用)"
    )
    parser.add_argument(
        "--skip_graph_building", action="store_true",
        help="グラフ構築をスキップ"
    )

    args = parser.parse_args()
    
    # 出力ディレクトリの準備
    output_base = Path(args.output_base_dir)
    embeddings_dir = output_base / "embeddings"
    graphs_dir = output_base / "graphs"
    
    output_base.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)
    
    print(f"出力ベースディレクトリ: {output_base}")
    print(f"エンベディング出力: {embeddings_dir}")
    print(f"グラフ出力: {graphs_dir}")
    
    # スクリプトパス
    current_dir = Path(__file__).parent
    multi_embed_script = current_dir / "vectorize_multi_field_openai.py"
    multi_graph_script = current_dir / "build_multi_knn_graph_openai.py"
    
    success = True
    
    # ステップ1: 複数フィールドエンベディング生成
    if not args.skip_embedding:
        embed_command = [
            sys.executable,
            "-u", # 出力バッファリングを無効化
            str(multi_embed_script),
            "--record_yaml_path", args.record_yaml_path,
            "--output_embeddings_path", str(embeddings_dir / "embeddings_template.npy"),
            "--openai_model", args.openai_model,
            "--api_batch_size", str(args.api_batch_size)
        ]
        
        if args.selected_combinations:
            embed_command.extend(["--selected_combinations", args.selected_combinations])
        
        success = run_command(embed_command, "複数フィールドエンベディング生成")
        
        if not success:
            print("エンベディング生成に失敗しました。処理を中断します。")
            return
    else:
        print("エンベディング生成をスキップしました")
    
    # エンベディングサマリーファイルの確認
    summary_file = embeddings_dir / "embedding_summary.json"
    if not summary_file.exists():
        print(f"エラー: エンベディングサマリーファイルが見つかりません: {summary_file}")
        return
    
    # ステップ2: K近傍グラフ構築と統合
    if not args.skip_graph_building:
        graph_command = [
            sys.executable,
            "-u", # 出力バッファリングを無効化
            str(multi_graph_script),
            "--embedding_summary_path", str(summary_file),
            "--k_neighbors", str(args.k_neighbors),
            "--output_dir", str(graphs_dir)
        ]
        
        if args.selected_combinations:
            graph_command.extend(["--selected_combinations", args.selected_combinations])
        
        success = run_command(graph_command, "K近傍グラフ構築と統合")
        
        if not success:
            print("グラフ構築に失敗しました。")
            return
    else:
        print("グラフ構築をスキップしました")
    
    # 結果サマリーの表示
    print(f"\n{'='*60}")
    print("パイプライン実行完了")
    print('='*60)
    
    print(f"出力ディレクトリ: {output_base}")
    
    if embeddings_dir.exists():
        embed_files = list(embeddings_dir.glob("embeddings_*.npy"))
        print(f"生成されたエンベディングファイル: {len(embed_files)}個")
        
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            print("エンベディング詳細:")
            for item in summary:
                print(f"  - {item['name']}: {item['record_count']}件 "
                     f"(次元: {item['dimension']})")
    
    if graphs_dir.exists():
        graph_files = list(graphs_dir.glob("knn_graph_*.json"))
        merged_files = list(graphs_dir.glob("merged_knn_graph_*.json"))
        print(f"個別グラフファイル: {len(graph_files)}個")
        print(f"統合グラフファイル: {len(merged_files)}個")
        
        # 統合サマリーがあれば表示
        merge_summary_file = graphs_dir / f"merge_summary_k{args.k_neighbors}.json"
        if merge_summary_file.exists():
            with open(merge_summary_file, 'r', encoding='utf-8') as f:
                merge_summary = json.load(f)
            print("グラフ統合詳細:")
            print(f"  - 統合前グラフ数: {len(merge_summary['source_combinations'])}")
            print(f"  - 統合後ノード数: {merge_summary['merged_graph']['nodes']}")
            print(f"  - 統合後エッジ数: {merge_summary['merged_graph']['edges']}")
    
    print(f"\nブロッキング用統合グラフ: {graphs_dir / f'merged_knn_graph_k{args.k_neighbors}.json'}")
    print("パイプライン実行完了")


if __name__ == "__main__":
    main() 