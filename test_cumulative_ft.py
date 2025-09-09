#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
累積ファインチューニング機能のテスト用スクリプト
"""

import os
import sys
import subprocess

def run_test_iteration(iteration):
    """指定されたイテレーションでテストを実行"""
    print(f"\n{'='*60}")
    print(f"イテレーション {iteration} のテスト実行")
    print(f"{'='*60}")
    
    command = [
        "python3", "run_full_evaluation_pipeline.py",
        "--record_yaml_path", "test_data/benchmark/test_dataset/test_records.yml",
        "--output_base_dir", f"test_data/results_test/run_test_ite{iteration}_test",
        "--data_type", "bib",
        "--model_ids", 
        "ft:gpt-4o-mini-2024-07-18:mlab:test-matching-random-ite0-100:TEST001",
        "ft:gpt-4o-mini-2024-07-18:mlab:test-matching-diversity-ite0-100:TEST002",
        "--k_neighbors", "5",
        "--skip_step_1", "--skip_step_2", "--skip_step_3"
    ]
    
    print("実行コマンド:")
    print(" ".join(command))
    print()
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"終了コード: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"エラー: {e}")
        return False

def check_generated_files(iteration):
    """生成されたファイルを確認"""
    print(f"\n--- イテレーション {iteration} の生成ファイル確認 ---")
    
    results_dir = f"test_data/results_test/run_test_ite{iteration}_test/evaluation_results"
    
    if not os.path.exists(results_dir):
        print(f"結果ディレクトリが存在しません: {results_dir}")
        return
    
    print(f"結果ディレクトリ: {results_dir}")
    
    # FTデータファイルを確認
    import glob
    ft_files = glob.glob(os.path.join(results_dir, "ft_data_*.jsonl"))
    
    for ft_file in ft_files:
        filename = os.path.basename(ft_file)
        try:
            with open(ft_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  {filename}: {len(lines)} 行")
        except Exception as e:
            print(f"  {filename}: 読み込みエラー - {e}")

def main():
    """メインテスト処理"""
    print("累積ファインチューニング機能のテスト開始")
    
    # テストデータの存在確認
    required_files = [
        "test_data/benchmark/test_dataset/test_records.yml",
        "test_data/results_test/run_test_ite0_test/evaluation_results/ft_data_random_ite0.jsonl",
        "test_data/results_test/run_test_ite0_test/evaluation_results/ft_data_diversity_ite0.jsonl"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"必要なテストファイルが見つかりません: {file_path}")
            return False
    
    print("✓ 必要なテストファイルが揃っています")
    
    # ite0の既存データを確認
    check_generated_files(0)
    
    # ite1のテストを実行
    success = run_test_iteration(1)
    
    if success:
        print("\n✓ テスト実行完了")
        check_generated_files(1)
        
        # 累積データファイルの確認
        cumulative_files = glob.glob("test_data/results_test/run_test_ite1_test/evaluation_results/*cumulative*.jsonl")
        if cumulative_files:
            print("\n--- 累積FTデータファイル ---")
            for file_path in cumulative_files:
                filename = os.path.basename(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"  {filename}: {len(lines)} 行")
                        print(f"    期待値: ite0 (3行) + ite1 (3行) = 6行")
                except Exception as e:
                    print(f"  {filename}: 読み込みエラー - {e}")
        else:
            print("\n⚠️  累積FTデータファイルが見つかりません")
            
    else:
        print("\n❌ テスト実行に失敗しました")
    
    return success

if __name__ == "__main__":
    import glob
    success = main()
    sys.exit(0 if success else 1)
