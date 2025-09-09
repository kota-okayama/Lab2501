#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
累積FTデータ統合機能の簡単なテスト
"""

import os
import sys

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from run_full_evaluation_pipeline import (
    extract_iteration_number,
    find_previous_ft_data,
    merge_ft_data_files,
    get_num_lines_in_jsonl
)

def test_iteration_extraction():
    """イテレーション番号抽出のテスト"""
    print("=== イテレーション番号抽出テスト ===")
    
    test_cases = [
        ("results_test/run_test_ite0_test", 0),
        ("results_test/run_test_ite1_test", 1),
        ("results_test/run_test_ite2_test", 2),
        ("results_test/run_normal_test", 0),
    ]
    
    for output_dir, expected in test_cases:
        result = extract_iteration_number(output_dir)
        status = "✓" if result == expected else "❌"
        print(f"{status} {output_dir} -> {result} (期待値: {expected})")

def test_ft_data_search():
    """FTデータ検索のテスト"""
    print("\n=== FTデータ検索テスト ===")
    
    output_base_dir = "test_data/results_test/run_test_ite1_test"
    current_iteration = 1
    
    strategies = ["random", "diversity"]
    
    for strategy in strategies:
        print(f"\n--- {strategy} 戦略 ---")
        previous_files = find_previous_ft_data(
            output_base_dir, "test", strategy, current_iteration
        )
        print(f"発見されたファイル数: {len(previous_files)}")

def test_merge_functionality():
    """統合機能のテスト"""
    print("\n=== 統合機能テスト ===")
    
    # テスト用の追加FTデータを作成
    ite1_random_file = "test_data/results_test/run_test_ite1_test/evaluation_results/ft_data_random_ite1.jsonl"
    ite1_diversity_file = "test_data/results_test/run_test_ite1_test/evaluation_results/ft_data_diversity_ite1.jsonl"
    
    # ite1用のランダムデータ
    with open(ite1_random_file, 'w', encoding='utf-8') as f:
        f.write('{"messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Test 1"}, {"role": "assistant", "content": "Yes"}]}\n')
        f.write('{"messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Test 2"}, {"role": "assistant", "content": "No"}]}\n')
        f.write('{"messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Test 3"}, {"role": "assistant", "content": "Yes"}]}\n')
    
    # ite1用のdiversityデータ
    with open(ite1_diversity_file, 'w', encoding='utf-8') as f:
        f.write('{"messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Diversity 1"}, {"role": "assistant", "content": "No"}]}\n')
        f.write('{"messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Diversity 2"}, {"role": "assistant", "content": "No"}]}\n')
        f.write('{"messages": [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": "Diversity 3"}, {"role": "assistant", "content": "Yes"}]}\n')
    
    # 統合テスト
    for strategy in ["random", "diversity"]:
        print(f"\n--- {strategy} 戦略の統合テスト ---")
        
        # ite0とite1のファイルを取得
        ite0_file = f"test_data/results_test/run_test_ite0_test/evaluation_results/ft_data_{strategy}_ite0.jsonl"
        ite1_file = f"test_data/results_test/run_test_ite1_test/evaluation_results/ft_data_{strategy}_ite1.jsonl"
        
        if os.path.exists(ite0_file) and os.path.exists(ite1_file):
            ite0_lines = get_num_lines_in_jsonl(ite0_file)
            ite1_lines = get_num_lines_in_jsonl(ite1_file)
            
            print(f"ite0ファイル: {ite0_lines} 行")
            print(f"ite1ファイル: {ite1_lines} 行")
            
            # 統合ファイルを作成
            output_file = f"test_data/results_test/run_test_ite1_test/evaluation_results/ft_data_{strategy}_cumulative_ite1.jsonl"
            total_lines = merge_ft_data_files([ite0_file, ite1_file], output_file)
            
            expected_lines = ite0_lines + ite1_lines
            status = "✓" if total_lines == expected_lines else "❌"
            print(f"{status} 統合結果: {total_lines} 行 (期待値: {expected_lines} 行)")
            
            if os.path.exists(output_file):
                actual_lines = get_num_lines_in_jsonl(output_file)
                print(f"実際のファイル: {actual_lines} 行")
        else:
            print(f"必要なファイルが見つかりません:")
            print(f"  ite0: {os.path.exists(ite0_file)} - {ite0_file}")
            print(f"  ite1: {os.path.exists(ite1_file)} - {ite1_file}")

def main():
    """メインテスト処理"""
    print("累積FTデータ統合機能のシンプルテスト")
    print("="*50)
    
    test_iteration_extraction()
    test_ft_data_search()
    test_merge_functionality()
    
    print("\n=== テスト完了 ===")
    print("生成された累積FTデータファイル:")
    
    import glob
    cumulative_files = glob.glob("test_data/results_test/run_test_ite1_test/evaluation_results/*cumulative*.jsonl")
    for file_path in cumulative_files:
        filename = os.path.basename(file_path)
        lines = get_num_lines_in_jsonl(file_path)
        print(f"  {filename}: {lines} 行")

if __name__ == "__main__":
    main()
