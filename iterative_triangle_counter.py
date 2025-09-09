#!/usr/bin/env python3
"""
反復的矛盾三角形カウンタ

既存のdetect_inconsistent_triangles.pyを使用して、
T,T,Fパターンの矛盾三角形が出なくなるまで反復実行し、
その総数をカウントする。

戦略:
1. detect_inconsistent_triangles.pyを実行
2. 結果から真のT,T,Fパターンを特定
3. そのパターンが存在する限り反復
4. 各反復で見つかった矛盾三角形数をカウント
"""

import os
import subprocess
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import re

class IterativeTriangleCounter:
    def __init__(self):
        self.script_path = "siamese_model_pytorch/detect_inconsistent_triangles.py"
        self.temp_dir = None
        
    def setup_temp_dir(self):
        """一時ディレクトリを作成"""
        self.temp_dir = tempfile.mkdtemp(prefix="triangle_counter_")
        print(f"一時ディレクトリ: {self.temp_dir}")
        
    def cleanup_temp_dir(self):
        """一時ディレクトリを削除"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"一時ディレクトリを削除: {self.temp_dir}")
    
    def find_yaml_file(self, datatype):
        """データタイプに対応するYAMLファイルを探す"""
        yaml_patterns = [
            f"benchmark/*{datatype}*/2k/record.yml",
            f"benchmark/*{datatype}*/2k/training_set_1.yml", 
            f"benchmark/*{datatype}*/2k/training_set_2.yml",
            f"benchmark/extracted_{datatype}/sampled_data_2000.yml",
            f"benchmark/{datatype}_*/sampled_data_2000.yml"
        ]
        
        import glob
        for pattern in yaml_patterns:
            files = glob.glob(pattern)
            if files:
                print(f"Found YAML for {datatype}: {files[0]}")
                return files[0]
        
        print(f"Warning: No YAML file found for {datatype}")
        return None
    
    def run_triangle_detection(self, csv_file, yaml_file, iteration):
        """detect_inconsistent_triangles.pyを実行"""
        output_dir = os.path.join(self.temp_dir, f"iter_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "python3", self.script_path,
            "--input-csv", csv_file,
            "--ground-truth-yaml", yaml_file,
            "--score-column", "predicted_similar_before",
            "--output-dir", output_dir,
            "--num-triangles", "1000"  # 多めに取得
        ]
        
        print(f"実行中: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"エラー: {result.stderr}")
                return None
            return output_dir
        except subprocess.TimeoutExpired:
            print("タイムアウト: 5分以内に完了しませんでした")
            return None
        except Exception as e:
            print(f"実行エラー: {e}")
            return None
    
    def count_ttf_triangles(self, output_dir):
        """T,T,Fパターンの矛盾三角形をカウント"""
        triangle_file = None
        for file in os.listdir(output_dir):
            if file.endswith("_inconsistent_triangles.csv"):
                triangle_file = os.path.join(output_dir, file)
                break
        
        if not triangle_file or not os.path.exists(triangle_file):
            print(f"矛盾三角形ファイルが見つかりません: {output_dir}")
            return 0
        
        try:
            df = pd.read_csv(triangle_file)
            if df.empty:
                return 0
            
            # T,T,Fパターンをカウント
            ttf_count = 0
            for _, row in df.iterrows():
                edges = [row['true_edge12'], row['true_edge23'], row['true_edge31']]
                # Noneや欠損値を除外
                valid_edges = [e for e in edges if pd.notna(e) and e is not None]
                
                if len(valid_edges) == 3:
                    true_count = sum(valid_edges)
                    if true_count == 2:  # T,T,Fパターン
                        ttf_count += 1
            
            print(f"  T,T,Fパターン: {ttf_count}個")
            return ttf_count
            
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return 0
    
    def process_dataset(self, csv_file, datatype):
        """1つのデータセットを処理"""
        print(f"\n=== 処理開始: {datatype} ===")
        print(f"CSVファイル: {csv_file}")
        
        yaml_file = self.find_yaml_file(datatype)
        if not yaml_file:
            return {"datatype": datatype, "error": "YAML file not found"}
        
        total_ttf_triangles = 0
        iteration = 1
        max_iterations = 10  # 無限ループ防止
        
        current_csv = csv_file
        
        while iteration <= max_iterations:
            print(f"\n--- 反復 {iteration} ---")
            
            output_dir = self.run_triangle_detection(current_csv, yaml_file, iteration)
            if not output_dir:
                break
            
            ttf_count = self.count_ttf_triangles(output_dir)
            total_ttf_triangles += ttf_count
            
            print(f"反復 {iteration}: {ttf_count}個のT,T,F矛盾三角形")
            
            if ttf_count == 0:
                print("T,T,F矛盾三角形が見つからなくなりました。処理完了。")
                break
            
            iteration += 1
        
        result = {
            "datatype": datatype,
            "csv_file": os.path.basename(csv_file),
            "yaml_file": os.path.basename(yaml_file) if yaml_file else "N/A",
            "total_ttf_triangles": total_ttf_triangles,
            "iterations": iteration - 1,
            "completed": ttf_count == 0 if iteration <= max_iterations else False
        }
        
        print(f"\n{datatype} 処理完了:")
        print(f"  総T,T,F矛盾三角形数: {total_ttf_triangles}")
        print(f"  反復回数: {iteration - 1}")
        
        return result
    
    def run_all_datasets(self):
        """全データセットを処理"""
        # 対象ファイルを検索
        import glob
        csv_files = glob.glob("results_*/run_2k_*/evaluation_results/*details.csv")
        
        if not csv_files:
            print("対象CSVファイルが見つかりません")
            return []
        
        print(f"見つかったCSVファイル: {len(csv_files)}個")
        for f in csv_files:
            print(f"  {f}")
        
        self.setup_temp_dir()
        results = []
        
        try:
            for csv_file in csv_files:
                # データタイプを抽出
                match = re.search(r'results_([^/]+)/run_2k_', csv_file)
                datatype = match.group(1) if match else "unknown"
                
                result = self.process_dataset(csv_file, datatype)
                results.append(result)
                
        finally:
            self.cleanup_temp_dir()
        
        return results
    
    def save_results(self, results):
        """結果をCSVファイルに保存"""
        df = pd.DataFrame(results)
        output_file = "ttf_triangle_count_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n結果をCSVファイルに保存: {output_file}")
        
        # サマリー表示
        print("\n" + "="*60)
        print("T,T,F矛盾三角形カウント結果サマリー")
        print("="*60)
        
        total_triangles = 0
        for result in results:
            print(f"\nデータタイプ: {result['datatype']}")
            if 'error' in result:
                print(f"  エラー: {result['error']}")
            else:
                print(f"  T,T,F矛盾三角形数: {result['total_ttf_triangles']:,}")
                print(f"  反復回数: {result['iterations']}")
                print(f"  完了: {'はい' if result['completed'] else 'いいえ'}")
                total_triangles += result['total_ttf_triangles']
        
        print(f"\n全データセット合計: {total_triangles:,}個のT,T,F矛盾三角形")

def main():
    counter = IterativeTriangleCounter()
    results = counter.run_all_datasets()
    counter.save_results(results)

if __name__ == "__main__":
    main()

