#!/usr/bin/env python3
"""
実験結果をまとめるプログラム

results_{datatype}/run_500_{num}_{datatype}/evaluation_results/ から
評価結果レポートファイルを読み取り、モデル名、適合率、再現率、F1値、混合行列を抽出する
"""

import os
import re
import pandas as pd
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional


def extract_model_name(line: str) -> Optional[str]:
    """
    モデル名を抽出する
    例: "### ファインチューニング前モデル (ft:gpt-4o-mini-2024-07-18:mlab:wdc-product-matching-random-0904-100:CBx7yhmc)"
    """
    pattern = r'### ファインチューニング[前後]モデル \(([^)]+)\)'
    match = re.search(pattern, line)
    if match:
        return match.group(1)
    return None


def extract_metrics(lines: List[str], start_idx: int) -> Optional[Dict[str, float]]:
    """
    適合率、再現率、F1値を抽出する
    """
    for i in range(start_idx, min(start_idx + 10, len(lines))):
        line = lines[i]
        if '適合率:' in line and '再現率:' in line and 'F1:' in line:
            # 例: "- 適合率: 0.7871, 再現率: 0.7935, F1: 0.7903"
            pattern = r'適合率:\s*([\d.]+),\s*再現率:\s*([\d.]+),\s*F1:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match:
                return {
                    'precision': float(match.group(1)),
                    'recall': float(match.group(2)),
                    'f1': float(match.group(3))
                }
    return None


def extract_confusion_matrix(lines: List[str], start_idx: int) -> Optional[Dict[str, int]]:
    """
    混合行列を抽出する
    """
    confusion_matrix = {}
    
    for i in range(start_idx, min(start_idx + 10, len(lines))):
        line = lines[i].strip()
        
        # TP, FNを含む行を探す
        if 'Actual: Positive' in line and 'TP:' in line and 'FN:' in line:
            # 例: "  Actual: Positive  | TP: 196                | FN: 51                "
            tp_match = re.search(r'TP:\s*(\d+)', line)
            fn_match = re.search(r'FN:\s*(\d+)', line)
            if tp_match and fn_match:
                confusion_matrix['TP'] = int(tp_match.group(1))
                confusion_matrix['FN'] = int(fn_match.group(1))
        
        # FP, TNを含む行を探す
        elif 'Actual: Negative' in line and 'FP:' in line and 'TN:' in line:
            # 例: "  Actual: Negative  | FP: 53                 | TN: 2921              "
            fp_match = re.search(r'FP:\s*(\d+)', line)
            tn_match = re.search(r'TN:\s*(\d+)', line)
            if fp_match and tn_match:
                confusion_matrix['FP'] = int(fp_match.group(1))
                confusion_matrix['TN'] = int(tn_match.group(1))
    
    if len(confusion_matrix) == 4:
        return confusion_matrix
    return None


def parse_report_file(file_path: str) -> List[Dict]:
    """
    レポートファイルを解析して結果を抽出する
    K近傍ペア評価と全ペア推論評価を分けて抽出
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # ファイル名から実験情報を抽出
        filename = os.path.basename(file_path)
        
        # データタイプとサンプル数を抽出
        parent_dir = Path(file_path).parent.parent.name  # run_500_100_wdc など
        datatype_match = re.search(r'run_500_(\d+)_(\w+)', parent_dir)
        if datatype_match:
            sample_size = datatype_match.group(1)
            datatype = datatype_match.group(2)
        else:
            sample_size = "unknown"
            datatype = "unknown"
        
        i = 0
        current_evaluation_type = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 評価タイプを判定
            if line.startswith('## K近傍ペア評価'):
                current_evaluation_type = 'knn'
            elif line.startswith('## 全ペア推論評価'):
                current_evaluation_type = 'all_pairs'
            
            # モデル情報を含む行を探す
            if line.startswith('### ファインチューニング') and 'モデル' in line and current_evaluation_type:
                model_name = extract_model_name(line)
                if model_name:
                    # 混合行列を探す
                    confusion_matrix = None
                    metrics = None
                    
                    for j in range(i + 1, min(i + 15, len(lines))):
                        if '混合行列:' in lines[j]:
                            confusion_matrix = extract_confusion_matrix(lines, j + 1)
                            break
                    
                    # メトリクスを探す
                    for j in range(i + 1, min(i + 15, len(lines))):
                        if '適合率:' in lines[j]:
                            metrics = extract_metrics(lines, j)
                            break
                    
                    if confusion_matrix and metrics:
                        result = {
                            'datatype': datatype,
                            'sample_size': sample_size,
                            'model_name': model_name,
                            'model_type': 'before' if 'ファインチューニング前' in line else 'after',
                            'evaluation_type': current_evaluation_type,
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'TP': confusion_matrix['TP'],
                            'FN': confusion_matrix['FN'],
                            'FP': confusion_matrix['FP'],
                            'TN': confusion_matrix['TN'],
                            'file_path': file_path
                        }
                        results.append(result)
            
            i += 1
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return results


def find_report_files(base_dir: str) -> List[str]:
    """
    results_{datatype}/run_500_{num}_{datatype}/evaluation_results/*_report.txt ファイルを探す
    """
    report_files = []
    
    # results_* ディレクトリを探す
    results_dirs = glob.glob(os.path.join(base_dir, 'results_*'))
    
    for results_dir in results_dirs:
        if os.path.isdir(results_dir):
            # run_500_* ディレクトリを探す
            run_dirs = glob.glob(os.path.join(results_dir, 'run_500_*'))
            
            for run_dir in run_dirs:
                if os.path.isdir(run_dir):
                    # evaluation_results ディレクトリ内の *_report.txt ファイルを探す
                    eval_dir = os.path.join(run_dir, 'evaluation_results')
                    if os.path.isdir(eval_dir):
                        report_pattern = os.path.join(eval_dir, '*_report.txt')
                        report_files.extend(glob.glob(report_pattern))
    
    return report_files


def extract_strategy_from_model_name(model_name: str) -> str:
    """
    モデル名から戦略を抽出する
    """
    if 'random' in model_name.lower():
        return 'random'
    elif 'diversity' in model_name.lower():
        return 'diversity'
    elif 'uncertainty' in model_name.lower():
        return 'uncertainty'
    elif 'inconsistency' in model_name.lower() or 'inconsisten' in model_name.lower():
        return 'inconsistency'
    elif 'gpt-4o-mini' in model_name.lower():
        return 'baseline'
    else:
        return 'other'


def generate_markdown_report(df: pd.DataFrame, output_file: str):
    """
    Markdown形式のレポートを生成する
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 実験結果サマリー\n\n")
        f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"総実験数: {len(df)}件\n\n")
        
        # データタイプ別の結果
        for datatype in sorted(df['datatype'].unique()):
            f.write(f"## {datatype.upper()} データ\n\n")
            datatype_df = df[df['datatype'] == datatype]
            
            for sample_size in sorted(datatype_df['sample_size'].unique()):
                f.write(f"### サンプルサイズ: {sample_size}\n\n")
                sample_df = datatype_df[datatype_df['sample_size'] == sample_size]
                
                # モデル名を短縮
                sample_df_display = sample_df.copy()
                sample_df_display['model_name_short'] = sample_df_display['model_name'].apply(
                    lambda x: x.split(':')[-2] if ':' in x else x
                )
                
                # テーブルヘッダー
                f.write("| モデルタイプ | モデル名 | 適合率 | 再現率 | F1値 | TP | FN | FP | TN |\n")
                f.write("|-------------|----------|--------|--------|------|----|----|----|----|\n")
                
                # データ行
                for _, row in sample_df_display.iterrows():
                    model_type_jp = "ファインチューニング前" if row['model_type'] == 'before' else "ファインチューニング後"
                    f.write(f"| {model_type_jp} | {row['model_name_short']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} |\n")
                
                f.write("\n")
                
                # 改善度の計算と表示
                before_results = sample_df[sample_df['model_type'] == 'before']
                after_results = sample_df[sample_df['model_type'] == 'after']
                
                if len(before_results) > 0 and len(after_results) > 0:
                    f.write("#### 改善度\n\n")
                    
                    # ペアごとの改善度を計算
                    improvements = []
                    for _, after_row in after_results.iterrows():
                        # 対応するbeforeモデルを探す（同じファイルから来た結果）
                        corresponding_before = before_results[
                            before_results['file_path'] == after_row['file_path']
                        ]
                        
                        if len(corresponding_before) > 0:
                            before_row = corresponding_before.iloc[0]
                            f1_improvement = after_row['f1'] - before_row['f1']
                            precision_improvement = after_row['precision'] - before_row['precision']
                            recall_improvement = after_row['recall'] - before_row['recall']
                            
                            # モデル名を短縮
                            before_model_short = before_row['model_name'].split(':')[-2] if ':' in before_row['model_name'] else before_row['model_name']
                            after_model_short = after_row['model_name'].split(':')[-2] if ':' in after_row['model_name'] else after_row['model_name']
                            
                            improvements.append({
                                'before_model': before_model_short,
                                'after_model': after_model_short,
                                'f1_improvement': f1_improvement,
                                'precision_improvement': precision_improvement,
                                'recall_improvement': recall_improvement
                            })
                    
                    if improvements:
                        f.write("| Before → After | F1改善 | 適合率改善 | 再現率改善 |\n")
                        f.write("|----------------|--------|------------|------------|\n")
                        for imp in improvements:
                            f.write(f"| {imp['before_model']} → {imp['after_model']} | {imp['f1_improvement']:+.4f} | {imp['precision_improvement']:+.4f} | {imp['recall_improvement']:+.4f} |\n")
                        f.write("\n")
                
                f.write("---\n\n")
        
        # 全体統計
        f.write("## 全体統計\n\n")
        f.write(f"- 処理したデータタイプ数: {df['datatype'].nunique()}\n")
        f.write(f"- 処理したサンプルサイズ数: {df['sample_size'].nunique()}\n")
        f.write(f"- 総モデル評価数: {len(df)}\n")
        f.write(f"- ファインチューニング前モデル数: {len(df[df['model_type'] == 'before'])}\n")
        f.write(f"- ファインチューニング後モデル数: {len(df[df['model_type'] == 'after'])}\n\n")
        
        # データタイプ別統計
        f.write("### データタイプ別統計\n\n")
        f.write("| データタイプ | 実験数 | 平均F1値 | 最高F1値 | 最低F1値 |\n")
        f.write("|-------------|--------|----------|----------|----------|\n")
        
        for datatype in sorted(df['datatype'].unique()):
            datatype_df = df[df['datatype'] == datatype]
            avg_f1 = datatype_df['f1'].mean()
            max_f1 = datatype_df['f1'].max()
            min_f1 = datatype_df['f1'].min()
            f.write(f"| {datatype.upper()} | {len(datatype_df)} | {avg_f1:.4f} | {max_f1:.4f} | {min_f1:.4f} |\n")


def main():
    """
    メイン処理
    """
    base_dir = '/Users/kasiwamochi/Document/Lab2501'
    
    print("実験結果レポートファイルを検索中...")
    report_files = find_report_files(base_dir)
    
    if not report_files:
        print("レポートファイルが見つかりませんでした。")
        return
    
    print(f"見つかったレポートファイル数: {len(report_files)}")
    
    all_results = []
    
    for report_file in report_files:
        print(f"処理中: {report_file}")
        results = parse_report_file(report_file)
        all_results.extend(results)
    
    if not all_results:
        print("抽出できる結果がありませんでした。")
        return
    
    # DataFrameに変換
    df = pd.DataFrame(all_results)
    
    # 結果を表示
    print(f"\n抽出された結果数: {len(all_results)}")
    print("\n=== 実験結果サマリー ===")
    
    # データタイプ別にグループ化して表示
    for datatype in df['datatype'].unique():
        print(f"\n--- {datatype.upper()} データ ---")
        datatype_df = df[df['datatype'] == datatype]
        
        for sample_size in sorted(datatype_df['sample_size'].unique()):
            print(f"\nサンプルサイズ: {sample_size}")
            sample_df = datatype_df[datatype_df['sample_size'] == sample_size]
            
            # 結果を整理して表示
            display_columns = ['model_type', 'model_name', 'precision', 'recall', 'f1', 'TP', 'FN', 'FP', 'TN']
            sample_display = sample_df[display_columns].copy()
            
            # モデル名を短縮
            sample_display['model_name_short'] = sample_display['model_name'].apply(
                lambda x: x.split(':')[-2] if ':' in x else x
            )
            
            print(sample_display[['model_type', 'model_name_short', 'precision', 'recall', 'f1']].to_string(index=False))
    
    # CSVファイルに保存
    output_file = os.path.join(base_dir, 'experiment_results_summary.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n結果をCSVファイルに保存しました: {output_file}")
    
    # Excelファイルにも保存（より見やすい形式で）
    try:
        output_excel = os.path.join(base_dir, 'experiment_results_summary.xlsx')
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # 全データ
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # データタイプ別シート
            for datatype in df['datatype'].unique():
                datatype_df = df[df['datatype'] == datatype]
                sheet_name = f'{datatype.capitalize()}_Results'
                datatype_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"結果をExcelファイルに保存しました: {output_excel}")
    except ImportError:
        print("openpyxlがインストールされていないため、Excelファイルの保存をスキップしました.")
    
    # Markdownファイルにも保存
    output_md = os.path.join(base_dir, 'experiment_results_summary.md')
    generate_markdown_report(df, output_md)
    print(f"結果をMarkdownファイルに保存しました: {output_md}")


if __name__ == '__main__':
    main()
