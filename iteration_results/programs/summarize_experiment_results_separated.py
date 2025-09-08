#!/usr/bin/env python3
"""
実験結果をまとめるプログラム（K近傍ペア評価と全ペア推論評価を分離版）

results_{datatype}/run_500_{num}_{datatype}/evaluation_results/ から
評価結果レポートファイルを読み取り、K近傍ペア評価と全ペア推論評価を分けて
モデル名、適合率、再現率、F1値、混合行列を抽出する
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
    elif 'inconsistency' in model_name.lower() or 'inconsisten' in model_name.lower() or 'inconsistecy' in model_name.lower():
        return 'inconsistency'
    elif 'gpt-4o-mini' in model_name.lower():
        return 'baseline'
    else:
        return 'other'


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
            sample_size = int(datatype_match.group(1))
            datatype = datatype_match.group(2)
        else:
            sample_size = 0
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
                        strategy = extract_strategy_from_model_name(model_name)
                        
                        result = {
                            'datatype': datatype,
                            'sample_size': sample_size,
                            'model_name': model_name,
                            'model_type': 'before' if 'ファインチューニング前' in line else 'after',
                            'evaluation_type': current_evaluation_type,
                            'strategy': strategy,
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


def generate_strategy_analysis_csv(df: pd.DataFrame, output_file: str):
    """
    戦略別・サンプル数別の分析結果をCSVで出力（グラフ作成用）
    ベースラインをサンプル数0として各戦略の始発点に設定
    """
    analysis_results = []
    
    for eval_type in ['knn', 'all_pairs']:
        eval_df = df[df['evaluation_type'] == eval_type]
        
        for datatype in sorted(eval_df['datatype'].unique()):
            datatype_df = eval_df[eval_df['datatype'] == datatype]
            
            # ベースラインの平均値を取得
            baseline_df = datatype_df[datatype_df['strategy'] == 'baseline']
            if len(baseline_df) > 0:
                baseline_f1 = baseline_df['f1'].mean()
                baseline_precision = baseline_df['precision'].mean()
                baseline_recall = baseline_df['recall'].mean()
            else:
                baseline_f1 = 0.0
                baseline_precision = 0.0
                baseline_recall = 0.0
            
            # ベースライン以外の戦略を処理
            for strategy in sorted(datatype_df['strategy'].unique()):
                if strategy == 'baseline':
                    continue  # ベースラインはスキップ
                
                # サンプル数0（ベースライン）を追加
                analysis_results.append({
                    'evaluation_type': eval_type,
                    'datatype': datatype,
                    'strategy': strategy,
                    'sample_size': 0,
                    'avg_f1': baseline_f1,
                    'avg_precision': baseline_precision,
                    'avg_recall': baseline_recall,
                    'count': len(baseline_df)
                })
                
                # 実際のサンプル数のデータを追加
                strategy_df = datatype_df[datatype_df['strategy'] == strategy]
                for sample_size in sorted(strategy_df['sample_size'].unique()):
                    sample_df = strategy_df[strategy_df['sample_size'] == sample_size]
                    
                    if len(sample_df) > 0:
                        avg_f1 = sample_df['f1'].mean()
                        avg_precision = sample_df['precision'].mean()
                        avg_recall = sample_df['recall'].mean()
                        count = len(sample_df)
                        
                        analysis_results.append({
                            'evaluation_type': eval_type,
                            'datatype': datatype,
                            'strategy': strategy,
                            'sample_size': sample_size,
                            'avg_f1': avg_f1,
                            'avg_precision': avg_precision,
                            'avg_recall': avg_recall,
                            'count': count
                        })
    
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df.to_csv(output_file, index=False, encoding='utf-8')
    return analysis_df


def generate_markdown_report(df: pd.DataFrame, output_file: str):
    """
    Markdown形式のレポートを生成する
    K近傍ペア評価と全ペア推論評価を分けて表示
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 実験結果サマリー（評価タイプ別）\n\n")
        f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"総実験数: {len(df)}件\n\n")
        
        # 評価タイプ別に処理
        for eval_type in ['knn', 'all_pairs']:
            eval_type_name = "K近傍ペア評価" if eval_type == 'knn' else "全ペア推論評価"
            f.write(f"# {eval_type_name}\n\n")
            
            eval_df = df[df['evaluation_type'] == eval_type]
            
            # データタイプ別の結果
            for datatype in sorted(eval_df['datatype'].unique()):
                f.write(f"## {datatype.upper()} データ\n\n")
                datatype_df = eval_df[eval_df['datatype'] == datatype]
                
                for sample_size in sorted(datatype_df['sample_size'].unique()):
                    f.write(f"### サンプルサイズ: {sample_size}\n\n")
                    sample_df = datatype_df[datatype_df['sample_size'] == sample_size]
                    
                    # モデル名を短縮
                    sample_df_display = sample_df.copy()
                    sample_df_display['model_name_short'] = sample_df_display['model_name'].apply(
                        lambda x: x.split(':')[-2] if ':' in x else x
                    )
                    
                    # テーブルヘッダー
                    f.write("| モデルタイプ | 戦略 | モデル名 | 適合率 | 再現率 | F1値 | TP | FN | FP | TN |\n")
                    f.write("|-------------|------|----------|--------|--------|------|----|----|----|----|\\n")
                    
                    # データ行
                    for _, row in sample_df_display.iterrows():
                        model_type_jp = "ファインチューニング前" if row['model_type'] == 'before' else "ファインチューニング後"
                        f.write(f"| {model_type_jp} | {row['strategy']} | {row['model_name_short']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} |\n")
                    
                    f.write("\n")
                    f.write("---\n\n")
        
        # 戦略別・サンプル数別統計（グラフ作成用データ）
        f.write("# 戦略別・サンプル数別統計\n\n")
        
        for eval_type in ['knn', 'all_pairs']:
            eval_type_name = "K近傍ペア評価" if eval_type == 'knn' else "全ペア推論評価"
            f.write(f"## {eval_type_name}\n\n")
            
            eval_df = df[df['evaluation_type'] == eval_type]
            
            for datatype in sorted(eval_df['datatype'].unique()):
                f.write(f"### {datatype.upper()} データ\n\n")
                datatype_df = eval_df[eval_df['datatype'] == datatype]
                
                # 戦略別・サンプル数別の平均F1値テーブル
                f.write("| 戦略 | サンプル数 | 平均F1値 | 平均適合率 | 平均再現率 | 実験数 |\n")
                f.write("|------|------------|----------|------------|------------|--------|\n")
                
                for strategy in sorted(datatype_df['strategy'].unique()):
                    strategy_df = datatype_df[datatype_df['strategy'] == strategy]
                    for sample_size in sorted(strategy_df['sample_size'].unique()):
                        sample_strategy_df = strategy_df[strategy_df['sample_size'] == sample_size]
                        if len(sample_strategy_df) > 0:
                            avg_f1 = sample_strategy_df['f1'].mean()
                            avg_precision = sample_strategy_df['precision'].mean()
                            avg_recall = sample_strategy_df['recall'].mean()
                            count = len(sample_strategy_df)
                            f.write(f"| {strategy} | {sample_size} | {avg_f1:.4f} | {avg_precision:.4f} | {avg_recall:.4f} | {count} |\n")
                
                f.write("\n")
        
        # 全体統計
        f.write("# 全体統計\n\n")
        f.write(f"- 処理したデータタイプ数: {df['datatype'].nunique()}\n")
        f.write(f"- 処理したサンプルサイズ数: {df['sample_size'].nunique()}\n")
        f.write(f"- 総モデル評価数: {len(df)}\n")
        f.write(f"- K近傍ペア評価数: {len(df[df['evaluation_type'] == 'knn'])}\n")
        f.write(f"- 全ペア推論評価数: {len(df[df['evaluation_type'] == 'all_pairs'])}\n")
        f.write(f"- ファインチューニング前モデル数: {len(df[df['model_type'] == 'before'])}\n")
        f.write(f"- ファインチューニング後モデル数: {len(df[df['model_type'] == 'after'])}\n\n")


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
    
    # 評価タイプ別にグループ化して表示
    for eval_type in ['knn', 'all_pairs']:
        eval_type_name = "K近傍ペア評価" if eval_type == 'knn' else "全ペア推論評価"
        print(f"\n--- {eval_type_name} ---")
        
        eval_df = df[df['evaluation_type'] == eval_type]
        
        for datatype in sorted(eval_df['datatype'].unique()):
            print(f"\n{datatype.upper()} データ:")
            datatype_df = eval_df[eval_df['datatype'] == datatype]
            
            # 戦略別・サンプル数別の統計
            for strategy in sorted(datatype_df['strategy'].unique()):
                strategy_df = datatype_df[datatype_df['strategy'] == strategy]
                print(f"  {strategy}戦略:")
                for sample_size in sorted(strategy_df['sample_size'].unique()):
                    sample_df = strategy_df[strategy_df['sample_size'] == sample_size]
                    if len(sample_df) > 0:
                        avg_f1 = sample_df['f1'].mean()
                        print(f"    サンプル数{sample_size}: 平均F1={avg_f1:.4f} (実験数: {len(sample_df)})")
    
    # CSVファイルに保存
    output_file = os.path.join(base_dir, 'experiment_results_separated.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n結果をCSVファイルに保存しました: {output_file}")
    
    # 戦略別分析CSVを生成
    analysis_output = os.path.join(base_dir, 'strategy_analysis.csv')
    analysis_df = generate_strategy_analysis_csv(df, analysis_output)
    print(f"戦略別分析結果をCSVファイルに保存しました: {analysis_output}")
    
    # Excelファイルにも保存
    try:
        output_excel = os.path.join(base_dir, 'experiment_results_separated.xlsx')
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # 全データ
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # 評価タイプ別シート
            for eval_type in ['knn', 'all_pairs']:
                eval_df = df[df['evaluation_type'] == eval_type]
                sheet_name = f'{eval_type.upper()}_Results'
                eval_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 戦略別分析シート
            analysis_df.to_excel(writer, sheet_name='Strategy_Analysis', index=False)
        
        print(f"結果をExcelファイルに保存しました: {output_excel}")
    except ImportError:
        print("openpyxlがインストールされていないため、Excelファイルの保存をスキップしました。")
    
    # Markdownファイルにも保存
    output_md = os.path.join(base_dir, 'experiment_results_separated.md')
    generate_markdown_report(df, output_md)
    print(f"結果をMarkdownファイルに保存しました: {output_md}")


if __name__ == '__main__':
    main()
