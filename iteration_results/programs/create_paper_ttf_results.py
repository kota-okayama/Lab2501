#!/usr/bin/env python3
"""
論文用T,T,F矛盾三角形結果作成

bibデータセットを除いた結果を論文用に整理して出力する。
"""

import pandas as pd
import os

def create_paper_ttf_results():
    """
    論文用のT,T,F矛盾三角形結果を作成
    """
    # 元の結果を読み込み
    df = pd.read_csv('graph_based_ttf_results.csv')
    
    # bibデータセットを除外
    df_paper = df[df['datatype'] != 'bib'].copy()
    
    # データタイプの正式名称マッピング
    datatype_mapping = {
        'wdc': 'WDC-Product',
        'person': 'Persons-Leipzig',
        'bibkyoto': 'Bib-Kyoto',
        'walmart-amazon': 'Walmart-Amazon',
        'music': 'Music-Leipzig'
    }
    
    # 正式名称に変換
    df_paper['dataset_name'] = df_paper['datatype'].map(datatype_mapping)
    
    # 必要な列のみを選択し、列名を論文用に変更
    paper_columns = {
        'dataset_name': 'Dataset',
        'total_pairs': 'Total Pairs',
        'total_triangles': 'Total Triangles',
        'ttf_triangles': 'T,T,F Triangles',
        'ttf_rate': 'T,T,F Rate (vs Triangles)',
        'ttf_vs_pairs_rate': 'T,T,F Rate (vs Pairs)'
    }
    
    df_paper_final = df_paper[list(paper_columns.keys())].copy()
    df_paper_final = df_paper_final.rename(columns=paper_columns)
    
    # パーセンテージ列を追加
    df_paper_final['T,T,F Rate (%) vs Triangles'] = (df_paper_final['T,T,F Rate (vs Triangles)'] * 100).round(4)
    df_paper_final['T,T,F Rate (%) vs Pairs'] = (df_paper_final['T,T,F Rate (vs Pairs)'] * 100).round(4)
    
    # 数値をフォーマット
    df_paper_final['Total Pairs'] = df_paper_final['Total Pairs'].apply(lambda x: f"{x:,}")
    df_paper_final['Total Triangles'] = df_paper_final['Total Triangles'].apply(lambda x: f"{x:,}")
    df_paper_final['T,T,F Triangles'] = df_paper_final['T,T,F Triangles'].apply(lambda x: f"{x:,}")
    
    # 不要な小数点列を削除
    df_paper_final = df_paper_final.drop(columns=['T,T,F Rate (vs Triangles)', 'T,T,F Rate (vs Pairs)'])
    
    # 列の順序を調整
    final_columns = [
        'Dataset',
        'Total Pairs',
        'Total Triangles', 
        'T,T,F Triangles',
        'T,T,F Rate (%) vs Triangles',
        'T,T,F Rate (%) vs Pairs'
    ]
    df_paper_final = df_paper_final[final_columns]
    
    return df_paper_final

def save_results_to_iteration_folder(df_paper):
    """
    結果をiteration_resultsフォルダに保存
    """
    # iteration_resultsフォルダが存在することを確認
    if not os.path.exists('iteration_results'):
        os.makedirs('iteration_results')
    if not os.path.exists('iteration_results/data'):
        os.makedirs('iteration_results/data')
    
    # CSVファイルとして保存
    csv_path = 'iteration_results/data/ttf_triangles_paper_results.csv'
    df_paper.to_csv(csv_path, index=False)
    print(f"論文用T,T,F結果をCSVファイルに保存: {csv_path}")
    
    # Excelファイルとして保存
    excel_path = 'iteration_results/data/ttf_triangles_paper_results.xlsx'
    df_paper.to_excel(excel_path, index=False, sheet_name='T,T,F Triangles')
    print(f"論文用T,T,F結果をExcelファイルに保存: {excel_path}")
    
    # Markdownテーブルとして保存
    md_path = 'iteration_results/data/ttf_triangles_paper_results.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# T,T,F Inconsistent Triangles Analysis Results\n\n")
        f.write("## Summary\n\n")
        f.write("This table shows the count of T,T,F inconsistent triangles found in each dataset using KNN graph-based analysis.\n\n")
        f.write("**T,T,F Pattern**: Triangles where exactly 2 out of 3 edges are predicted as True and 1 edge is predicted as False, indicating logical inconsistency in the model's predictions.\n\n")
        f.write("## Results\n\n")
        # 手動でMarkdownテーブルを作成
        headers = df_paper.columns.tolist()
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for _, row in df_paper.iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in headers) + " |\n")
        f.write("\n\n## Notes\n\n")
        f.write("- **Total Pairs**: Number of record pairs evaluated by the LLM\n")
        f.write("- **Total Triangles**: Number of triangles found in the KNN graph\n")
        f.write("- **T,T,F Triangles**: Number of triangles with T,T,F inconsistent pattern\n")
        f.write("- **T,T,F Rate (%) vs Triangles**: Percentage of inconsistent triangles among all triangles\n")
        f.write("- **T,T,F Rate (%) vs Pairs**: Percentage of inconsistent triangles relative to total pairs\n")
        f.write("\n**Analysis Method**: KNN graph-based triangle detection for computational efficiency\n")
    
    print(f"論文用T,T,F結果をMarkdownファイルに保存: {md_path}")
    
    return csv_path, excel_path, md_path

def print_summary(df_paper):
    """
    結果のサマリーを表示
    """
    print("\n" + "="*80)
    print("論文用T,T,F矛盾三角形結果サマリー (bibデータセット除外)")
    print("="*80)
    
    # 数値列を元に戻して計算
    df_calc = df_paper.copy()
    df_calc['Total Pairs'] = df_calc['Total Pairs'].str.replace(',', '').astype(int)
    df_calc['Total Triangles'] = df_calc['Total Triangles'].str.replace(',', '').astype(int)
    df_calc['T,T,F Triangles'] = df_calc['T,T,F Triangles'].str.replace(',', '').astype(int)
    
    total_pairs = df_calc['Total Pairs'].sum()
    total_triangles = df_calc['Total Triangles'].sum()
    total_ttf = df_calc['T,T,F Triangles'].sum()
    
    print(f"\n対象データセット数: {len(df_paper)}")
    print(f"総ペア数: {total_pairs:,}")
    print(f"総三角形数: {total_triangles:,}")
    print(f"T,T,F矛盾三角形数: {total_ttf:,}")
    print(f"全体T,T,F率（対三角形）: {total_ttf/total_triangles:.6f} ({total_ttf/total_triangles*100:.4f}%)")
    print(f"全体T,T,F率（対ペア数）: {total_ttf/total_pairs:.6f} ({total_ttf/total_pairs*100:.4f}%)")
    
    print(f"\n各データセット別結果:")
    for _, row in df_paper.iterrows():
        print(f"  {row['Dataset']}: {row['T,T,F Triangles']} triangles ({row['T,T,F Rate (%) vs Triangles']:.4f}%)")

def main():
    print("論文用T,T,F矛盾三角形結果を作成中...")
    
    # 論文用結果を作成
    df_paper = create_paper_ttf_results()
    
    # iteration_resultsフォルダに保存
    csv_path, excel_path, md_path = save_results_to_iteration_folder(df_paper)
    
    # サマリーを表示
    print_summary(df_paper)
    
    print(f"\n✅ 論文用結果ファイルが作成されました:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Excel: {excel_path}")
    print(f"  - Markdown: {md_path}")

if __name__ == "__main__":
    main()
