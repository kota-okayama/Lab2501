#!/usr/bin/env python3
"""
戦略別・サンプル数別のF1値グラフを作成するプログラム

strategy_analysis.csvを読み込んで、戦略ごとの折れ線グラフを作成する
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def setup_matplotlib():
    """
    Matplotlibの設定
    """
    # 日本語フォントの設定
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # スタイル設定
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def create_strategy_comparison_graph(df: pd.DataFrame, eval_type: str, datatype: str, output_dir: str):
    """
    特定の評価タイプ・データタイプについて戦略別比較グラフを作成
    """
    # データをフィルタリング
    filtered_df = df[(df['evaluation_type'] == eval_type) & (df['datatype'] == datatype)]
    
    if len(filtered_df) == 0:
        print(f"No data found for {eval_type} - {datatype}")
        return
    
    # グラフ作成
    plt.figure(figsize=(12, 8))
    
    # 戦略別に線を描画
    strategies = sorted(filtered_df['strategy'].unique())
    colors = sns.color_palette("husl", len(strategies))
    
    for i, strategy in enumerate(strategies):
        strategy_df = filtered_df[filtered_df['strategy'] == strategy].sort_values('sample_size')
        
        if len(strategy_df) > 0:
            plt.plot(strategy_df['sample_size'], strategy_df['avg_f1'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=strategy.capitalize(), color=colors[i])
            
            # データポイントに値を表示
            for _, row in strategy_df.iterrows():
                plt.annotate(f'{row["avg_f1"]:.3f}', 
                           (row['sample_size'], row['avg_f1']),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # グラフの設定
    eval_type_name = "K近傍ペア評価" if eval_type == 'knn' else "全ペア推論評価"
    plt.title(f'{datatype.upper()} データ - {eval_type_name}\n戦略別F1値の変化', fontsize=14, fontweight='bold')
    plt.xlabel('ファインチューニング用サンプル数', fontsize=12)
    plt.ylabel('F1値', fontsize=12)
    plt.legend(title='戦略', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Y軸の範囲を調整
    y_min = filtered_df['avg_f1'].min() - 0.05
    y_max = filtered_df['avg_f1'].max() + 0.05
    plt.ylim(max(0, y_min), min(1, y_max))
    
    # X軸の設定
    plt.xticks(sorted(filtered_df['sample_size'].unique()))
    
    plt.tight_layout()
    
    # ファイル保存
    filename = f'{datatype}_{eval_type}_strategy_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"グラフを保存しました: {filepath}")


def create_all_datatypes_comparison(df: pd.DataFrame, eval_type: str, output_dir: str):
    """
    全データタイプを一つのグラフで比較
    """
    datatypes = sorted(df['datatype'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    eval_type_name = "K近傍ペア評価" if eval_type == 'knn' else "全ペア推論評価"
    fig.suptitle(f'{eval_type_name} - 全データタイプ比較', fontsize=16, fontweight='bold')
    
    for idx, datatype in enumerate(datatypes):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        filtered_df = df[(df['evaluation_type'] == eval_type) & (df['datatype'] == datatype)]
        
        if len(filtered_df) == 0:
            ax.set_title(f'{datatype.upper()} - データなし')
            continue
        
        # 戦略別に線を描画
        strategies = sorted(filtered_df['strategy'].unique())
        colors = sns.color_palette("husl", len(strategies))
        
        for i, strategy in enumerate(strategies):
            strategy_df = filtered_df[filtered_df['strategy'] == strategy].sort_values('sample_size')
            
            if len(strategy_df) > 0:
                ax.plot(strategy_df['sample_size'], strategy_df['avg_f1'], 
                       marker='o', linewidth=2, markersize=6, 
                       label=strategy.capitalize(), color=colors[i])
        
        ax.set_title(f'{datatype.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('サンプル数', fontsize=10)
        ax.set_ylabel('F1値', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Y軸の範囲を調整
        y_min = filtered_df['avg_f1'].min() - 0.05
        y_max = filtered_df['avg_f1'].max() + 0.05
        ax.set_ylim(max(0, y_min), min(1, y_max))
        
        # X軸の設定
        ax.set_xticks(sorted(filtered_df['sample_size'].unique()))
    
    # 余ったサブプロットを非表示
    for idx in range(len(datatypes), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # ファイル保存
    filename = f'all_datatypes_{eval_type}_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"全データタイプ比較グラフを保存しました: {filepath}")


def create_strategy_improvement_heatmap(df: pd.DataFrame, output_dir: str):
    """
    戦略別・データタイプ別の改善度ヒートマップを作成
    """
    for eval_type in ['knn', 'all_pairs']:
        eval_df = df[df['evaluation_type'] == eval_type]
        
        # 各戦略の最大F1値を取得
        improvement_data = []
        
        for datatype in sorted(eval_df['datatype'].unique()):
            datatype_df = eval_df[eval_df['datatype'] == datatype]
            baseline_f1 = datatype_df[datatype_df['sample_size'] == 0]['avg_f1'].iloc[0] if len(datatype_df[datatype_df['sample_size'] == 0]) > 0 else 0
            
            for strategy in sorted(datatype_df['strategy'].unique()):
                strategy_df = datatype_df[datatype_df['strategy'] == strategy]
                max_f1 = strategy_df['avg_f1'].max()
                improvement = max_f1 - baseline_f1
                
                improvement_data.append({
                    'datatype': datatype.upper(),
                    'strategy': strategy.capitalize(),
                    'improvement': improvement,
                    'max_f1': max_f1
                })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            pivot_df = improvement_df.pivot(index='datatype', columns='strategy', values='improvement')
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, 
                       fmt='.3f', cbar_kws={'label': 'F1値改善度'})
            
            eval_type_name = "K近傍ペア評価" if eval_type == 'knn' else "全ペア推論評価"
            plt.title(f'{eval_type_name} - 戦略別改善度ヒートマップ', fontsize=14, fontweight='bold')
            plt.xlabel('戦略', fontsize=12)
            plt.ylabel('データタイプ', fontsize=12)
            plt.tight_layout()
            
            filename = f'strategy_improvement_heatmap_{eval_type}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"改善度ヒートマップを保存しました: {filepath}")


def main():
    """
    メイン処理
    """
    base_dir = '/Users/kasiwamochi/Document/Lab2501'
    input_file = os.path.join(base_dir, 'strategy_analysis.csv')
    output_dir = os.path.join(base_dir, 'graphs')
    
    # 出力ディレクトリを作成
    Path(output_dir).mkdir(exist_ok=True)
    
    # データを読み込み
    if not os.path.exists(input_file):
        print(f"エラー: {input_file} が見つかりません。")
        print("まず summarize_experiment_results_separated.py を実行してください。")
        return
    
    df = pd.read_csv(input_file)
    print(f"データを読み込みました: {len(df)} 行")
    
    # Matplotlib設定
    setup_matplotlib()
    
    # 評価タイプとデータタイプの組み合わせごとにグラフ作成
    for eval_type in ['knn', 'all_pairs']:
        eval_df = df[df['evaluation_type'] == eval_type]
        
        for datatype in sorted(eval_df['datatype'].unique()):
            create_strategy_comparison_graph(df, eval_type, datatype, output_dir)
        
        # 全データタイプ比較グラフ
        create_all_datatypes_comparison(df, eval_type, output_dir)
    
    # 改善度ヒートマップ
    create_strategy_improvement_heatmap(df, output_dir)
    
    print(f"\n全てのグラフが {output_dir} に保存されました。")
    print("\n作成されたグラフ:")
    print("1. 個別データタイプ別戦略比較グラフ")
    print("2. 全データタイプ一覧比較グラフ")
    print("3. 戦略別改善度ヒートマップ")


if __name__ == '__main__':
    main()
