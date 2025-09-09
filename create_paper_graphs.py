#!/usr/bin/env python3
"""
論文用の戦略別・サンプル数別のF1値グラフを作成するプログラム

strategy_analysis.csvを読み込んで、論文に適した高品質なグラフを作成する
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np


def setup_matplotlib_for_paper():
    """
    論文用のMatplotlib設定
    """
    # 論文用フォント設定
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    plt.rcParams['font.size'] = 14  # ベースフォントサイズを大きく
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    # 線とマーカーの設定
    plt.rcParams['lines.linewidth'] = 2.5  # 太い線
    plt.rcParams['lines.markersize'] = 8   # 大きなマーカー
    plt.rcParams['axes.linewidth'] = 1.2   # 軸線を太く
    
    # グリッドの設定
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3
    
    # 図のサイズ（論文用・横長）
    plt.rcParams['figure.figsize'] = (12, 5)
    
    # 高解像度設定
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # スタイル設定
    sns.set_style("whitegrid")
    
    # 論文用カラーパレット（モノクロ印刷でも区別可能）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    sns.set_palette(colors)


def get_strategy_display_name(strategy):
    """
    戦略名を論文用の表示名に変換
    """
    strategy_names = {
        'random': 'Random',
        'diversity': 'Diversity',
        'uncertainty': 'Uncertainty', 
        'inconsistency': 'Inconsistency',
        'baseline': 'Baseline'
    }
    return strategy_names.get(strategy, strategy.capitalize())


def get_datatype_display_name(datatype):
    """
    データタイプを論文用の表示名に変換
    """
    datatype_names = {
        'bibkyoto': 'Bibliography',
        'music': 'Music',
        'person': 'Person',
        'walmart': 'Walmart-Amazon',
        'wdc': 'WDC-Product'
    }
    return datatype_names.get(datatype, datatype.upper())


def get_evaluation_display_name(eval_type):
    """
    評価タイプを論文用の表示名に変換
    """
    eval_names = {
        'knn': 'K-NN Pair Evaluation',
        'all_pairs': 'All-Pair Inference'
    }
    return eval_names.get(eval_type, eval_type)


def create_strategy_comparison_graph(df: pd.DataFrame, eval_type: str, datatype: str, output_dir: str):
    """
    論文用の戦略別比較グラフを作成
    """
    # データをフィルタリング
    filtered_df = df[(df['evaluation_type'] == eval_type) & (df['datatype'] == datatype)]
    
    if len(filtered_df) == 0:
        print(f"No data found for {eval_type} - {datatype}")
        return
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 戦略別に線を描画（論文用順序：baseline → 提案手法 → その他）
    all_strategies = filtered_df['strategy'].unique()
    strategy_order = []
    
    # 1. Baseline を最初に
    if 'baseline' in all_strategies:
        strategy_order.append('baseline')
    
    # 2. 提案手法を次に（inconsistency が提案手法と仮定）
    if 'inconsistency' in all_strategies:
        strategy_order.append('inconsistency')
    
    # 3. その他の手法をアルファベット順で
    other_strategies = sorted([s for s in all_strategies if s not in ['baseline', 'inconsistency']])
    strategy_order.extend(other_strategies)
    
    strategies = strategy_order
    
    # 線のスタイル設定（モノクロ印刷対応）
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, strategy in enumerate(strategies):
        strategy_df = filtered_df[filtered_df['strategy'] == strategy].sort_values('sample_size')
        
        if len(strategy_df) > 0:
            ax.plot(strategy_df['sample_size'], strategy_df['avg_f1'], 
                   marker=markers[i % len(markers)], 
                   linestyle=line_styles[i % len(line_styles)],
                   linewidth=2.5, markersize=8, 
                   label=get_strategy_display_name(strategy),
                   markerfacecolor='white', markeredgewidth=2)
    
    # グラフの設定
    eval_name = get_evaluation_display_name(eval_type)
    datatype_name = get_datatype_display_name(datatype)
    
    ax.set_title(f'{datatype_name} Dataset - {eval_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Fine-tuning Sample Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    
    # 凡例の設定
    legend = ax.legend(title='Strategy', title_fontsize=12, fontsize=12, 
                      loc='best', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # グリッドの設定
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Y軸の範囲を調整
    y_min = filtered_df['avg_f1'].min() - 0.05
    y_max = filtered_df['avg_f1'].max() + 0.05
    ax.set_ylim(max(0, y_min), min(1, y_max))
    
    # X軸の設定
    x_ticks = sorted(filtered_df['sample_size'].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks])
    
    # 軸の設定
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2)
    
    plt.tight_layout()
    
    # ファイル保存
    filename = f'{datatype}_{eval_type}_strategy_comparison_paper.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Paper graph saved: {filepath}")


def create_all_datatypes_comparison(df: pd.DataFrame, eval_type: str, output_dir: str):
    """
    全データタイプを一つのグラフで比較（論文用）
    """
    datatypes = sorted(df['datatype'].unique())
    
    # サブプロット数を計算
    n_datatypes = len(datatypes)
    cols = 3
    rows = (n_datatypes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    eval_name = get_evaluation_display_name(eval_type)
    fig.suptitle(f'{eval_name} - All Datasets Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # 線のスタイル設定
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, datatype in enumerate(datatypes):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        filtered_df = df[(df['evaluation_type'] == eval_type) & (df['datatype'] == datatype)]
        
        if len(filtered_df) == 0:
            ax.set_title(f'{get_datatype_display_name(datatype)} - No Data', fontsize=12)
            ax.set_visible(False)
            continue
        
        # 戦略別に線を描画（論文用順序：baseline → 提案手法 → その他）
        all_strategies = filtered_df['strategy'].unique()
        strategy_order = []
        
        # 1. Baseline を最初に
        if 'baseline' in all_strategies:
            strategy_order.append('baseline')
        
        # 2. 提案手法を次に（inconsistency が提案手法と仮定）
        if 'inconsistency' in all_strategies:
            strategy_order.append('inconsistency')
        
        # 3. その他の手法をアルファベット順で
        other_strategies = sorted([s for s in all_strategies if s not in ['baseline', 'inconsistency']])
        strategy_order.extend(other_strategies)
        
        strategies = strategy_order
        
        for i, strategy in enumerate(strategies):
            strategy_df = filtered_df[filtered_df['strategy'] == strategy].sort_values('sample_size')
            
            if len(strategy_df) > 0:
                ax.plot(strategy_df['sample_size'], strategy_df['avg_f1'], 
                       marker=markers[i % len(markers)], 
                       linestyle=line_styles[i % len(line_styles)],
                       linewidth=2, markersize=6, 
                       label=get_strategy_display_name(strategy),
                       markerfacecolor='white', markeredgewidth=1.5)
        
        ax.set_title(f'{get_datatype_display_name(datatype)}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Size', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Y軸の範囲を調整
        y_min = filtered_df['avg_f1'].min() - 0.05
        y_max = filtered_df['avg_f1'].max() + 0.05
        ax.set_ylim(max(0, y_min), min(1, y_max))
        
        # X軸の設定
        x_ticks = sorted(filtered_df['sample_size'].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks])
    
    # 余ったサブプロットを非表示
    for idx in range(len(datatypes), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # ファイル保存
    filename = f'all_datatypes_{eval_type}_comparison_paper.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Paper comparison graph saved: {filepath}")


def create_strategy_improvement_heatmap(df: pd.DataFrame, output_dir: str):
    """
    論文用の戦略別改善度ヒートマップを作成
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
                    'datatype': get_datatype_display_name(datatype),
                    'strategy': get_strategy_display_name(strategy),
                    'improvement': improvement,
                    'max_f1': max_f1
                })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            pivot_df = improvement_df.pivot(index='datatype', columns='strategy', values='improvement')
            
            plt.figure(figsize=(12, 5))
            
            # ヒートマップの作成
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, 
                       fmt='.3f', cbar_kws={'label': 'F1 Score Improvement'},
                       linewidths=0.5, linecolor='white',
                       annot_kws={'size': 11, 'weight': 'bold'})
            
            eval_name = get_evaluation_display_name(eval_type)
            plt.title(f'{eval_name} - Strategy Improvement Heatmap', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Strategy', fontsize=14, fontweight='bold')
            plt.ylabel('Dataset', fontsize=14, fontweight='bold')
            
            # 軸ラベルの回転
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            filename = f'strategy_improvement_heatmap_{eval_type}_paper.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Paper heatmap saved: {filepath}")


def main():
    """
    メイン処理
    """
    base_dir = '/Users/kasiwamochi/Document/Lab2501'
    input_file = os.path.join(base_dir, 'strategy_analysis.csv')
    output_dir = os.path.join(base_dir, 'iteration_results', 'graphs_paper')
    
    # 出力ディレクトリを作成
    Path(output_dir).mkdir(exist_ok=True)
    
    # データを読み込み
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        print("Please run summarize_experiment_results_separated.py first.")
        return
    
    df = pd.read_csv(input_file)
    print(f"Data loaded: {len(df)} rows")
    
    # 論文用Matplotlib設定
    setup_matplotlib_for_paper()
    
    print("Creating paper-quality graphs...")
    
    # 評価タイプとデータタイプの組み合わせごとにグラフ作成
    for eval_type in ['knn', 'all_pairs']:
        eval_df = df[df['evaluation_type'] == eval_type]
        
        for datatype in sorted(eval_df['datatype'].unique()):
            create_strategy_comparison_graph(df, eval_type, datatype, output_dir)
        
        # 全データタイプ比較グラフ
        create_all_datatypes_comparison(df, eval_type, output_dir)
    
    # 改善度ヒートマップ
    create_strategy_improvement_heatmap(df, output_dir)
    
    print(f"\n✅ All paper-quality graphs saved to: {output_dir}")
    print("\n📊 Graph features for papers:")
    print("- High resolution (300 DPI)")
    print("- Large fonts and markers")
    print("- Monochrome-friendly colors and line styles")
    print("- Professional layout")
    print("- English labels and titles")
    
    # ファイル数を表示
    graph_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"\n📈 Total graphs created: {len(graph_files)}")


if __name__ == '__main__':
    main()
