#!/usr/bin/env python3
"""
実験結果の成果物を整理してiteration_resultsフォルダにまとめるプログラム
"""

import os
import shutil
from pathlib import Path
import glob


def create_directory_structure(base_dir: str):
    """
    ディレクトリ構造を作成
    """
    iteration_dir = os.path.join(base_dir, 'iteration_results')
    
    # メインディレクトリ
    Path(iteration_dir).mkdir(exist_ok=True)
    
    # サブディレクトリ
    subdirs = [
        'data',           # CSVファイルなど
        'reports',        # Markdownレポート
        'graphs',         # グラフファイル
        'excel',          # Excelファイル
        'programs'        # 作成したプログラム
    ]
    
    for subdir in subdirs:
        Path(os.path.join(iteration_dir, subdir)).mkdir(exist_ok=True)
    
    return iteration_dir


def copy_files(base_dir: str, iteration_dir: str):
    """
    ファイルを適切なサブディレクトリにコピー
    """
    # データファイル（CSV）
    data_files = [
        'strategy_analysis.csv',
        'experiment_results_separated.csv',
        'experiment_results_summary.csv'
    ]
    
    for file in data_files:
        src = os.path.join(base_dir, file)
        if os.path.exists(src):
            dst = os.path.join(iteration_dir, 'data', file)
            shutil.copy2(src, dst)
            print(f"コピー: {file} -> data/")
    
    # レポートファイル（Markdown）
    report_files = [
        'experiment_results_separated.md',
        'experiment_results_summary.md'
    ]
    
    for file in report_files:
        src = os.path.join(base_dir, file)
        if os.path.exists(src):
            dst = os.path.join(iteration_dir, 'reports', file)
            shutil.copy2(src, dst)
            print(f"コピー: {file} -> reports/")
    
    # Excelファイル
    excel_files = [
        'experiment_results_separated.xlsx',
        'experiment_results_summary.xlsx'
    ]
    
    for file in excel_files:
        src = os.path.join(base_dir, file)
        if os.path.exists(src):
            dst = os.path.join(iteration_dir, 'excel', file)
            shutil.copy2(src, dst)
            print(f"コピー: {file} -> excel/")
    
    # グラフファイル
    graphs_src = os.path.join(base_dir, 'graphs')
    if os.path.exists(graphs_src):
        graphs_dst = os.path.join(iteration_dir, 'graphs')
        # 既存のgraphsディレクトリを削除してから新しくコピー
        if os.path.exists(graphs_dst):
            shutil.rmtree(graphs_dst)
        shutil.copytree(graphs_src, graphs_dst)
        print(f"コピー: graphs/ -> graphs/ (全{len(os.listdir(graphs_src))}ファイル)")
    
    # プログラムファイル
    program_files = [
        'summarize_experiment_results_separated.py',
        'create_strategy_graphs.py',
        'organize_results.py'
    ]
    
    for file in program_files:
        src = os.path.join(base_dir, file)
        if os.path.exists(src):
            dst = os.path.join(iteration_dir, 'programs', file)
            shutil.copy2(src, dst)
            print(f"コピー: {file} -> programs/")


def create_readme(iteration_dir: str):
    """
    READMEファイルを作成
    """
    readme_content = """# 実験結果まとめ

このフォルダには実験結果の分析とグラフ作成の成果物が含まれています。

## ディレクトリ構成

### 📊 data/
- `strategy_analysis.csv` - 戦略別・サンプル数別統計（グラフ作成用）
- `experiment_results_separated.csv` - 全実験結果（評価タイプ分離版）
- `experiment_results_summary.csv` - 全実験結果（元版）

### 📝 reports/
- `experiment_results_separated.md` - 評価タイプ分離版レポート
- `experiment_results_summary.md` - 元版レポート

### 📈 graphs/
- 個別データタイプ別戦略比較グラフ
- 全データタイプ一覧比較グラフ
- 戦略別改善度ヒートマップ

### 📋 excel/
- `experiment_results_separated.xlsx` - 評価タイプ分離版Excel
- `experiment_results_summary.xlsx` - 元版Excel

### 💻 programs/
- `summarize_experiment_results_separated.py` - メイン分析プログラム
- `create_strategy_graphs.py` - グラフ作成プログラム
- `organize_results.py` - このファイル整理プログラム

## 使用方法

### データ分析の再実行
```bash
python3 programs/summarize_experiment_results_separated.py
```

### グラフ作成の再実行
```bash
python3 programs/create_strategy_graphs.py
```

## 主要な分析結果

### 戦略の種類
- **Random**: ランダムサンプリング
- **Diversity**: 多様性ベースサンプリング
- **Uncertainty**: 不確実性ベースサンプリング
- **Inconsistency**: 不整合ベースサンプリング

### 評価タイプ
- **K近傍ペア評価**: 近傍ペアでの評価
- **全ペア推論評価**: 全ペアでの推論評価

### データタイプ
- **BIBKYOTO**: 書誌データ（京都）
- **MUSIC**: 音楽データ
- **PERSON**: 人物データ
- **WALMART**: Walmart商品データ
- **WDC**: WDC商品データ

### グラフの見方
- **横軸**: ファインチューニング用サンプル数（0=ベースライン, 100, 200, 300, 400）
- **縦軸**: F1値
- **線**: 戦略別の性能変化
- **サンプル数0**: 各戦略の共通始発点（ベースライン性能）

## 注意事項
- ベースライン（サンプル数0）は全戦略で共通の始発点として設定
- K近傍ペア評価と全ペア推論評価は分離して分析
- グラフの日本語フォント警告は表示に影響しません
"""
    
    readme_path = os.path.join(iteration_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"作成: README.md")


def main():
    """
    メイン処理
    """
    base_dir = '/Users/kasiwamochi/Document/Lab2501'
    
    print("実験結果の成果物を整理中...")
    
    # ディレクトリ構造作成
    iteration_dir = create_directory_structure(base_dir)
    print(f"作成: {iteration_dir}")
    
    # ファイルコピー
    copy_files(base_dir, iteration_dir)
    
    # README作成
    create_readme(iteration_dir)
    
    print(f"\n✅ 整理完了!")
    print(f"📁 成果物は以下に整理されました: {iteration_dir}")
    print("\n📋 ディレクトリ構成:")
    print("├── data/           # CSVデータファイル")
    print("├── reports/        # Markdownレポート")
    print("├── graphs/         # グラフファイル")
    print("├── excel/          # Excelファイル")
    print("├── programs/       # 作成したプログラム")
    print("└── README.md       # 説明書")
    
    # ファイル数を表示
    total_files = 0
    for root, dirs, files in os.walk(iteration_dir):
        total_files += len(files)
    
    print(f"\n📊 総ファイル数: {total_files}個")


if __name__ == '__main__':
    main()
