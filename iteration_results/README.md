# 実験結果まとめ

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
python3 iteration_results/programs/summarize_experiment_results_separated.py
```

### グラフ作成の再実行
```bash
python3 iteration_results/programs/create_strategy_graphs.py
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
