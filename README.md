# LLM Entity Matching Evaluation Pipeline

## 概要

`run_full_evaluation_pipeline.py`は、大規模言語モデル（LLM）を用いたエンティティマッチングの性能評価から、ファインチューニング用データ生成までを自動化する包括的なパイプラインです。

このパイプラインは、書誌データ、音楽データ、人物データなどの重複検出タスクにおいて、ファインチューニング前後のLLMの性能を比較評価し、さらなる改善のためのトレーニングデータを生成します。

## パイプライン構成

### 5つの主要ステップ

1. **Embedding生成とK近傍グラフ構築**
   - OpenAI Embeddingを使用してレコードの埋め込み表現を生成
   - 複数フィールドの組み合わせでEmbeddingを作成
   - K近傍グラフを構築し、類似レコードのネットワークを作成

2. **評価ペア抽出**
   - K近傍グラフからLLM評価対象となるレコードペアを抽出
   - 重複を排除したユニークなペアのリストを生成

3. **モデル性能評価**
   - ファインチューニング前後の2つのLLMで同一ペアを評価
   - 非同期処理により効率的に大量のペアを処理
   - 性能レポート（F1スコア、精度、再現率等）を生成

4. **矛盾する三角形の検出**
   - 推移律に反するペア（A=B, B=C, A≠C）を検出
   - モデルの一貫性の問題を特定

5. **ファインチューニング用データ準備**
   - 検出した矛盾ペアと判断困難なペアを組み合わせ
   - 次回のファインチューニング用JSONLデータを生成

## 必要な環境

### 依存関係
- Python 3.8+
- OpenAI API アクセス
- 必要なPythonパッケージ（requirements.txtを参照）

### プロジェクト構造
```
プロジェクトルート/
├── run_full_evaluation_pipeline.py
├── openai_embedding_experiment/
│   └── run_multi_embedding_pipeline.py
├── siamese_model_pytorch/
│   ├── evaluate_finetuning_performance_async.py
│   ├── detect_inconsistent_triangles.py
│   └── prepare_finetuning_data.py
└── benchmark/
    └── bib_kyoto_20241024/
        └── 1k/
            └── record.yml
```

## 使用方法

### 基本的な実行コマンド

```bash
python3 run_full_evaluation_pipeline.py \
    --record_yaml_path "benchmark/bib_kyoto_20241024/1k/record.yml" \
    --output_base_dir "results_bibkyoto" \
    --data_type "bib" \
    --model_before_ft "gpt-4o-mini-2024-07-18" \
    --model_after_ft "ft:gpt-4o-mini-2024-07-18:mlab:bib-matching-inconsistency-0519:BYiGHy7V"
```

### 必須引数

| 引数 | 説明 | 例 |
|------|------|-----|
| `--record_yaml_path` | 入力レコードと正解クラスタのYAMLファイル | `benchmark/bib_kyoto_20241024/1k/record.yml` |
| `--output_base_dir` | 全出力のベースディレクトリ | `results_bibkyoto` |
| `--data_type` | データ種類 (`bib`, `music`, `person`) | `bib` |
| `--model_before_ft` | ファインチューニング前モデルID | `gpt-4o-mini-2024-07-18` |
| `--model_after_ft` | ファインチューニング後モデルID | `ft:gpt-4o-mini-2024-07-18:mlab:bib-matching-inconsistency-0519:BYiGHy7V` |

### オプション引数

#### Step 1: Embedding & Graphing
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--openai_embedding_model` | `text-embedding-ada-002` | OpenAIエンベディングモデル |
| `--api_batch_size` | `50` | APIバッチサイズ |
| `--k_neighbors` | `15` | K近傍のK値 |
| `--embedding_combinations` | `full` | エンベディング組み合わせ |

#### Step 3: Model Evaluation
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--max_concurrent` | `20` | 最大同時リクエスト数 |
| `--requests_per_minute` | `3000` | 1分間の最大リクエスト数 |

#### Step 4: Inconsistency Detection
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--inconsistency_threshold` | `0.8` | 矛盾検出スコア閾値 |
| `--inconsistency_top_n` | `100` | 検出する矛盾ペア上位N件 |

### ステップスキップオプション

特定のステップをスキップして実行することも可能です：

```bash
# Step 1をスキップ（既にEmbeddingが存在する場合）
--skip_embedding_and_graphing

# Step 2をスキップ（既にペアが抽出済みの場合）
--skip_pair_extraction

# Step 3をスキップ（既に評価済みの場合）
--skip_evaluation

# Step 4をスキップ（矛盾検出不要の場合）
--skip_inconsistency_detection

# Step 5をスキップ（ファインチューニングデータ不要の場合）
--skip_finetuning_data_preparation
```

## 出力ファイル

パイプライン実行後、以下のディレクトリ構造で結果が生成されます：

```
results_bibkyoto/
├── embeddings/                          # Step 1
│   ├── full_embedding.json
│   └── ...
├── graphs/                              # Step 1
│   └── merged_knn_graph_k15.json
├── llm_evaluation_pairs/                # Step 2
│   └── candidate_pairs_from_record_k15.csv
└── evaluation_results/                  # Step 3-5
    ├── eval_async_*_details.csv        # 詳細評価結果
    ├── finetuning_performance_report_*.txt  # 性能レポート
    ├── *_inconsistent_triangles.csv    # 矛盾ペア
    └── finetuning_data_from_*.jsonl     # 次回FT用データ
```

## 実行例

### 完全パイプライン実行
```bash
python3 run_full_evaluation_pipeline.py \
    --record_yaml_path "benchmark/bib_kyoto_20241024/1k/record.yml" \
    --output_base_dir "results_bibkyoto" \
    --data_type "bib" \
    --model_before_ft "gpt-4o-mini-2024-07-18" \
    --model_after_ft "ft:gpt-4o-mini-2024-07-18:mlab:bib-matching-inconsistency-0519:BYiGHy7V" \
    --k_neighbors 15 \
    --max_concurrent 10 \
    --inconsistency_top_n 100
```

### 評価のみ実行（Embedding等をスキップ）
```bash
python3 run_full_evaluation_pipeline.py \
    --record_yaml_path "benchmark/bib_kyoto_20241024/1k/record.yml" \
    --output_base_dir "results_bibkyoto" \
    --data_type "bib" \
    --model_before_ft "gpt-4o-mini-2024-07-18" \
    --model_after_ft "ft:gpt-4o-mini-2024-07-18:mlab:bib-matching-inconsistency-0519:BYiGHy7V" \
    --skip_embedding_and_graphing \
    --skip_pair_extraction
```

## 注意事項

1. **OpenAI API制限**: 大量のリクエストを送信するため、API制限に注意してください
2. **処理時間**: データサイズによっては数時間から数日かかる場合があります
3. **メモリ使用量**: 大きなデータセットでは十分なメモリが必要です
4. **ファイル権限**: 出力ディレクトリへの書き込み権限が必要です

## トラブルシューティング

### よくあるエラー

1. **ファイルが見つからない**: 入力ファイルパスを確認してください
2. **API認証エラー**: OpenAI APIキーが正しく設定されているか確認してください
3. **メモリ不足**: バッチサイズを小さくするか、より大きなメモリを持つ環境で実行してください

### ログ確認

各ステップで詳細なログが出力されます。エラーが発生した場合は、該当ステップのログを確認してください。

## 関連ファイル

- `requirements.txt`: 必要なPythonパッケージ
- `setup.sh`: 環境セットアップスクリプト
- 各サブスクリプトの詳細な使用方法は、それぞれのファイル内のdocstringを参照してください

## ライセンス

このプロジェクトのライセンスについては、プロジェクトルートのLICENSEファイルを参照してください。