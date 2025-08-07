# 複数フィールドエンベディングシステム 使用ガイド

## 概要

このシステムは、書誌データの複数フィールド（タイトル、著者、出版社、出版日）から異なる組み合わせでエンベディングを生成し、K近傍グラフを構築して統合することで、効果的なレコード重複検出を行うためのツールです。

## システム構成

### 主要スクリプト

1. **エンベディング生成**
   - `simple_multi_field_embeddings.py` - シンプル版
   - `custom_field_embeddings.py` - カスタムフィールド指定版
   - `vectorize_multi_field_openai.py` - 拡張版

2. **K近傍グラフ構築**
   - `build_multi_knn_graph_openai.py` - 複数グラフ構築・統合

3. **評価・分析**
   - `evaluate_knn_blocking_recall.py` - 再現率計算
   - `extract_llm_pairs.py` - ペア抽出
   - `evaluate_finetuning_performance.py` - LLM評価

4. **統合パイプライン**
   - `run_multi_embedding_pipeline.py` - 全体統合実行

## 使用手順

### 事前準備

1. **OpenAI APIキーの設定**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **必要なライブラリのインストール**
   ```bash
   pip install openai numpy scikit-learn faiss-cpu networkx pyyaml pandas
   ```

### Step 1: エンベディング生成

#### 基本的な使用方法（シンプル版）
```bash
python openai_embedding_experiment/simple_multi_field_embeddings.py \
  --yaml_path benchmark/bib_japan_20241024/extract_subset_10k.yml \
  --output_dir openai_embedding_experiment/embeddings_output \
  --combinations "full,title_only,author_only,title_author"
```

#### カスタムフィールド指定
```bash
python openai_embedding_experiment/custom_field_embeddings.py \
  --yaml_path benchmark/bib_japan_20241024/extract_subset_10k.yml \
  --output_dir openai_embedding_experiment/embeddings_output \
  --fields "title_only:bib1_title;author_only:bib1_author;publisher_only:bib1_publisher;full:bib1_title,bib1_author,bib1_publisher,bib1_pubdate"
```

#### 利用可能なフィールド
- `bib1_title`: タイトル
- `bib1_author`: 著者
- `bib1_publisher`: 出版社
- `bib1_pubdate`: 出版日

#### よく使用される組み合わせ
- `full`: 全フィールド結合
- `title_only`: タイトルのみ
- `author_only`: 著者のみ
- `title_author`: タイトル+著者

### Step 2: K近傍グラフ構築と統合

```bash
python openai_embedding_experiment/build_multi_knn_graph_openai.py \
  --embedding_summary_path openai_embedding_experiment/embeddings_output/embedding_summary.json \
  --k_neighbors 15 \
  --output_dir openai_embedding_experiment/graphs_output
```

#### K値の調整
- `--k_neighbors 10`: 
- `--k_neighbors 20`: 

#### 特定組み合わせのみ使用
```bash
python openai_embedding_experiment/build_multi_knn_graph_openai.py \
  --embedding_summary_path openai_embedding_experiment/embeddings_output/embedding_summary.json \
  --k_neighbors 15 \
  --output_dir openai_embedding_experiment/graphs_output \
  --selected_combinations "full,title_only"
```

### Step 3: 再現率評価

#### 統合グラフでの評価
```bash
python siamese_model_pytorch/evaluate_knn_blocking_recall.py \
  --knn_graph openai_embedding_experiment/graphs_output/merged_knn_graph_k15.json \
  --embeddings openai_embedding_experiment/embeddings_output/embeddings_full.npy \
  --record_ids openai_embedding_experiment/embeddings_output/record_ids_full.pkl \
  --ground_truth_yaml benchmark/bib_japan_20241024/extract_subset_10k.yml
```

#### 個別グラフでの比較評価
```bash
# 全体エンベディングのみ
python siamese_model_pytorch/evaluate_knn_blocking_recall.py \
  --knn_graph openai_embedding_experiment/graphs_output/knn_graph_full_k15.json \
  --embeddings openai_embedding_experiment/embeddings_output/embeddings_full.npy \
  --record_ids openai_embedding_experiment/embeddings_output/record_ids_full.pkl \
  --ground_truth_yaml benchmark/bib_japan_20241024/extract_subset_10k.yml
```

### Step 4: LLMによる評価

#### ペア抽出
```bash
python siamese_model_pytorch/extract_llm_pairs.py \
  --input_directory openai_embedding_experiment/graphs_output \
  --knn_graph_filename merged_knn_graph_k15.json \
  --output_directory openai_embedding_experiment/llm_pairs_new \
  --output_pairs_filename extracted_pairs_from_merged_graph.csv
```

#### LLM評価実行
```bash
python siamese_model_pytorch/evaluate_finetuning_performance.py \
  --record_yaml_dir benchmark/bib_japan_20241024 \
  --record_yaml_filename extract_subset_10k.yml \
  --eval_pairs_csv_dir openai_embedding_experiment/llm_pairs_new \
  --eval_pairs_csv_filename extracted_pairs_from_merged_graph.csv \
  --output_dir openai_embedding_experiment/llm_evaluation_results \
  --model_id_after_finetuning gpt-4o-mini-2024-07-18
```

### Step 5: 統合パイプライン実行

全工程を一括実行する場合：

```bash
python openai_embedding_experiment/run_multi_embedding_pipeline.py \
  --record_yaml_path benchmark/bib_japan_20241024/extract_subset_10k.yml \
  --output_base_dir openai_embedding_experiment/pipeline_output \
  --selected_combinations "full,title_only,author_only" \
  --k_neighbors 15
```

## 出力ファイル

### エンベディング生成後
- `embeddings_[組み合わせ名].npy`: エンベディング配列
- `record_ids_[組み合わせ名].pkl`: レコードID一覧
- `embedding_summary.json`: 生成結果サマリー

### グラフ構築後
- `knn_graph_[組み合わせ名]_k[K値].json`: 個別K近傍グラフ
- `merged_knn_graph_k[K値].json`: 統合K近傍グラフ
- `merge_summary_k[K値].json`: 統合結果サマリー

### 評価結果
- 再現率レポート（テキスト出力）
- LLM評価結果CSV
- 詳細分析レポート

## パフォーマンス調整

### APIコール最適化
- `--api_batch_size 50`: バッチサイズ調整
- 適切な待機時間設定

### K値の選択指針
- **K=5-10**: 高精度、低再現率
- **K=15-20**: バランス型
- **K=25+**: 高再現率、低精度

### フィールド組み合わせ戦略
1. **基本セット**: `full,title_only,author_only`
2. **拡張セット**: 上記 + `title_author,publisher_only`
3. **カスタム**: 用途に応じた自由な組み合わせ

## トラブルシューティング

### よくあるエラー

1. **OpenAI APIキー未設定**
   ```
   ValueError: OPENAI_API_KEY environment variable not set
   ```
   → APIキーを環境変数に設定

2. **ファイルが見つからない**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   → パスの確認、前の処理の完了確認

3. **メモリ不足**
   ```
   MemoryError: Unable to allocate array
   ```
   → バッチサイズの削減、データサイズの調整

### デバッグ方法
- 各ステップの出力ファイルを確認
- ログメッセージの詳細確認
- 小さなサンプルデータでのテスト実行

## 応用例

### 実験的評価
1. 異なるK値での比較
2. フィールド組み合わせの効果検証
3. 統合手法の比較

### 実運用での活用
1. 大規模データセットへの適用
2. ドメイン特化チューニング
3. リアルタイム処理への組み込み

## 注意事項

- OpenAI APIの利用料金に注意
- 大きなデータセットでは処理時間が長くなる可能性
- メモリ使用量の監視推奨
- 中間結果の定期的なバックアップ推奨 