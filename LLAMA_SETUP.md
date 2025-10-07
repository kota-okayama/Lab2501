# Llamaを使用したエンティティマッチング実験

OpenAIの代わりにLlamaを使用してエンティティマッチング実験を行うためのセットアップと実行手順です。

## 前提条件

- Python 3.8以上
- 十分なディスク容量（Llamaモデルは数GB必要）
- 十分なメモリ（8GB以上推奨）

## 1. Ollamaのインストール

### macOS/Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### または、公式サイトからダウンロード:
https://ollama.com/download

## 2. Ollamaサーバーの起動

```bash
ollama serve
```

別ターミナルで以下のコマンドを実行し、サーバーが起動していることを確認：
```bash
curl http://localhost:11434/api/tags
```

## 3. Llamaモデルのインストール

推奨モデル（サイズと性能のバランスが良い）：
```bash
ollama pull llama3.1
```

その他の利用可能なモデル：
```bash
# 軽量版（4GB RAM）
ollama pull llama3.1:8b

# 高性能版（16GB RAM）
ollama pull llama3.1:70b

# コード特化版
ollama pull codellama
```

インストール済みモデルの確認：
```bash
ollama list
```

## 4. 必要なPythonライブラリのインストール

```bash
pip3 install -r requirements.txt
```

追加で必要な場合：
```bash
pip3 install aiohttp>=3.8.0
```

## 5. テスト実行

### 簡単なテスト（20ペア）:
```bash
./run_llama_test.sh
```

### カスタマイズされたテスト:
```bash
python3 test_llama_evaluation.py \
    --pairs_csv "results_wdc/run_1k_wdc/llm_evaluation_pairs/candidate_pairs_from_test_subset_1_k10.csv" \
    --ground_truth_yaml "benchmark/product_wdc/wdc_valid/small_clusters/1k/test_subset_1.yml" \
    --model "llama3.1" \
    --ollama_url "http://localhost:11434" \
    --max_concurrent 3 \
    --limit_pairs 50
```

## 6. 設定オプション

### 引数の説明:
- `--model`: 使用するLlamaモデル名（`ollama list`で確認可能）
- `--ollama_url`: OllamaサーバーのURL（デフォルト: http://localhost:11434）
- `--max_concurrent`: 同時リクエスト数（Ollamaの場合は3-5推奨）
- `--limit_pairs`: テスト用にペア数を制限

### パフォーマンス調整:
- メモリが少ない場合: `--max_concurrent 1`
- 高性能マシンの場合: `--max_concurrent 10`

## 7. 結果の確認

結果は以下のディレクトリに保存されます：
```
results_wdc/run_1k_wdc/evaluation_results/
├── llama_eval_[model]_[timestamp]_details.csv  # 詳細結果
└── llama_eval_[model]_[timestamp]_report.txt   # サマリーレポート
```

## 8. トラブルシューティング

### Ollamaサーバーに接続できない:
```bash
# サーバーの状態確認
ps aux | grep ollama

# ポート確認
lsof -i :11434

# サーバー再起動
pkill ollama
ollama serve
```

### メモリ不足:
- より軽量なモデルを使用: `ollama pull llama3.1:8b`
- 同時リクエスト数を減らす: `--max_concurrent 1`

### 応答が遅い:
- GPUが利用可能か確認
- モデルサイズを調整
- `temperature`や`num_predict`パラメータを調整

## 9. 他のLLMバックエンドへの拡張

このスクリプトは以下のように拡張可能です：

### vLLMを使用する場合:
```python
# vLLM用の推論関数を追加
async def get_llm_evaluation_vllm(...)
```

### Transformersを使用する場合:
```python
# Transformers用の推論関数を追加
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## 10. OpenAIとの性能比較

既存の`evaluate_finetuning_performance_async.py`とこの`test_llama_evaluation.py`の結果を比較することで、LlamaとOpenAI GPTの性能差を評価できます。

主な比較指標：
- F1スコア
- 精度・再現率
- 処理時間
- コスト（OpenAIは従量課金、Llamaはローカル実行）
