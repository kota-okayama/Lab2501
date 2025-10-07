# レンタルサーバーでのLlama実験セットアップガイド

## 前提条件
- Linux サーバー（Ubuntu/CentOS/RHEL等）
- リモートデスクトップまたはSSH接続
- 最低8GB RAM推奨（Llama3.1:8b用）
- 10GB以上の空きディスク容量

## 1. ファイル転送

以下のファイルをサーバーに転送してください：

### 必須ファイル:
```
Lab2501/
├── test_llama_evaluation_clean.py          # Llama評価スクリプト
├── requirements.txt                        # Python依存関係
├── benchmark/product_wdc/wdc_valid/small_clusters/1k/test_subset_1.yml  # テストデータ
├── results_wdc/run_1k_wdc/llm_evaluation_pairs/candidate_pairs_from_test_subset_1_k10.csv  # 候補ペア
└── server_run_script.sh                    # サーバー実行用スクリプト（下記で作成）
```

### 転送方法（選択肢）:
1. **SCP使用** (ローカルから):
   ```bash
   scp -r Lab2501/ user@server_ip:/home/user/
   ```

2. **rsync使用** (ローカルから):
   ```bash
   rsync -avz Lab2501/ user@server_ip:/home/user/Lab2501/
   ```

3. **リモートデスクトップ経由**:
   - ファイルマネージャーでドラッグ&ドロップ
   - または共有フォルダ経由

## 2. サーバー側セットアップ

### Python環境準備
```bash
# Python3とpipの確認
python3 --version
pip3 --version

# 仮想環境作成（推奨）
python3 -m venv llama_env
source llama_env/bin/activate

# 依存関係インストール
pip3 install -r requirements.txt
```

### Ollamaインストール（Linux用）
```bash
# Ollamaインストール
curl -fsSL https://ollama.com/install.sh | sh

# サービス開始
sudo systemctl start ollama
sudo systemctl enable ollama

# または手動起動
ollama serve &

# モデルダウンロード
ollama pull llama3.1
```

## 3. 実行スクリプト作成

サーバー用の実行スクリプトを作成：

```bash
#!/bin/bash
# server_run_script.sh

# 設定
PROJECT_DIR="/home/user/Lab2501"  # 実際のパスに変更
PAIRS_CSV="$PROJECT_DIR/results_wdc/run_1k_wdc/llm_evaluation_pairs/candidate_pairs_from_test_subset_1_k10.csv"
GROUND_TRUTH_YAML="$PROJECT_DIR/benchmark/product_wdc/wdc_valid/small_clusters/1k/test_subset_1.yml"
MODEL="llama3.1"
OLLAMA_URL="http://localhost:11434"
LIMIT_PAIRS=50  # サーバー性能に応じて調整

echo "=== Llamaエンティティマッチング実験（サーバー実行） ==="
echo "プロジェクトディレクトリ: $PROJECT_DIR"
echo "モデル: $MODEL"
echo "制限ペア数: $LIMIT_PAIRS"
echo "=========================================="

cd "$PROJECT_DIR"

# 仮想環境アクティベート（使用している場合）
source llama_env/bin/activate

# Ollamaサーバー確認
echo "Ollamaサーバーの確認..."
curl -s http://localhost:11434/api/tags > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ Ollamaサーバーが起動しています"
else
    echo "✗ Ollamaサーバーに接続できません"
    echo "以下のコマンドでサーバーを起動してください："
    echo "  sudo systemctl start ollama"
    echo "  または: ollama serve &"
    exit 1
fi

# モデル確認
echo "モデル '$MODEL' の確認..."
ollama list | grep -q "$MODEL"
if [ $? -eq 0 ]; then
    echo "✓ モデル '$MODEL' が利用可能です"
else
    echo "✗ モデル '$MODEL' が見つかりません"
    echo "以下のコマンドでモデルをダウンロードしてください："
    echo "  ollama pull $MODEL"
    exit 1
fi

# 実験実行
echo "実験開始..."
python3 test_llama_evaluation_clean.py \
    --pairs_csv "$PAIRS_CSV" \
    --ground_truth_yaml "$GROUND_TRUTH_YAML" \
    --model "$MODEL" \
    --ollama_url "$OLLAMA_URL" \
    --max_concurrent 3 \
    --limit_pairs "$LIMIT_PAIRS"

echo "実験完了"
```

## 4. 実行手順

```bash
# プロジェクトディレクトリに移動
cd /home/user/Lab2501

# 実行権限付与
chmod +x server_run_script.sh

# 実行
./server_run_script.sh
```

## 5. バックグラウンド実行（長時間実行用）

```bash
# nohupでバックグラウンド実行
nohup ./server_run_script.sh > experiment_output.log 2>&1 &

# 実行状況確認
tail -f experiment_output.log

# プロセス確認
ps aux | grep python3
```

## 6. 結果確認

実行後、以下のファイルが生成されます：
```
results_wdc/run_1k_wdc/evaluation_results/
├── llama_eval_llama3.1_[timestamp]_details.csv
├── llama_eval_llama3.1_[timestamp]_report.txt
└── llama_evaluation_cache.pkl
```

## 7. トラブルシューティング

### メモリ不足の場合：
```bash
# より軽量なモデルを使用
ollama pull llama3.1:8b
# スクリプト内のMODEL="llama3.1:8b"に変更
```

### 同時実行数調整：
```bash
# --max_concurrent の値を減らす（1-2推奨）
--max_concurrent 1
```

### ディスク容量確認：
```bash
df -h
# 必要に応じて不要ファイル削除
```

## 8. 性能最適化

### GPU利用（利用可能な場合）：
```bash
# NVIDIA GPUドライバー確認
nvidia-smi

# CUDA対応Ollamaの場合、自動でGPU使用
```

### CPU最適化：
```bash
# CPUコア数確認
nproc

# 環境変数設定
export OMP_NUM_THREADS=$(nproc)
```

## 9. 結果のローカル転送

```bash
# 結果をローカルにコピー
scp -r user@server_ip:/home/user/Lab2501/results_wdc/run_1k_wdc/evaluation_results/ ./server_results/
```
