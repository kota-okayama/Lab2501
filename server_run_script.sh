#!/bin/bash
# サーバー実行用スクリプト

# 設定（実際の環境に合わせて変更してください）
PROJECT_DIR="/home/user/Lab2501"  # サーバー上のプロジェクトディレクトリ
PAIRS_CSV="$PROJECT_DIR/results_wdc/run_1k_wdc/llm_evaluation_pairs/candidate_pairs_from_test_subset_1_k10.csv"
GROUND_TRUTH_YAML="$PROJECT_DIR/benchmark/product_wdc/wdc_valid/small_clusters/1k/test_subset_1.yml"
MODEL="llama3.1"
OLLAMA_URL="http://localhost:11434"
LIMIT_PAIRS=100  # サーバー性能に応じて調整

echo "=== Llamaエンティティマッチング実験（サーバー実行） ==="
echo "プロジェクトディレクトリ: $PROJECT_DIR"
echo "モデル: $MODEL"
echo "制限ペア数: $LIMIT_PAIRS"
echo "=========================================="

cd "$PROJECT_DIR" || exit 1

# 仮想環境アクティベート（使用している場合）
if [ -d "llama_env" ]; then
    echo "仮想環境をアクティベートします..."
    source llama_env/bin/activate
fi

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

# 必要なファイルの存在確認
if [ ! -f "$PAIRS_CSV" ]; then
    echo "✗ 候補ペアファイルが見つかりません: $PAIRS_CSV"
    exit 1
fi

if [ ! -f "$GROUND_TRUTH_YAML" ]; then
    echo "✗ 正解データファイルが見つかりません: $GROUND_TRUTH_YAML"
    exit 1
fi

if [ ! -f "test_llama_evaluation_clean.py" ]; then
    echo "✗ 評価スクリプトが見つかりません: test_llama_evaluation_clean.py"
    exit 1
fi

echo "✓ 必要なファイルが揃っています"

# 実験実行
echo "実験開始..."
echo "開始時刻: $(date)"

python3 test_llama_evaluation_clean.py \
    --pairs_csv "$PAIRS_CSV" \
    --ground_truth_yaml "$GROUND_TRUTH_YAML" \
    --model "$MODEL" \
    --ollama_url "$OLLAMA_URL" \
    --max_concurrent 3 \
    --limit_pairs "$LIMIT_PAIRS"

EXIT_CODE=$?
echo "終了時刻: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 実験が正常に完了しました"
    echo ""
    echo "結果ファイル:"
    find results_wdc/run_1k_wdc/evaluation_results/ -name "llama_eval_*" -newer test_llama_evaluation_clean.py 2>/dev/null || echo "  結果ファイルの検索に失敗しました"
else
    echo "✗ 実験が異常終了しました (終了コード: $EXIT_CODE)"
fi
