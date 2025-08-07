#!/bin/bash

# 引数チェック (4つ必要になる)
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <train_benchmark_yaml_path> <eval_benchmark_yaml_path> <fasttext_model_path> <output_base_dir>"
    exit 1
fi

TRAIN_BENCHMARK_YAML="$1"
EVAL_BENCHMARK_YAML="$2"
FASTTEXT_MODEL_PATH="$3"
OUTPUT_BASE_DIR="$4"

# --- 固定パラメータ (変更なし) ---
EPOCHS=10
K_NEIGHBORS=10
EMBEDDING_DIM=128
FASTTEXT_DIM=300
LEARNING_RATE=0.001
BATCH_SIZE=128
DROPOUT_RATE=0.3
CONTRASTIVE_MARGIN=1.0

# --- 出力パスの定義 ---
# モデル名は訓練データに基づいて一意にする
# TRAIN_BENCHMARK_YAML のファイル名（拡張子なし）を取得する例
TRAIN_DATA_NAME=$(basename "$TRAIN_BENCHMARK_YAML" .yml) # 例: record (5k/record なら record)

# 出力ディレクトリ (訓練データと評価データで一部パスを分けることも検討)
# 今回はモデルと学習ペアのみ訓練データ名を付加
TRAIN_PAIR_DATA_DIR="${OUTPUT_BASE_DIR}/generated_pairs/${TRAIN_DATA_NAME}"
MODEL_DIR="${OUTPUT_BASE_DIR}/trained_models/${TRAIN_DATA_NAME}" # 訓練データごとのモデル

# 評価データのベクトルとグラフは、どの訓練モデルを使ったかと、どの評価データかを示すようにできる
# ここではシンプルに、OUTPUT_BASE_DIR直下にeval用のものを作る
EVAL_VECTOR_DIR="${OUTPUT_BASE_DIR}/vectorized_data_eval"
EVAL_KNN_GRAPH_DIR="${OUTPUT_BASE_DIR}/knn_graph_eval"

# LLM評価用ペアリストの出力ディレクトリとファイルパスを定義
EVAL_LLM_PAIRS_DIR="${OUTPUT_BASE_DIR}/llm_evaluation_pairs_eval"
EVAL_LLM_PAIRS_CSV_PATH="${EVAL_LLM_PAIRS_DIR}/llm_pairs_from_scenarioC_eval_k${K_NEIGHBORS}.csv"

# ファイルパス
POSITIVE_PAIRS_PATH="${TRAIN_PAIR_DATA_DIR}/positive_pairs.json"
NEGATIVE_PAIRS_PATH="${TRAIN_PAIR_DATA_DIR}/negative_pairs.json"
# 学習済みモデルのフルパス (このパスで存在チェックを行う)
TRAINED_MODEL_FULL_PATH="${MODEL_DIR}/base_network_emb${EMBEDDING_DIM}_epoch${EPOCHS}.pth"

EVAL_RECORD_EMBEDDINGS_PATH="${EVAL_VECTOR_DIR}/record_embeddings_eval.npy"
EVAL_RECORD_IDS_PATH="${EVAL_VECTOR_DIR}/record_ids_eval.pkl"
EVAL_KNN_GRAPH_PATH="${EVAL_KNN_GRAPH_DIR}/knn_graph_eval_k${K_NEIGHBORS}.json"
# シナリオCのグラフ出力パスを定義
EVAL_SCENARIO_C_KNN_GRAPH_PATH="${EVAL_KNN_GRAPH_DIR}/knn_graph_scenarioC_eval_k${K_NEIGHBORS}.json"

# 出力ディレクトリ作成
mkdir -p "$TRAIN_PAIR_DATA_DIR" "$MODEL_DIR" "$EVAL_VECTOR_DIR" "$EVAL_KNN_GRAPH_DIR" "$EVAL_LLM_PAIRS_DIR"
echo "Output directories created under ${OUTPUT_BASE_DIR}"


# --- モデルの存在確認と学習スキップ ---
if [ -f "$TRAINED_MODEL_FULL_PATH" ]; then
    echo -e "\n--- Trained model found at $TRAINED_MODEL_FULL_PATH ---"
    echo "Skipping Step 1 (Pair Generation) and Step 2 (Model Training)."
else
    echo -e "\n--- Trained model NOT found at $TRAINED_MODEL_FULL_PATH ---"
    echo "Proceeding with Step 1 and Step 2."

    # 1. 学習データペアの生成 (訓練データ使用)
    echo -e "\n--- Step 1: Generating training pairs for $TRAIN_BENCHMARK_YAML ---"
    python3 -m data_processing.generate_pairs \
        --input_yaml "$TRAIN_BENCHMARK_YAML" \
        --output_dir "$TRAIN_PAIR_DATA_DIR"
    if [ $? -ne 0 ]; then echo "Error in generate_pairs.py. Exiting."; exit 1; fi

    # 2. Siamese Network モデルの学習 (訓練データ使用)
    echo -e "\n--- Step 2: Training Siamese Network model using $TRAIN_BENCHMARK_YAML ---"
    python3 -m siamese_model_pytorch.train \
        --positive_pairs_path "$POSITIVE_PAIRS_PATH" \
        --negative_pairs_path "$NEGATIVE_PAIRS_PATH" \
        --record_yaml_path "$TRAIN_BENCHMARK_YAML" \
        --fasttext_model_path "$FASTTEXT_MODEL_PATH" \
        --model_save_dir "$MODEL_DIR" \
        --epochs $EPOCHS \
        --embedding_dim $EMBEDDING_DIM \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --dropout_rate $DROPOUT_RATE \
        --contrastive_margin $CONTRASTIVE_MARGIN
    if [ $? -ne 0 ]; then echo "Error in train.py. Exiting."; exit 1; fi
fi # モデル存在確認の終了

# 3. 全レコードのベクトル化 (評価データ使用、学習/ロードしたモデルで)
echo -e "\n--- Step 3: Vectorizing records from $EVAL_BENCHMARK_YAML ---"
python3 -m siamese_model_pytorch.vectorize_records \
    --trained_model_path "$TRAINED_MODEL_FULL_PATH" \
    --fasttext_model_path "$FASTTEXT_MODEL_PATH" \
    --record_yaml_path "$EVAL_BENCHMARK_YAML" \
    --output_embeddings_path "$EVAL_RECORD_EMBEDDINGS_PATH" \
    --output_ids_path "$EVAL_RECORD_IDS_PATH" \
    --embedding_dim $EMBEDDING_DIM \
    --fasttext_dim $FASTTEXT_DIM
if [ $? -ne 0 ]; then echo "Error in vectorize_records.py. Exiting."; exit 1; fi

# 4. K近傍グラフの構築 (評価データから作成したベクトル使用)
echo -e "\n--- Step 4: Building K-NN graph for $EVAL_BENCHMARK_YAML ---"
python3 -m siamese_model_pytorch.build_knn_graph \
    --embeddings_path "$EVAL_RECORD_EMBEDDINGS_PATH" \
    --ids_path "$EVAL_RECORD_IDS_PATH" \
    --output_graph_path "$EVAL_KNN_GRAPH_PATH" \
    --k_neighbors $K_NEIGHBORS
if [ $? -ne 0 ]; then echo "Error in build_knn_graph.py. Exiting."; exit 1; fi

# 5. K近傍グラフのブロッキング性能評価 (評価データ使用)
echo -e "\n--- Step 5: Evaluating K-NN blocking recall for $EVAL_BENCHMARK_YAML ---"
python3 -m siamese_model_pytorch.evaluate_knn_blocking_recall \
    --knn_graph "$EVAL_KNN_GRAPH_PATH" \
    --embeddings "$EVAL_RECORD_EMBEDDINGS_PATH" \
    --record_ids "$EVAL_RECORD_IDS_PATH" \
    --ground_truth_yaml "$EVAL_BENCHMARK_YAML" \
    --output_graph_c_path "$EVAL_SCENARIO_C_KNN_GRAPH_PATH"
if [ $? -ne 0 ]; then echo "Error in evaluate_knn_blocking_recall.py. Exiting."; exit 1; fi

# 6. LLM評価用ペアの抽出 (シナリオCのグラフから)
echo -e "\n--- Step 6: Extracting LLM evaluation pairs from Scenario C graph for $EVAL_BENCHMARK_YAML ---"
python3 -m siamese_model_pytorch.extract_llm_pairs \
    --input_directory "$EVAL_KNN_GRAPH_DIR" \
    --knn_graph_filename "$(basename "$EVAL_SCENARIO_C_KNN_GRAPH_PATH")" \
    --output_directory "$EVAL_LLM_PAIRS_DIR" \
    --output_pairs_filename "$(basename "$EVAL_LLM_PAIRS_CSV_PATH")"
if [ $? -ne 0 ]; then echo "Error in extract_llm_pairs.py. Exiting."; exit 1; fi

echo -e "\n--- All pipeline steps completed successfully! --- " 