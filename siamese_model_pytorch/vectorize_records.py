import torch
import numpy as np
import os
import sys
import pickle
import time
import argparse

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.load_yaml_data import load_bibliographic_data
from data_processing.feature_extraction import load_fasttext_model, get_text_representation
from .network import BaseNetwork

# --- ハードコードされた設定値は argparse で引数として受け取るため、ここでは削除またはコメントアウト ---
# EMBEDDING_DIM = 128
# INPUT_DIM_FASTTEXT = 300
# TRAINED_MODEL_FILENAME = f"base_network_emb{EMBEDDING_DIM}_epoch10.pth"
# MODEL_LOAD_PATH = "siamese_model_pytorch/saved_models"
# OUTPUT_PATH = "siamese_model_pytorch/vectorized_data"
# FASTTEXT_LANG = "ja" # load_fasttext_model に直接パスを渡すので不要


def vectorize_all_records(args):  # args を引数として受け取る
    print("Starting process to vectorize all records...")
    start_time = time.time()

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 学習済みBaseNetworkモデルのロード ---
    print("\nStep 1: Loading trained BaseNetwork model...")
    if not os.path.exists(args.trained_model_path):
        print(f"Error: Trained model not found at {args.trained_model_path}")
        print("Please ensure the model was trained and saved correctly.")
        return

    # BaseNetworkの初期化 (入力次元はfastTextの次元、埋め込み次元も引数から)
    base_network = BaseNetwork(input_dim=args.fasttext_dim, embedding_dim=args.embedding_dim)
    try:
        #  FutureWarning に対応するため weights_only=True を推奨 (ただしモデル保存形式による)
        #  もし問題があれば weights_only=False のままにするか、エラー内容に応じて対応
        base_network.load_state_dict(torch.load(args.trained_model_path, map_location=device, weights_only=True))
    except RuntimeError as e:
        if "weights_only" in str(e):
            print("Warning: Failed to load model with weights_only=True. Trying with weights_only=False.")
            print(f"  Details: {e}")
            try:
                base_network.load_state_dict(
                    torch.load(args.trained_model_path, map_location=device, weights_only=False)
                )
            except Exception as e_fallback:
                print(
                    f"Error loading model state_dict from {args.trained_model_path} even with weights_only=False: {e_fallback}"
                )
                print("Ensure --fasttext_dim and --embedding_dim match the saved model's architecture.")
                return
        else:
            print(f"Error loading model state_dict from {args.trained_model_path}: {e}")
            print("Ensure --fasttext_dim and --embedding_dim match the saved model's architecture.")
            return
    except Exception as e:
        print(f"Unexpected error loading model state_dict from {args.trained_model_path}: {e}")
        return

    base_network.to(device)
    base_network.eval()
    print(f"Trained BaseNetwork model loaded from {args.trained_model_path}")

    # --- fastTextモデルのロード ---
    print("\nStep 2: Loading fastText model...")
    ft_model = load_fasttext_model(model_full_path=args.fasttext_model_path)  # 正しい引数名を使用
    if not ft_model:
        print(f"Failed to load fastText model from {args.fasttext_model_path}. Exiting.")
        return

    if ft_model.get_dimension() != args.fasttext_dim:
        print(
            f"Error: fastText dimension ({ft_model.get_dimension()}) "
            f"does not match expected BaseNetwork input_dim ({args.fasttext_dim})."
        )
        print("Please check --fasttext_dim argument or the fastText model.")
        return

    # --- 書誌データのロード ---
    print("\nStep 3: Loading all bibliographic records...")
    all_records_list = load_bibliographic_data(args.record_yaml_path)
    if not all_records_list:
        print(f"Failed to load bibliographic records from {args.record_yaml_path}. Exiting.")
        return
    print(f"Loaded {len(all_records_list)} records.")

    # --- 全レコードのベクトル化 ---
    print("\nStep 4: Vectorizing all records...")
    record_embeddings = []
    record_ids_ordered = []
    processed_count = 0

    for record_info in all_records_list:
        record_id = record_info.get("record_id")
        if record_id is None:
            print(f"Warning: Record found without a 'record_id'. Skipping. Record data: {record_info.get('data', {})}")
            continue
        record_id = str(record_id)

        record_data = record_info.get("data", {})
        text_repr = get_text_representation(record_data)
        if not text_repr:
            print(f"Warning: No text representation for record_id {record_id}. Skipping.")
            continue

        fasttext_vector_np = ft_model.get_sentence_vector(text_repr)
        fasttext_vector_torch = torch.from_numpy(fasttext_vector_np.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = base_network(fasttext_vector_torch)

        record_embeddings.append(embedding.squeeze(0).cpu().numpy())
        record_ids_ordered.append(record_id)
        processed_count += 1
        if processed_count % 100 == 0 or processed_count == len(all_records_list):
            print(f"  Processed {processed_count}/{len(all_records_list)} records...")

    print(f"Finished vectorizing {processed_count} records.")

    if not record_embeddings:
        print("No records were vectorized. Exiting.")
        return

    embeddings_array = np.array(record_embeddings, dtype=np.float32)

    # --- 保存処理 ---
    print("\nStep 5: Saving vectorized data...")
    output_embeddings_dir = os.path.dirname(args.output_embeddings_path)
    output_ids_dir = os.path.dirname(args.output_ids_path)
    if output_embeddings_dir:  # 空文字列でないことを確認
        os.makedirs(output_embeddings_dir, exist_ok=True)
    if output_ids_dir:  # 空文字列でないことを確認
        os.makedirs(output_ids_dir, exist_ok=True)

    try:
        np.save(args.output_embeddings_path, embeddings_array)
        print(f"Record embeddings saved to {args.output_embeddings_path} (shape: {embeddings_array.shape})")

        with open(args.output_ids_path, "wb") as f:
            pickle.dump(record_ids_ordered, f)
        print(f"Corresponding record IDs saved to {args.output_ids_path} (count: {len(record_ids_ordered)})")
    except IOError as e:
        print(f"Error saving output files: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return

    total_time = time.time() - start_time
    print(f"\nVectorization process finished in {total_time // 60:.0f}m {total_time % 60:.0f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize all records using a trained BaseNetwork model.")
    parser.add_argument(
        "--trained_model_path", type=str, required=True, help="Path to the trained BaseNetwork model (.pth file)."
    )
    parser.add_argument(
        "--fasttext_model_path", type=str, required=True, help="Path to the pre-trained fastText model (.bin file)."
    )
    parser.add_argument(
        "--record_yaml_path",
        type=str,
        required=True,
        help="Path to the YAML file containing bibliographic records to vectorize.",
    )
    parser.add_argument(
        "--output_embeddings_path",
        type=str,
        required=True,
        help="Path to save the output record embeddings (.npy file).",
    )
    parser.add_argument(
        "--output_ids_path", type=str, required=True, help="Path to save the output record IDs (.pkl file)."
    )

    # BaseNetworkの構造に関わるため、学習時と一致させる必要がある
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of the embedding layer in BaseNetwork (must match the trained model).",
    )
    parser.add_argument(
        "--fasttext_dim",
        type=int,
        default=300,
        help="Dimension of the fastText model output (input to BaseNetwork, must match the trained model).",
    )

    cli_args = parser.parse_args()
    vectorize_all_records(cli_args)  # パースした引数を関数に渡す
