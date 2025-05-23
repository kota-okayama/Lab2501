import torch
import torch.optim as optim
from torch.utils.data import DataLoader  # SubsetRandomSampler は一旦コメントアウト
import numpy as np
import os
import sys
import time
import argparse  # argparse をインポート
import json  # json をインポート

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.load_yaml_data import load_bibliographic_data
from data_processing.feature_extraction import load_fasttext_model

from .network import BaseNetwork, SiameseNetwork, ContrastiveLoss
from .dataset import BibliographicPairDataset

# --- ハイパーパラメータ ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # メモリに応じて調整
NUM_EPOCHS = 10  # まずは少ないエポック数でテスト
EMBEDDING_DIM = 128  # BaseNetworkの出力次元
DROPOUT_RATE = 0.3
CONTRASTIVE_MARGIN = 1.0
FASTTEXT_LANG = "ja"
# YAML_PATH = 'benchmark/bib_japan_20241024/1k/record.yml' # メイン関数内で定義
MODEL_SAVE_PATH = "siamese_model_pytorch/saved_models"


def load_pairs_from_json(file_path):
    """JSONファイルからペアのリストをロードする"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"Successfully loaded {len(pairs)} pairs from {file_path}")
        return pairs
    except FileNotFoundError:
        print(f"Error: Pair file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return []


def train(args):  # args を引数として受け取る
    print("Starting training process...")
    start_time = time.time()

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- データの準備 ---
    print("\nStep 1: Loading and preparing data...")
    all_records_list = load_bibliographic_data(args.record_yaml_path)
    if not all_records_list:
        print(f"Failed to load bibliographic records from {args.record_yaml_path}. Exiting.")
        return
    records_map = {str(record["record_id"]): record for record in all_records_list}  # record_idを文字列に統一

    positive_pairs = load_pairs_from_json(args.positive_pairs_path)
    negative_pairs = load_pairs_from_json(args.negative_pairs_path)

    if not positive_pairs and not negative_pairs:
        print("No training pairs loaded (both positive and negative lists are empty). Exiting.")
        return

    # ラベル付け (positive: 1, negative: 0)
    # generate_pairs.pyの出力形式が (id1, id2, label) のタプルであることを想定
    # もしgenerate_pairs.pyがラベルを含まないなら、ここで付与する必要がある
    # 今回は generate_pairs.py が (id1, id2, label) 形式で出力すると仮定。
    # もしそうでない場合は、ここで調整が必要。
    # 例: positive_pairs_with_label = [(p[0], p[1], 1) for p in positive_pairs]
    #     negative_pairs_with_label = [(n[0], n[1], 0) for n in negative_pairs]
    #     all_pairs = positive_pairs_with_label + negative_pairs_with_label

    # generate_pairs.py が既に (id1, id2, label) 形式で出力していると仮定し、そのまま結合
    all_pairs = positive_pairs + negative_pairs

    if not all_pairs:
        print("No training pairs available after combining positive and negative pairs. Exiting.")
        return
    print(f"Total pairs for dataset: {len(all_pairs)}")

    ft_model = load_fasttext_model(
        model_full_path=args.fasttext_model_path
    )  # model_path引数を指定 -> model_full_path に変更
    if not ft_model:
        print(f"Failed to load fastText model from {args.fasttext_model_path}. Exiting.")
        return

    full_dataset = BibliographicPairDataset(all_pairs, records_map, ft_model)
    if len(full_dataset) == 0:
        print("Dataset is empty after filtering (e.g., records not found in records_map). Cannot train. Exiting.")
        return
    print(f"Full dataset size (after filtering): {len(full_dataset)}")

    # 全データで訓練
    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # 例: CPUコア数に応じて調整 (0はメインプロセスでロード)
        pin_memory=True if device.type == "cuda" else False,  # CUDAが利用可能な場合
    )
    print(f"DataLoader created with batch size {args.batch_size}, num_workers=4, pin_memory={train_loader.pin_memory}.")

    # --- モデル、損失関数、オプティマイザの初期化 ---
    print("\nStep 2: Initializing model, loss function, and optimizer...")
    base_network = BaseNetwork(
        input_dim=ft_model.get_dimension(), embedding_dim=args.embedding_dim, dropout_rate=args.dropout_rate
    ).to(device)
    siamese_model = SiameseNetwork(base_network).to(device)
    criterion = ContrastiveLoss(margin=args.contrastive_margin).to(device)
    optimizer = optim.Adam(siamese_model.parameters(), lr=args.learning_rate)

    print("Model, loss, and optimizer initialized.")

    # --- 学習ループ ---
    print("\nStep 3: Starting training loop...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        siamese_model.train()
        running_loss = 0.0
        processed_batches = 0

        for i, batch_data in enumerate(train_loader):
            if batch_data is None:  # データセット側でエラーがあった場合Noneを返すことがある
                print(f"Warning: Received None for batch {i+1}. Skipping this batch.")
                continue

            input1, input2, label = batch_data
            input1, input2, label = input1.to(device), input2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = siamese_model(input1, input2)
            loss = criterion(output1, output2, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

            if (i + 1) % 200 == 0:
                avg_batch_loss = loss.item()
                print(
                    f"  Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], Current Batch Loss: {avg_batch_loss:.4f}"
                )

        epoch_loss = running_loss / processed_batches if processed_batches > 0 else 0
        epoch_end_time = time.time()
        print(
            f"Epoch [{epoch+1}/{args.epochs}] completed. Average Training Loss: {epoch_loss:.4f}. Time: {epoch_end_time - epoch_start_time:.2f}s"
        )

    # --- モデルの保存 ---
    print("\nStep 4: Saving the trained model...")
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_filename = f"base_network_emb{args.embedding_dim}_epoch{args.epochs}.pth"
    save_path = os.path.join(args.model_save_dir, model_filename)
    torch.save(base_network.state_dict(), save_path)
    print(f"Trained BaseNetwork model saved to {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time // 60:.0f}m {total_time % 60:.0f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Siamese Network for bibliographic record similarity.")
    parser.add_argument(
        "--positive_pairs_path",
        type=str,
        required=True,
        help="Path to the JSON file containing positive training pairs.",
    )
    parser.add_argument(
        "--negative_pairs_path",
        type=str,
        required=True,
        help="Path to the JSON file containing negative training pairs.",
    )
    parser.add_argument(
        "--record_yaml_path", type=str, required=True, help="Path to the YAML file containing bibliographic records."
    )
    parser.add_argument(
        "--fasttext_model_path", type=str, required=True, help="Path to the pre-trained fastText model (.bin)."
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="siamese_model_pytorch/saved_models",
        help="Directory to save the trained model.",
    )

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of the embedding layer in BaseNetwork."
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate in BaseNetwork.")
    parser.add_argument("--contrastive_margin", type=float, default=1.0, help="Margin for ContrastiveLoss.")

    cli_args = parser.parse_args()
    train(cli_args)
