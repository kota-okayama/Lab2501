import pandas as pd
import json
import os
import argparse
import random


def sample_pairs_from_evaluation_results(input_csv_path, output_dir, num_positive_samples, num_negative_samples):
    """
    評価結果CSVからPositiveペアとNegativeペアをランダムサンプリングし、JSONファイルに出力する。

    Args:
        input_csv_path (str): evaluate_finetuning_performance.py が出力したCSVファイルのパス。
                              'record_id_1', 'record_id_2', 'ground_truth_similar' 列が必要。
        output_dir (str): サンプリング結果のJSONファイルを保存するディレクトリ。
        num_positive_samples (int): サンプリングするPositiveペアの数。
        num_negative_samples (int): サンプリングするNegativeペアの数。
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"入力CSVファイル {input_csv_path} を読み込みました。合計 {len(df)} ペア。")
    except FileNotFoundError:
        print(f"エラー: 入力CSVファイルが見つかりません: {input_csv_path}")
        return
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中にエラーが発生しました: {e}")
        return

    required_columns = ["record_id_1", "record_id_2", "ground_truth_similar"]
    if not all(col in df.columns for col in required_columns):
        print(f"エラー: 入力CSVファイルには必要な列 ({', '.join(required_columns)}) が含まれていません。")
        return

    # Boolean型への変換を試みる (True/False 文字列, 1/0 数値などに対応)
    try:
        # 文字列 'True'/'False' や 'true'/'false' を正しくブール値に変換
        if df["ground_truth_similar"].dtype == "object":
            df["ground_truth_similar"] = df["ground_truth_similar"].str.lower().map({"true": True, "false": False})
        # 数値やブール型の場合はそのままastype(bool)で良い
        df["ground_truth_similar"] = df["ground_truth_similar"].astype(bool)
    except Exception as e:
        print(f"エラー: 'ground_truth_similar' 列をブール値に変換できませんでした。内容を確認してください。詳細: {e}")
        # Trueと評価できるものの数を数えてみる
        try:
            true_like_values = df["ground_truth_similar"].apply(lambda x: str(x).lower() in ["true", "1", "yes"]).sum()
            false_like_values = df["ground_truth_similar"].apply(lambda x: str(x).lower() in ["false", "0", "no"]).sum()
            print(
                f"参考: 'true'/'1'/'yes' に見える値の数: {true_like_values}, 'false'/'0'/'no'に見える値の数: {false_like_values}"
            )
        except:
            pass  # 参考情報の表示に失敗しても処理は続行しない
        return

    positive_pairs_df = df[df["ground_truth_similar"] == True]
    negative_pairs_df = df[df["ground_truth_similar"] == False]

    print(f"読み込まれたPositiveペア数: {len(positive_pairs_df)}")
    print(f"読み込まれたNegativeペア数: {len(negative_pairs_df)}")

    # Positiveペアのサンプリング
    if len(positive_pairs_df) < num_positive_samples:
        print(
            f"警告: 要求されたPositiveサンプル数 ({num_positive_samples}) が利用可能なPositiveペア数 ({len(positive_pairs_df)}) より多いため、利用可能な全ペアを使用します。"
        )
        sampled_positive_df = positive_pairs_df
    else:
        sampled_positive_df = positive_pairs_df.sample(
            n=num_positive_samples, random_state=42
        )  # 再現性のためにrandom_stateを設定

    # Negativeペアのサンプリング
    if len(negative_pairs_df) < num_negative_samples:
        print(
            f"警告: 要求されたNegativeサンプル数 ({num_negative_samples}) が利用可能なNegativeペア数 ({len(negative_pairs_df)}) より多いため、利用可能な全ペアを使用します。"
        )
        sampled_negative_df = negative_pairs_df
    else:
        sampled_negative_df = negative_pairs_df.sample(
            n=num_negative_samples, random_state=42
        )  # 再現性のためにrandom_stateを設定

    # train.py が期待する形式 (リストのリスト) に変換
    # 必ず文字列としてIDを保存する
    sampled_positive_list = [
        [str(row["record_id_1"]), str(row["record_id_2"])] for _, row in sampled_positive_df.iterrows()
    ]
    sampled_negative_list = [
        [str(row["record_id_1"]), str(row["record_id_2"])] for _, row in sampled_negative_df.iterrows()
    ]

    os.makedirs(output_dir, exist_ok=True)

    positive_output_path = os.path.join(output_dir, "positive_sampled_pairs.json")
    negative_output_path = os.path.join(output_dir, "negative_sampled_pairs.json")

    try:
        with open(positive_output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_positive_list, f, ensure_ascii=False, indent=4)
        print(f"{len(sampled_positive_list)} 組のPositiveペアを {positive_output_path} に保存しました。")

        with open(negative_output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_negative_list, f, ensure_ascii=False, indent=4)
        print(f"{len(sampled_negative_list)} 組のNegativeペアを {negative_output_path} に保存しました。")
    except IOError as e:
        print(f"エラー: JSONファイルへの書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"エラー: JSONファイルの保存中に予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="評価結果CSVからファインチューニング用の学習ペアをサンプリングするスクリプト。"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="evaluate_finetuning_performance.py が出力した詳細結果CSVファイルのパス。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="サンプリングされたペアのJSONファイルを保存するディレクトリのパス。",
    )
    parser.add_argument(
        "--num_positive",
        type=int,
        default=1000,
        help="サンプリングするPositiveペアの数 (デフォルト: 1000)。",
    )
    parser.add_argument(
        "--num_negative",
        type=int,
        default=1000,
        help="サンプリングするNegativeペアの数 (デフォルト: 1000)。",
    )
    args = parser.parse_args()

    sample_pairs_from_evaluation_results(args.input_csv, args.output_dir, args.num_positive, args.num_negative)

    print("サンプリング処理完了。")
