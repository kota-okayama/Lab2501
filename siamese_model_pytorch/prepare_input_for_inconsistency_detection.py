import pandas as pd
import os

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
DETAILED_RESULTS_FILENAME = "detailed_evaluation_results.csv"
DETAILED_RESULTS_PATH = os.path.join(EVALUATION_RESULTS_DIR, DETAILED_RESULTS_FILENAME)

OUTPUT_FILENAME = "llm_ft_results_for_inconsistency.csv"  # detect_inconsistent_triangles.py が読み込むファイル名
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)

# --- ---


def main():
    print(f"読み込み元: {DETAILED_RESULTS_PATH}")
    if not os.path.exists(DETAILED_RESULTS_PATH):
        print(
            f"エラー: {DETAILED_RESULTS_PATH} が見つかりません。先に evaluate_finetuning_performance.py を実行してください。"
        )
        return

    try:
        df = pd.read_csv(DETAILED_RESULTS_PATH)
    except Exception as e:
        print(f"エラー: {DETAILED_RESULTS_PATH} の読み込みに失敗しました: {e}")
        return

    # 必要な列が存在するか確認
    required_cols = ["record_id_1", "record_id_2", "score_before"]
    if not all(col in df.columns for col in required_cols):
        print(f"エラー: {DETAILED_RESULTS_PATH} に必要な列 {required_cols} のいずれかがありません。")
        print(f"存在する列: {df.columns.tolist()}")
        return

    output_df = df[["record_id_1", "record_id_2", "score_before"]].copy()
    output_df.rename(columns={"score_before": "llm_similarity_score"}, inplace=True)

    def to_float_or_zero(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    output_df["llm_similarity_score"] = output_df["llm_similarity_score"].apply(to_float_or_zero)

    try:
        output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"変換後のファイルを {OUTPUT_PATH} に保存しました。")
        print(f"  - レコード数: {len(output_df)}")
        print(f"  - 列: {output_df.columns.tolist()}")
    except Exception as e:
        print(f"エラー: {OUTPUT_PATH} への保存に失敗しました: {e}")


if __name__ == "__main__":
    main()
