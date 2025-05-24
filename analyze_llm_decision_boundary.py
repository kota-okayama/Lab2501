import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルのパス
csv_file_path = "pipeline-output30k-30k/detailed_evaluation_results.csv"
# LLMの予測ラベルが格納されている列名
predicted_label_column = "predicted_similar_before"
# LLMの予測スコアが格納されている列名
score_column = "score_before"

try:
    df = pd.read_csv(csv_file_path)
    print(f"CSVファイル '{csv_file_path}' を読み込みました。総ペア数: {len(df)}")

    # 予測ラベル列の型をブール型に変換
    if df[predicted_label_column].dtype == "object":
        df[predicted_label_column] = (
            df[predicted_label_column]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .astype(bool)
        )
    else:
        df[predicted_label_column] = df[predicted_label_column].astype(bool)

    # スコア列を数値型に変換
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")

    df_cleaned = df.dropna(subset=[score_column, predicted_label_column])
    if len(df_cleaned) < len(df):
        print(
            f"警告: '{score_column}' または '{predicted_label_column}' が不正な {len(df) - len(df_cleaned)} ペアを除外しました。"
        )

    if df_cleaned.empty:
        print("エラー: 分析可能なデータがありません。")
    else:
        df_llm_true = df_cleaned[df_cleaned[predicted_label_column] == True]
        print(f"\n--- LLM予測: 一致 ({predicted_label_column}=True) のペアの '{score_column}' 分析 ---")
        if not df_llm_true.empty:
            print(df_llm_true[score_column].describe())
            print(f"  スコア範囲: {df_llm_true[score_column].min()} - {df_llm_true[score_column].max()}")
        else:
            print("LLM予測がTrueのデータがありません。")

        df_llm_false = df_cleaned[df_cleaned[predicted_label_column] == False]
        print(f"\n--- LLM予測: 不一致 ({predicted_label_column}=False) のペアの '{score_column}' 分析 ---")
        if not df_llm_false.empty:
            print(df_llm_false[score_column].describe())
            print(f"  スコア範囲: {df_llm_false[score_column].min()} - {df_llm_false[score_column].max()}")
        else:
            print("LLM予測がFalseのデータがありません。")

        plt.figure(figsize=(12, 6))
        sns.histplot(
            df_llm_true[score_column],
            color="lightgreen",
            label=f"LLM Pred: True",
            kde=True,
            stat="density",
            common_norm=False,
        )
        sns.histplot(
            df_llm_false[score_column],
            color="lightcoral",
            label=f"LLM Pred: False",
            kde=True,
            stat="density",
            common_norm=False,
        )
        plt.title(f"'{score_column}' の分布 (LLM予測 '{predicted_label_column}' 別)")
        plt.xlabel(f"'{score_column}'")
        plt.ylabel("密度")
        plt.legend()
        plt.grid(axis="y", alpha=0.75)

        plot_file_path = "llm_score_by_prediction_plot.png"
        plt.savefig(plot_file_path)
        print(f"\nLLM予測別スコア分布のヒストグラムを '{plot_file_path}' に保存しました。")
        plt.close()

        # LLMがTrueと予測したスコアの最小値と、Falseと予測したスコアの最大値の間のギャップやオーバーラップを確認
        if not df_llm_true.empty and not df_llm_false.empty:
            min_score_for_llm_true = df_llm_true[score_column].min()
            max_score_for_llm_false = df_llm_false[score_column].max()
            print(f"\nLLMがTrueと予測したスコアの最小値: {min_score_for_llm_true}")
            print(f"LLMがFalseと予測したスコアの最大値: {max_score_for_llm_false}")

            if max_score_for_llm_false < min_score_for_llm_true:
                print(
                    f"  -> LLMは明確な閾値 ({max_score_for_llm_false} と {min_score_for_llm_true} の間) を持っている可能性があります。"
                )
                print(
                    f"  この境界領域 ({max_score_for_llm_false - 0.1:.2f} ～ {min_score_for_llm_true + 0.1:.2f} など) からのサンプリングが有効かもしれません。"
                )
            else:
                print(
                    f"  -> スコア {min_score_for_llm_true:.2f} ～ {max_score_for_llm_false:.2f} の範囲では、LLMの予測がTrueとFalseでオーバーラップしています。"
                )
                print(
                    f"  このオーバーラップ領域からのサンプリングが特に有効かもしれません (モデルが混乱している領域)。"
                )

        # 0.5に近いスコアを持つペアの数
        near_0_5_count = df_cleaned[(df_cleaned[score_column] >= 0.4) & (df_cleaned[score_column] <= 0.6)].shape[0]
        print(f"\n'{score_column}' が 0.4～0.6 の範囲にあるペアの総数 (LLM予測問わず): {near_0_5_count} ペア")


except FileNotFoundError:
    print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
except Exception as e:
    print(f"エラー: 処理中に予期せぬエラーが発生しました: {e}")
