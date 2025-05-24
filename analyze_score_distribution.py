import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルのパス (適宜変更してください)
csv_file_path = "pipeline-output30k-30k/detailed_evaluation_results.csv"

try:
    df = pd.read_csv(csv_file_path)
    print(f"CSVファイル '{csv_file_path}' を読み込みました。総ペア数: {len(df)}")

    # 'ground_truth_similar' 列の型をブール型に変換 (True/False 文字列を想定)
    # より堅牢にするために、'true'/'false' (小文字) や 1/0 も考慮
    if df["ground_truth_similar"].dtype == "object":
        df["ground_truth_similar"] = (
            df["ground_truth_similar"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .astype(bool)
        )
    else:
        df["ground_truth_similar"] = df["ground_truth_similar"].astype(bool)

    # 'score_before' 列を数値型に変換 (エラー時はNaNにする)
    df["score_before"] = pd.to_numeric(df["score_before"], errors="coerce")

    # NaNが含まれる行を削除 (score_before または ground_truth_similar が不正な場合)
    df_cleaned = df.dropna(subset=["score_before", "ground_truth_similar"])
    if len(df_cleaned) < len(df):
        print(
            f"警告: 'score_before' または 'ground_truth_similar' が不正な {len(df) - len(df_cleaned)} ペアを除外しました。"
        )

    if df_cleaned.empty:
        print("エラー: 分析可能なデータがありません。CSVファイルの内容を確認してください。")
    else:
        # --- 一致ペア (Ground Truth = True) の分析 ---
        df_true = df_cleaned[df_cleaned["ground_truth_similar"] == True]
        print("\n--- Ground Truth: 一致 (True) のペアの 'score_before' 分析 ---")
        if not df_true.empty:
            print(df_true["score_before"].describe())
        else:
            print("一致ペアのデータがありません。")

        # --- 不一致ペア (Ground Truth = False) の分析 ---
        df_false = df_cleaned[df_cleaned["ground_truth_similar"] == False]
        print("\n--- Ground Truth: 不一致 (False) のペアの 'score_before' 分析 ---")
        if not df_false.empty:
            print(df_false["score_before"].describe())
        else:
            print("不一致ペアのデータがありません。")

        # --- ヒストグラムによる可視化 ---
        plt.figure(figsize=(12, 6))
        sns.histplot(
            df_true["score_before"],
            color="skyblue",
            label="GT: True (一致)",
            kde=True,
            stat="density",
            common_norm=False,
        )
        sns.histplot(
            df_false["score_before"],
            color="salmon",
            label="GT: False (不一致)",
            kde=True,
            stat="density",
            common_norm=False,
        )
        plt.title("'score_before' の分布 (Ground Truth 別)")
        plt.xlabel("'score_before'")
        plt.ylabel("密度")
        plt.legend()
        plt.grid(axis="y", alpha=0.75)

        # プロットをファイルに保存 (表示する代わりに)
        plot_file_path = "score_distribution_plot.png"
        plt.savefig(plot_file_path)
        print(f"\nスコア分布のヒストグラムを '{plot_file_path}' に保存しました。")
        plt.close()  # プロットウィンドウが表示されないように閉じる

        print("\n分析完了。")
        if not df_true.empty and not df_false.empty:
            # 0.5に近いスコアの数を簡単に集計
            true_near_0_5 = df_true[(df_true["score_before"] >= 0.4) & (df_true["score_before"] <= 0.6)].shape[0]
            false_near_0_5 = df_false[(df_false["score_before"] >= 0.4) & (df_false["score_before"] <= 0.6)].shape[0]
            print(f"\n'score_before' が 0.4～0.6 の範囲にあるペアの数:")
            print(f"  Ground Truth が True: {true_near_0_5} ペア")
            print(f"  Ground Truth が False: {false_near_0_5} ペア")

        # --- スコア範囲ごとの集計表作成 ---
        if not df_cleaned.empty:
            bins = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
            labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

            df_cleaned.loc[:, "score_bin"] = pd.cut(
                df_cleaned["score_before"], bins=bins, labels=labels, right=True, include_lowest=True
            )

            summary_table = pd.DataFrame(index=labels + ["合計"])

            true_counts = (
                df_cleaned[df_cleaned["ground_truth_similar"] == True]["score_bin"].value_counts().sort_index()
            )
            false_counts = (
                df_cleaned[df_cleaned["ground_truth_similar"] == False]["score_bin"].value_counts().sort_index()
            )

            summary_table["GT: True (一致) のペア数"] = true_counts
            summary_table["GT: False (不一致) のペア数"] = false_counts
            summary_table = summary_table.fillna(0).astype(int)

            summary_table["この範囲の総ペア数"] = (
                summary_table["GT: True (一致) のペア数"] + summary_table["GT: False (不一致) のペア数"]
            )

            # 全体での True/False の総数
            total_true = summary_table["GT: True (一致) のペア数"].sum()
            total_false = summary_table["GT: False (不一致) のペア数"].sum()
            total_all = summary_table["この範囲の総ペア数"].sum()

            # 割合計算 (分母が0にならないように注意)
            summary_table["GT: True の割合 (%)"] = summary_table.apply(
                lambda row: (
                    (row["GT: True (一致) のペア数"] / row["この範囲の総ペア数"] * 100)
                    if row["この範囲の総ペア数"] > 0
                    else 0
                ),
                axis=1,
            ).round(2)
            summary_table["GT: False の割合 (%)"] = summary_table.apply(
                lambda row: (
                    (row["GT: False (不一致) のペア数"] / row["この範囲の総ペア数"] * 100)
                    if row["この範囲の総ペア数"] > 0
                    else 0
                ),
                axis=1,
            ).round(2)

            # 合計行の計算
            summary_table.loc["合計", "GT: True (一致) のペア数"] = total_true
            summary_table.loc["合計", "GT: False (不一致) のペア数"] = total_false
            summary_table.loc["合計", "この範囲の総ペア数"] = total_all
            summary_table.loc["合計", "GT: True の割合 (%)"] = (total_true / total_all * 100) if total_all > 0 else 0
            summary_table.loc["合計", "GT: False の割合 (%)"] = (total_false / total_all * 100) if total_all > 0 else 0
            summary_table.loc["合計", ["GT: True の割合 (%)", "GT: False の割合 (%)"]] = summary_table.loc[
                "合計", ["GT: True の割合 (%)", "GT: False の割合 (%)"]
            ].round(2)

            print("\n\n--- 'score_before' と Ground Truth の関係 (集計表) ---")
            # コンソール表示用に一部列を調整（必要なら）
            # print(summary_table.to_string()) # to_string() で全ての列を表示
            # 表示する列を絞る場合:
            display_columns = [
                "GT: True (一致) のペア数",
                "GT: False (不一致) のペア数",
                "この範囲の総ペア数",
                "GT: True の割合 (%)",
                "GT: False の割合 (%)",
            ]
            print(summary_table[display_columns].to_markdown(index=True))

            summary_table_csv_path = "score_distribution_summary_table.csv"
            try:
                summary_table.to_csv(summary_table_csv_path, encoding="utf-8-sig")
                print(f"\n集計表を '{summary_table_csv_path}' に保存しました。")
            except Exception as e:
                print(f"エラー: 集計表のCSV保存に失敗: {e}")


except FileNotFoundError:
    print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
except Exception as e:
    print(f"エラー: 処理中に予期せぬエラーが発生しました: {e}")
