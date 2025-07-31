import pandas as pd
import argparse
from pathlib import Path
import re


def format_confusion_matrix(row):
    """DataFrameの行から混合行列のテキストブロックを作成する"""
    try:
        tp = int(row.get('TP', 0))
        fn = int(row.get('FN', 0))
        fp = int(row.get('FP', 0))
        tn = int(row.get('TN', 0))
    except (ValueError, TypeError):
        return "  N/A"

    matrix_str = (
        f"    +-----------------+-----------------+\n"
        f"    | TP: {tp:<12d} | FN: {fn:<12d} |\n"
        f"    | FP: {fp:<12d} | TN: {tn:<12d} |\n"
        f"    +-----------------+-----------------+"
    )
    return matrix_str


def append_group_to_report(group_data, report_content):
    """指定されたグループのデータを整形してレポートコンテンツに追加する"""
    (record_count, data_type), group = group_data
    report_content.append(
        f"==========================================================\n"
    )
    report_content.append(
        f" Record Count: {record_count} | Data Type: {data_type}\n"
    )
    report_content.append(
        f"==========================================================\n\n"
    )

    # グループ内でさらにスコープでグループ化
    scope_grouped = group.groupby('Evaluation Scope')
    for scope, scope_group in scope_grouped:
        report_content.append(f"## {scope}\n\n")

        # テーブルヘッダー
        header = (
            f"{'Model':<40} | {'Type':<12} | "
            f"{'F1':<7} | {'ARI':<7} | "
            f"Confusion Matrix\n"
        )
        separator = (
            f"{'-'*40}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-"
            f"{'-'*40}\n"
        )
        report_content.append(header)
        report_content.append(separator)

        # テーブルの行
        for _, row in scope_group.iterrows():
            f1_val = row.get('F1 Score', '-')
            ari_val = row.get('ARI', '-')
            model_val = row.get('Model', 'N/A')
            eval_type_val = row.get('Evaluation Type', 'N/A')

            f1 = f"{f1_val:.4f}" if isinstance(f1_val, float) else str(f1_val)
            ari = f"{ari_val:.4f}" if isinstance(ari_val, float) else str(ari_val)

            row_str = (
                f"{model_val:<40} | {eval_type_val:<12} | "
                f"{f1:<7} | {ari:<7} |\n"
            )
            report_content.append(row_str)

            # 混合行列
            cm_str = format_confusion_matrix(row)
            report_content.append(f"{cm_str}\n\n")

        report_content.append("\n")


def main(args):
    """メイン処理"""
    input_csv = Path(args.input_csv)
    output_txt = Path(args.output_txt)
    output_no_count_txt = output_txt.with_name(
        f"{output_txt.stem}_no_count{output_txt.suffix}"
    )

    if not input_csv.exists():
        print(f"Error: Input CSV file not found at '{input_csv}'")
        print("Please run summarize_evaluation_results.py first.")
        return

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # NaN値をハイフンに置換
    df.fillna('-', inplace=True)

    report_content_main = ["# Evaluation Summary Report\n\n"]
    report_content_no_count = [
        "# Evaluation Summary Report (No Record Count)\n\n"
    ]

    # グループ化し、レコードカウントの有無で分離する
    groups = list(df.groupby(['Record Count', 'Data Type']))
    counted_groups = []
    na_groups = []

    for group_data in groups:
        (record_count, _), _ = group_data
        if str(record_count) == 'N/A' or str(record_count) == '-':
            na_groups.append(group_data)
        else:
            counted_groups.append(group_data)

    # レコードカウントの数値でソートするためのキー関数
    def get_sort_key(group_item):
        (record_count, _), _ = group_item
        numeric_part = re.search(r'(\d+)', str(record_count))
        return int(numeric_part.group(1)) if numeric_part else float('inf')

    counted_groups.sort(key=get_sort_key)

    # メインレポートを生成
    if counted_groups:
        for group_data in counted_groups:
            append_group_to_report(group_data, report_content_main)
    else:
        report_content_main.append("No reports with specific record counts found.\n")

    # レコードカウントなしのレポートを生成
    if na_groups:
        for group_data in na_groups:
            append_group_to_report(group_data, report_content_no_count)

    # メインファイルを書き込み
    output_txt.parent.mkdir(exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.writelines(report_content_main)
    print(f"Successfully created human-readable summary: {output_txt}")

    # レコードカウントなしのファイルも書き込み
    if na_groups:
        with open(output_no_count_txt, 'w', encoding='utf-8') as f:
            f.writelines(report_content_no_count)
        print(
            "Successfully created summary for reports with no record count: "
            f"{output_no_count_txt}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format the summary CSV into a human-readable text report."
    )
    parser.add_argument(
        '--input-csv',
        default='ground_results/evaluation_summary.csv',
        help='Path to the input summary CSV file.'
    )
    parser.add_argument(
        '--output-txt',
        default='ground_results/evaluation_summary_readable.txt',
        help='Path to the output readable text file.'
    )

    args = parser.parse_args()
    main(args) 