"""
複数の評価レポートテキストファイルから結果を集計し、
レコードカウントごとに個別のサマリーファイル（Markdown, CSV, Excel）を生成する。
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

# pandasとopenpyxlはExcel/CSV出力に必要です
# pip install pandas openpyxl
try:
    import pandas as pd
except ImportError:
    print("このスクリプトの全機能を利用するには、pandasが必要です。\n'pip install pandas' を実行してください。")
    pd = None

try:
    import openpyxl
except ImportError:
    print("Excelファイルを生成するには、openpyxlが必要です。\n'pip install openpyxl' を実行してください。")
    openpyxl = None


def parse_metric_block(block_content):
    """テキストブロックからメトリクスを抽出する"""
    metrics = {}
    # (... implementation is the same as before ...)
    tp_match = re.search(r"TP:\s*(\d+)", block_content)
    fn_match = re.search(r"FN:\s*(\d+)", block_content)
    fp_match = re.search(r"FP:\s*(\d+)", block_content)
    tn_match = re.search(r"TN:\s*(\d+)", block_content)
    if all([tp_match, fn_match, fp_match, tn_match]):
        metrics['TP'] = int(tp_match.group(1))
        metrics['FN'] = int(fn_match.group(1))
        metrics['FP'] = int(fp_match.group(1))
        metrics['TN'] = int(tn_match.group(1))

    patterns = [
        re.compile(
            r"^\s*-\s*適合率:\s*(?P<precision>[\d.]+),"
            r"\s*再現率:\s*(?P<recall>[\d.]+),"
            r"\s*F1:\s*(?P<f1>[\d.]+)",
            re.MULTILINE
        ),
        re.compile(
            r"F1:\s*(?P<f1>[\d.]+),"
            r"\s*Precision:\s*(?P<precision>[\d.]+),"
            r"\s*Recall:\s*(?P<recall>[\d.]+)"
        ),
        re.compile(
            r"適合率:\s*(?P<precision>[\d.]+),"
            r"\s*再現率:\s*(?P<recall>[\d.]+),"
            r"\s*F1:\s*(?P<f1>[\d.]+)"
        ),
        re.compile(
            r"F1-score:\s*(?P<f1>[\d.]+),"
            r"\s*Precision:\s*(?P<precision>[\d.]+),"
            r"\s*Recall:\s*(?P<recall>[\d.]+)"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(block_content)
        if match:
            metrics['F1 Score'] = float(match.group('f1'))
            metrics['Precision'] = float(match.group('precision'))
            metrics['Recall'] = float(match.group('recall'))
            break
    return metrics


def parse_report_file(file_path):
    """単一のレポートファイルを解析して、構造化されたデータを返す"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    record_count_match = re.search(r'(\d+k)', str(file_path))
    record_count = record_count_match.group(1) if record_count_match else 'N/A'

    data_type_match = re.search(
        r"^\s*(?:-\s*)?データタイプ:\s*(.+)$", content, re.MULTILINE
    )
    data_type = 'N/A'
    if data_type_match:
        data_type = data_type_match.group(1).strip()
    else:
        path_str = str(file_path).lower()
        if 'person' in path_str:
            data_type = 'person'
        elif 'music' in path_str:
            data_type = 'music'
        elif 'bib' in path_str:
            data_type = 'bib'
        elif 'walmart-amazon' in path_str:
            data_type = 'walmart-amazon'
        elif 'wdc' in path_str:
            data_type = 'wdc'

    # ファイル名と内容から戦略を決定
    strategy = 'default'
    if ('eval_async_knn' in str(file_path).lower() or
            'knn_graph' in str(file_path).lower() or
            'Selecting戦略' in content):
        strategy = 'selecting'

    sections = re.split(r'## (K近傍ペア評価|全ペア推論評価|Selecting戦略 評価)', content)
    for i in range(1, len(sections), 2):
        scope_title, scope_content = sections[i], sections[i+1]

        current_scope = "Unknown Scope"
        if "K近傍" in scope_title:
            current_scope = "K-Nearest Pairs"
        elif "全ペア推論" in scope_title:
            current_scope = "All-Pairs Inference"
        elif "Selecting戦略" in scope_title:
            current_scope = "Selecting Strategy"

        sub_sections = re.split(r'###\s*(.+?)\s*\n', scope_content)
        for j in range(1, len(sub_sections), 2):
            model_title, block_content = (
                sub_sections[j].strip(), sub_sections[j+1]
            )
            if "モデル" not in model_title:
                continue

            eval_type = "N/A"
            if "Zero-shot" in model_title:
                eval_type = "Zero-shot"
            elif "Few-shot" in model_title:
                eval_type = "Few-shot"
            elif "ファインチューニング前" in model_title:
                eval_type = "Before-FT"
            elif "ファインチューニング後" in model_title:
                eval_type = "After-FT"

            model_name_match = re.search(r'\((.*?)\)', model_title)
            model_name = (model_name_match.group(1) if model_name_match
                          else model_title)

            metrics = parse_metric_block(block_content)
            if metrics:
                results.append({
                    "Record Count": record_count, "Data Type": data_type,
                    "Model": model_name, "Evaluation Type": eval_type,
                    "Evaluation Scope": current_scope, "Strategy": strategy,
                    **metrics
                })
    return results


def format_markdown_table(records, columns):
    """整形されたMarkdownテーブルの文字列リストを返す"""
    if not records:
        return []
    widths = {col: len(col) for col in columns}
    for record in records:
        for col in columns:
            widths[col] = max(widths[col], len(str(record.get(col, ''))))

    header = "| " + " | ".join(
        [col.ljust(widths[col]) for col in columns]
    ) + " |"
    separator = "|-" + "-|-".join(
        ["-" * widths[col] for col in columns]
    ) + "-|"

    rows = [header, separator]
    for record in records:
        row_str = " | ".join(
            [str(record.get(col, '')).ljust(widths[col]) for col in columns]
        )
        rows.append(f"| {row_str} |")
    return rows


def generate_markdown_content(records):
    """指定されたレコード群からMarkdownレポートの文字列を生成する"""
    report_lines = []
    perf_cols = [
        'Model', 'Evaluation Type', 'Strategy',
        'F1 Score', 'Precision', 'Recall'
    ]

    grouped_by_type = defaultdict(list)
    for record in records:
        grouped_by_type[record.get('Data Type')].append(record)

    for data_type in sorted(grouped_by_type.keys()):
        report_lines.append(f"## Data Type: {data_type}\n")

        grouped_by_scope = defaultdict(list)
        for row in grouped_by_type[data_type]:
            grouped_by_scope[row.get('Evaluation Scope')].append(row)

        for scope in sorted(grouped_by_scope.keys()):
            report_lines.append(f"### {scope}\n")

            # F1スコアで降順にソート
            scope_records = sorted(
                grouped_by_scope[scope],
                key=lambda x: x.get('F1 Score', 0.0),
                reverse=True
            )

            report_lines.append("#### Performance Metrics\n")
            report_lines.extend(format_markdown_table(scope_records, perf_cols))
            report_lines.append("\n")

            report_lines.append("#### Confusion Matrix\n")
            for record in scope_records:
                model_info = (
                    f"**Model**: {record.get('Model')}    "
                    f"**Evaluation Type**: {record.get('Evaluation Type')}    "
                    f"**Strategy**: {record.get('Strategy')}\n"
                )
                report_lines.extend([
                    model_info,
                    "| 予測ラベル | Predicted: Positive | Predicted: Negative |",
                    "|---|---|---|",
                    (f"| Actual: Positive | TP: {record.get('TP')} | "
                     f"FN: {record.get('FN')} |"),
                    (f"| Actual: Negative | FP: {record.get('FP')} | "
                     f"TN: {record.get('TN')} |\n")
                ])
            report_lines.append("\n" + "-"*40 + "\n")

    return "\n".join(report_lines)


def save_files(df, output_dir, file_prefix):
    """DataFrameからCSV, Excel, Markdownファイルを保存する"""
    if df.empty:
        return

    output_dir.mkdir(exist_ok=True)

    # --- CSV ---
    csv_path = output_dir / f"{file_prefix}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  - Saved: {csv_path}")

    # --- Excel ---
    if pd and openpyxl:
        excel_path = output_dir / f"{file_prefix}.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All_Data', index=False)

                # Pivot table
                pivot = df.pivot_table(
                    index=[
                        'Data Type', 'Model', 'Evaluation Type', 'Strategy'
                    ],
                    columns='Evaluation Scope',
                    values=[
                        'F1 Score', 'Precision', 'Recall',
                        'TP', 'FN', 'FP', 'TN'
                    ],
                    aggfunc='first'
                )
                pivot.to_excel(writer, sheet_name='Summary_Pivot')
            print(f"  - Saved: {excel_path}")
        except Exception as e:
            print(f"Could not create Excel file for {file_prefix}: {e}")

    # --- Markdown ---
    md_path = output_dir / f"{file_prefix}.md"
    md_content = generate_markdown_content(df.to_dict('records'))
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  - Saved: {md_path}")


def main(args):
    if not pd:
        return

    all_results = []
    for dir_pattern in args.search_dirs:
        for report_file in Path.cwd().glob(f"{dir_pattern}/**/*_report.txt"):
            # bibデータは 'results_bibkyoto' ディレクトリからのみ取得する
            path_str_lower = str(report_file).lower()
            if ('results_bib' in path_str_lower and
                    'results_bibkyoto' not in path_str_lower):
                continue
            all_results.extend(parse_report_file(report_file))

    if not all_results:
        print("\nNo report files found. Exiting.")
        return

    df = pd.DataFrame(all_results)
    if 'Strategy' not in df.columns:
        df['Strategy'] = 'default'
    df['Strategy'] = df['Strategy'].fillna('default')
    # モデルIDと評価スコープ、戦略、数値メトリクスがすべて一致する場合に重複排除
    subset_cols = [
        'Model', 'Evaluation Scope', 'Strategy', 'F1 Score',
        'Precision', 'Recall', 'TP', 'FN', 'FP', 'TN'
    ]
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)

    # --- N/A (テストデータ) とそれ以外を分離 ---
    df_na = df[df['Data Type'] == 'N/A']
    df_main = df[df['Data Type'] != 'N/A']

    output_dir = Path(args.output_dir)

    # --- N/Aデータのファイル保存 ---
    if not df_na.empty:
        print(f"\nProcessing N/A test data ({len(df_na)} entries)...")
        save_files(df_na, output_dir, "evaluation_summary_NA")

    # --- メインデータのファイル保存 (レコードカウントごと) ---
    if not df_main.empty:
        for rc, group_df in df_main.groupby('Record Count'):
            print(
                f"\nProcessing Record Count '{rc}' "
                f"({len(group_df)} entries)..."
            )
            # F1スコアで降順にソート
            sorted_group_df = group_df.sort_values(
                by='F1 Score', ascending=False
            ).reset_index(drop=True)
            # ファイル名に使えない文字を置換
            safe_rc = str(rc).replace('/', '_')
            file_prefix = f"evaluation_summary_{safe_rc}"
            save_files(sorted_group_df, output_dir, file_prefix)

    print("\nAll processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize LLM evaluation results into separate files."
    )
    parser.add_argument(
        '--search-dirs', nargs='+',
        default=[
            'results*', 'run_*', 'openai_embedding_experiment', 'llm_pairs*'
        ],
        help='List of directory patterns to search for reports.'
    )
    parser.add_argument(
        '--output-dir', default='ground_results',
        help='Directory to save the summary files.'
    )
    args = parser.parse_args()
    main(args)
