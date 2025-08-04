import csv
import yaml
import os
import argparse
import pandas as pd

# グローバル変数として書誌データを保持
BIB_DATA = {}


def load_bib_data(yaml_path):
    """
    record.ymlから書誌データをロードする。
    BIB_DATA をグローバルに設定する。
    (evaluate_finetuning_performance.py の load_bib_data_and_gt_clusters を簡略化)
    """
    global BIB_DATA
    BIB_DATA = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        return False
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            if isinstance(possible_records_dict, dict):
                for key, value_list in possible_records_dict.items():
                    if key in ["version", "type", "id", "summary", "inf_attr"] and possible_records_dict is all_data:
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])
                                if "data" in record and isinstance(record["data"], dict):
                                    BIB_DATA[record_id_str] = record["data"]
                                else:
                                    record_data_candidate = {
                                        k_rec: v_rec
                                        for k_rec, v_rec in record.items()
                                        if k_rec not in ["id", "cluster_id"]
                                    }
                                    if record_data_candidate:
                                        BIB_DATA[record_id_str] = record_data_candidate

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。")
            return False

        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        return True

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
        return False
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラー: {e}")
        return False


def get_specific_bib_details(record_id):
    """
    指定されたレコードIDの書誌情報から特定のフィールドを抽出する。
    """
    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return {"title": "情報なし", "authors": "情報なし", "pubdate": "情報なし", "publisher": "情報なし"}

    title = bib_details.get("title", bib_details.get("bib1_title", "タイトル不明"))

    authors_list = bib_details.get("author", bib_details.get("bib1_author", []))
    authors_str = ""
    if isinstance(authors_list, list):
        authors_str = ", ".join(authors_list) if authors_list else "著者不明"
    elif isinstance(authors_list, str):
        authors_str = authors_list if authors_list else "著者不明"
    else:
        authors_str = "著者不明"

    pubdate = bib_details.get("pubdate", bib_details.get("bib1_pubdate", "出版日不明"))
    publisher = bib_details.get("publisher", bib_details.get("bib1_publisher", "出版社不明"))

    return {"title": title, "authors": authors_str, "pubdate": pubdate, "publisher": publisher}


def main():
    parser = argparse.ArgumentParser(description="detailed_evaluation_results.csv に書誌情報を付加するスクリプト")
    parser.add_argument(
        "--detailed_csv_path", type=str, required=True, help="入力CSVファイル (detailed_evaluation_results.csv) のパス"
    )
    parser.add_argument("--record_yaml_path", type=str, required=True, help="書誌情報YAMLファイルのパス")
    parser.add_argument(
        "--output_csv_path", type=str, required=True, help="書誌情報を付加した結果を出力する新しいCSVファイルのパス"
    )

    args = parser.parse_args()

    print(f"書誌YAMLファイル: {args.record_yaml_path}")
    if not load_bib_data(args.record_yaml_path):
        print("書誌データのロードに失敗したため、処理を中断します。")
        return

    print(f"入力詳細結果CSV: {args.detailed_csv_path}")
    try:
        df = pd.read_csv(args.detailed_csv_path)
    except FileNotFoundError:
        print(f"エラー: 入力CSVファイルが見つかりません: {args.detailed_csv_path}")
        return
    except Exception as e:
        print(f"エラー: 入力CSVファイルの読み込み中にエラー: {e}")
        return

    new_columns_1 = []
    new_columns_2 = []

    print("各ペアに書誌情報を付加しています...")
    for index, row in df.iterrows():
        details_1 = get_specific_bib_details(row["record_id_1"])
        details_2 = get_specific_bib_details(row["record_id_2"])

        new_columns_1.append(details_1)
        new_columns_2.append(details_2)

        if (index + 1) % 1000 == 0:
            print(f"  {index + 1} / {len(df)} ペア処理完了...")

    df_details_1 = pd.DataFrame(new_columns_1).add_prefix("record1_")
    df_details_2 = pd.DataFrame(new_columns_2).add_prefix("record2_")

    # record_id_1 と record_id_2 の後に追加したい
    # 元の列のリストを取得
    original_columns = df.columns.tolist()

    # 挿入位置を見つける (record_id_2 の次)
    try:
        insert_pos = original_columns.index("record_id_2") + 1
    except ValueError:  # record_id_2 がない場合は最初に追加（ありえないはずだが念のため）
        insert_pos = 0

    # record1 の詳細を追加
    for col_name in reversed(df_details_1.columns):  # reversed で挿入順を維持
        df.insert(insert_pos, col_name, df_details_1[col_name])

    # record2 の詳細を追加 (record1 の詳細のさらに後)
    insert_pos_rec2 = insert_pos + len(df_details_1.columns)
    for col_name in reversed(df_details_2.columns):
        df.insert(insert_pos_rec2, col_name, df_details_2[col_name])

    print(f"出力CSVファイル: {args.output_csv_path}")
    try:
        df.to_csv(args.output_csv_path, index=False, encoding="utf-8-sig")
        print(f"処理結果を {args.output_csv_path} に保存しました。")
    except Exception as e:
        print(f"エラー: 出力CSVファイルの保存中にエラー: {e}")

    print("処理完了。")


if __name__ == "__main__":
    main()
