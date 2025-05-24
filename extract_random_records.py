import yaml
import random
import argparse
from collections import defaultdict


def extract_random_records_from_yaml(yaml_file_path, num_records_to_extract):
    """
    YAMLファイルからランダムに指定件数のレコードオブジェクトを抽出する。

    Args:
        yaml_file_path (str): レコード情報が含まれるYAMLファイルのパス。
        num_records_to_extract (int): 抽出するレコードの件数。

    Returns:
        list: 抽出されたレコードオブジェクト(dict)のリスト。
              レコード数が足りない場合は、存在する全レコードオブジェクトを返す。
              エラー時は空リストを返す。
    """
    all_record_objects = []
    original_yaml_structure = {}
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            # records 以外のトップレベルキーを保持 (version, type, id, summary, inf_attr など)
            for key, value in data.items():
                if key != "records":
                    original_yaml_structure[key] = value

    except FileNotFoundError:
        print(f"エラー: YAMLファイルが見つかりません: {yaml_file_path}")
        return [], None
    except yaml.YAMLError as e:
        print(f"エラー: YAMLファイルの解析中にエラーが発生しました: {e}")
        return [], None
    except Exception as e:
        print(f"エラー: ファイル読み込み中に予期せぬエラーが発生しました: {e}")
        return [], None

    if "records" not in data or not isinstance(data["records"], dict):
        print("エラー: YAMLファイルに 'records' キーが見つからないか、形式が正しくありません。")
        return [], None

    for record_group_key, record_list in data["records"].items():
        if isinstance(record_list, list):
            for record_item in record_list:
                if isinstance(record_item, dict) and "id" in record_item:
                    all_record_objects.append(record_item)
        # else:
        #     print(f"警告: 'records' の下の {record_group_key} の値がリストではありません。")

    if not all_record_objects:
        print("YAMLファイルから抽出可能なレコードが見つかりませんでした。")
        return [], original_yaml_structure  # レコードがなくても他の構造は返す

    print(f"YAMLファイルから合計 {len(all_record_objects)} 件のレコードオブジェクトを読み込みました。")

    if len(all_record_objects) <= num_records_to_extract:
        print(
            f"{num_records_to_extract}件の抽出要求に対し、全 {len(all_record_objects)} 件のレコードオブジェクトを返します。"
        )
        return all_record_objects, original_yaml_structure
    else:
        extracted_records = random.sample(all_record_objects, num_records_to_extract)
        print(f"{len(extracted_records)} 件のレコードオブジェクトをランダムに抽出しました。")
        return extracted_records, original_yaml_structure


def main():
    parser = argparse.ArgumentParser(
        description="YAMLファイルからランダムにレコードを抽出し、構造を維持して新しいYAMLに出力します。"
    )
    parser.add_argument("yaml_file", help="入力YAMLファイルのパス")
    parser.add_argument("-n", "--num_extract", type=int, default=2000, help="抽出するレコードの件数 (デフォルト: 2000)")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="extracted_records_subset.yml",  # 出力ファイル名のデフォルトを変更
        help="抽出されたレコードを保存するYAMLファイル名 (デフォルト: extracted_records_subset.yml)",
    )

    args = parser.parse_args()

    extracted_record_objects, other_yaml_data = extract_random_records_from_yaml(args.yaml_file, args.num_extract)

    if extracted_record_objects is not None and other_yaml_data is not None:
        if not extracted_record_objects:
            print(
                "抽出されたレコードがありません。YAMLファイルは生成されませんが、元の構造の一部は保持されている可能性があります。"
            )
            # 空の records を持つYAMLを出力することもできるが、ここでは何もしないか、
            # other_yaml_data のみを出力するかを選択できる。
            # other_yaml_data['records'] = {}
            # if not other_yaml_data.get('records') and not extracted_record_objects:
            #    print("レコードも元の構造に関する情報もありません。出力しません。")
            #    return

        # 新しいYAMLデータ構造を構築
        output_data = defaultdict(list)
        for record_obj in extracted_record_objects:
            # レコードオブジェクトが 'cluster_id' を持つと仮定
            # 元のYAMLの records の下のキーが cluster_id であれば、それを使うのが理想だが、
            # ここでは簡単のため、各レコードの cluster_id を使う
            cluster_id_key = record_obj.get("cluster_id", "unknown_cluster")
            output_data[str(cluster_id_key)].append(record_obj)

        final_yaml_output = dict(other_yaml_data)  # records以外のトップレベルキーをコピー
        final_yaml_output["records"] = dict(output_data)

        # summary情報も更新 (num_of_records と num_of_pairs)
        if "summary" in final_yaml_output and isinstance(final_yaml_output["summary"], dict):
            final_yaml_output["summary"]["num_of_records"] = len(extracted_record_objects)
            # num_of_pairs の再計算 (簡単のため、ここでは cluster_id ごとのレコード数を集計)
            new_num_of_pairs = defaultdict(int)
            for cluster_id_key, records_in_cluster in final_yaml_output["records"].items():
                new_num_of_pairs[cluster_id_key] = len(records_in_cluster)
            final_yaml_output["summary"]["num_of_pairs"] = dict(new_num_of_pairs)
            # creation_date, update_date なども更新するならここに追加
            from datetime import datetime

            final_yaml_output["summary"]["update_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if "creation_date" not in final_yaml_output["summary"]:
                final_yaml_output["summary"]["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                yaml.dump(final_yaml_output, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
            print(f"抽出されたレコードをYAML形式で {args.output_file} に保存しました。")
        except IOError as e:
            print(f"エラー: 出力ファイル '{args.output_file}' への書き込み中にエラーが発生しました: {e}")
        except yaml.YAMLError as e:
            print(f"エラー: YAMLデータのシリアライズ中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
