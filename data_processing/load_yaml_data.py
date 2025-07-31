import yaml
import os

def load_yaml_data(yaml_path):
    """
    YAMLファイルを読み込み、レコードリストとinf_attrを返す。
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        records = data.get('records', {})
        inf_attr = data.get('inf_attr', {})

        records_list = []
        for record_items in records.values():
            for item in record_items:
                entry = {
                    "record_id": item.get('id'),
                    "cluster_id": item.get('cluster_id'),
                    "data": item.get('data', {})
                }
                records_list.append(entry)
        
        return records_list, inf_attr

    except FileNotFoundError:
        print(f"Error: The file {yaml_path} was not found.")
        return [], {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")
        return [], {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], {}


def load_bibliographic_data(yaml_path):
    """
    書誌情報YAMLファイルを読み込み、各レコードをリストとして返す。
    各レコードにはrecord_idとcluster_idが付与される。
    下位互換性のために残されている。
    """
    records_list, _ = load_yaml_data(yaml_path)
    return records_list


def load_and_group_bibliographic_data(yaml_file_path):
    """
    書誌情報YAMLファイルを読み込み、クラスタIDでグループ化された辞書を返す。
    """
    if not os.path.exists(yaml_file_path):
        print(f"Error: File not found at {yaml_file_path}")
        return {}

    records_list, _ = load_yaml_data(yaml_file_path)
    grouped_data = {}
    for record in records_list:
        cluster_id = record.get("cluster_id")
        if cluster_id:
            if cluster_id not in grouped_data:
                grouped_data[cluster_id] = []
            grouped_data[cluster_id].append(record.get("data", {}))
    return grouped_data


if __name__ == "__main__":
    # YAMLファイルのパス (ユーザーの環境に合わせて変更してください)
    # 例: 'benchmark/bib_japan_20241024/1k/record.yml'
    #      'F:/lab/Lab2411-archive/benchmark/bib_japan_20241024/1k/record.yml' (WSLからWindowsパスを参照する場合)

    # WSL内の絶対パス、またはこのスクリプトからの相対パスを指定
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"

    # WSL環境でWindowsの絶対パスを参照する場合は、パスの先頭に /mnt/ をつけ、ドライブレターを小文字にします。
    # 例: Fドライブの /lab/Lab2411-archive/... の場合
    # windows_absolute_path = 'F:/lab/Lab2411-archive/benchmark/bib_japan_20241024/1k/record.yml'
    # if os.name != 'nt': # WSL (Linux)環境かどうかを簡易的に判定
    #     yaml_path = '/' + windows_absolute_path.replace(':', '').replace('\\', '/').replace('\', '/')
    #     yaml_path = '/mnt/' + yaml_path[1].lower() + yaml_path[2:]
    # else: # Windows環境の場合 (直接実行することは少ないが念のため)
    #     yaml_path = windows_absolute_path

    print(f"Attempting to load data from: {yaml_path}")

    bibliographic_records = load_bibliographic_data(yaml_path)

    if bibliographic_records:
        print(f"Successfully loaded {len(bibliographic_records)} records.")

        # 最初の5件のレコード情報を表示
        print("\nFirst 5 records:")
        for i, record in enumerate(bibliographic_records[:5]):
            print(f"--- Record {i+1} ---")
            print(f"  Record ID: {record.get('record_id')}")
            print(f"  Cluster ID: {record.get('cluster_id')}")
            print(f"  Title: {record.get('data', {}).get('bib1_title')}")
            print(f"  Author: {record.get('data', {}).get('bib1_author')}")
    else:
        print("Failed to load records.")

    # PyYAMLがインストールされていない場合に備えてメッセージ
    try:
        import yaml
    except ImportError:
        print("\n--------------------------------------")
        print("PyYAML library is not installed. Please install it by running:")
        print("pip install PyYAML")
        print("---------------------------------------")
