import pandas as pd
import yaml
import argparse
from datetime import datetime
import uuid
from collections import defaultdict
import random
import os
import json
import glob

def read_wdc_input_path(input_path):
    """
    Reads all .json files from a directory, or a single specified .json file.
    """
    records = []
    
    if os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, '*.json'))
        print(f"Found {len(json_files)} JSON files to process in directory '{input_path}'...")
    elif os.path.isfile(input_path):
        json_files = [input_path]
        print(f"Processing single file: '{input_path}'")
    else:
        print(f"Error: Input path '{input_path}' is not a valid file or directory.")
        return []

    for file_path in json_files:
        print(f"  - Reading {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Could not process {file_path}. Error: {e}. Skipping.")
    
    print(f"Total records read: {len(records)}")
    return records

def write_yml_file(output_file, inf_attr, records_by_cluster):
    """YAMLファイル書き出し処理を共通化"""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_of_records = sum(len(records) for records in records_by_cluster.values())
    cluster_sizes = defaultdict(int)
    for cluster_id in records_by_cluster:
        size = len(records_by_cluster[cluster_id])
        cluster_sizes[size] += 1

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    summary = {
        'creation_date': now,
        'update_date': now,
        'num_of_records': num_of_records,
        'num_of_pairs': dict(sorted(cluster_sizes.items())),
        'config_match': None,
        'config_mismatch': None,
    }

    output_data = {
        'summary': summary,
        'inf_attr': inf_attr,
        'records': dict(records_by_cluster)
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, sort_keys=False, allow_unicode=True, indent=2)
        print(f"Successfully created {output_file}")
    except Exception as e:
        print(f"Error writing YML file: {e}")

def convert_wdc_to_yml(
    input_path,
    partitions=None,
    attributes_to_include=None, 
    random_seed=None,
    order_by='random',
    training_partitions=0,
    top_n_for_training=0,
    output_dir=None
):
    """
    WDCデータセットをクラスタリングし、--partitionで指定された定義に基づいて
    複数の排反なサブセットグループに一括で分割する。
    """
    all_raw_records = read_wdc_input_path(input_path)
    if not all_raw_records:
        return

    # 'id' と 'cluster_id' が存在しないレコードを除外
    validated_records = [r for r in all_raw_records if 'id' in r and 'cluster_id' in r]
    print(f"Validated {len(validated_records)} records with 'id' and 'cluster_id'.")
    
    # inf_attrの決定 (最初のレコードから全属性を取得)
    if attributes_to_include is None:
        default_attrs = list(validated_records[0].keys())
        # ID関連と不要な属性を除外
        attributes_to_include = [attr for attr in default_attrs if attr not in ['id', 'cluster_id', 'label', 'unseen']]
        print(f"Auto-detected attributes to include: {attributes_to_include}")

    inf_attr = {col: 'TEXT_EN' for col in attributes_to_include}

    # --- レコードをクラスタごとに整理 ---
    all_records_by_cluster = defaultdict(list)
    for raw_rec in validated_records:
        cluster_id = str(raw_rec['cluster_id'])
        
        filtered_data = {k: str(v) for k, v in raw_rec.items() if k in attributes_to_include}
        
        record = {
            'id': str(uuid.uuid4()), # 新しいユニークIDを発行
            'cluster_id': cluster_id,
            'data': filtered_data
        }
        all_records_by_cluster[cluster_id].append(record)

    # --- ここから下の分割ロジックはwalmart_amazon版と同一 ---
    if not partitions:
        print("Error: At least one --partition must be specified.")
        return

    if random_seed is not None:
        random.seed(random_seed)
    
    all_cluster_ids = list(all_records_by_cluster.keys())
    
    if order_by == 'size' and training_partitions > 0:
        # ( ... ハイブリッド戦略ロジック ... )
        multi_record_clusters = [cid for cid in all_cluster_ids if len(all_records_by_cluster[cid]) > 1]
        single_record_clusters = [cid for cid in all_cluster_ids if len(all_records_by_cluster[cid]) == 1]
        
        multi_record_clusters.sort(key=lambda cid: len(all_records_by_cluster[cid]), reverse=True)
        random.shuffle(single_record_clusters)

        if top_n_for_training > 0:
            guaranteed_clusters = multi_record_clusters[:top_n_for_training]
            remaining_clusters = multi_record_clusters[top_n_for_training:] + single_record_clusters
            random.shuffle(remaining_clusters)
            all_cluster_ids = guaranteed_clusters + remaining_clusters
        else:
             all_cluster_ids = multi_record_clusters + single_record_clusters
    
    else:
        if order_by == 'size':
            all_cluster_ids.sort(key=lambda cid: len(all_records_by_cluster[cid]), reverse=True)
        else:
            random.shuffle(all_cluster_ids)

    master_cluster_idx = 0
    for i, partition_def in enumerate(partitions):
        try:
            prefix, count_str, size_str = partition_def.split(':')
            count = int(count_str)
            size = int(size_str)
        except ValueError:
            print(f"Error: Invalid partition format '{partition_def}'. Expected 'prefix:count:size'.")
            continue
        
        for j in range(count):
            start_idx = master_cluster_idx
            current_record_count = 0
            while current_record_count < size and master_cluster_idx < len(all_cluster_ids):
                cid = all_cluster_ids[master_cluster_idx]
                current_record_count += len(all_records_by_cluster[cid])
                master_cluster_idx += 1
            
            subset_cluster_ids = all_cluster_ids[start_idx:master_cluster_idx]
            current_subset_clusters = {cid: all_records_by_cluster[cid] for cid in subset_cluster_ids}

            if not current_subset_clusters:
                 print(f"Warning: Ran out of clusters. Stopping partition generation for prefix '{prefix}'.")
                 break
            
            output_filename = f"{prefix}_{j+1}.yml"
            
            if output_dir:
                output_filepath = os.path.join(output_dir, output_filename)
            else:
                output_filepath = output_filename

            write_yml_file(output_filepath, inf_attr, current_subset_clusters)
            print(f"  - Created subset {j+1}: {sum(len(v) for v in current_subset_clusters.values())} records in {len(current_subset_clusters)} clusters.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert WDC JSON dataset into partitioned YML subsets.')
    parser.add_argument('input_path', type=str, 
                        help='Path to the directory containing WDC JSON files or a single JSON file.')
    parser.add_argument('--partition', action='append', required=True,
                        help='Define a group of subsets. Format: "prefix:count:size". Can be specified multiple times.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the output YML files (default: current directory).')
    parser.add_argument('--attributes', type=str, 
                        help='Comma-separated list of attributes to include. Auto-detected if not provided.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility of shuffling (default: 42).')
    parser.add_argument('--order_by', type=str, default='random', choices=['random', 'size'],
                        help='Order of clusters for partitioning: "random" (default) or "size" (descending).')
    parser.add_argument('--training_partitions', type=int, default=0,
                        help='With --order_by size, specifies how many of the first partition definitions are for training.')
    parser.add_argument('--top_n_for_training', type=int, default=0,
                        help='With --order_by size, guarantees the top N largest clusters are included in the training partitions.')
    
    args = parser.parse_args()
    
    attributes = args.attributes.split(',') if args.attributes else None
    
    convert_wdc_to_yml(
        args.input_path,
        partitions=args.partition,
        attributes_to_include=attributes,
        random_seed=args.random_seed,
        order_by=args.order_by,
        training_partitions=args.training_partitions,
        top_n_for_training=args.top_n_for_training,
        output_dir=args.output_dir
    )
