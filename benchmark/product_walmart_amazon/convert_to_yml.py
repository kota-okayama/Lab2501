import pandas as pd
import yaml
import argparse
from datetime import datetime
import uuid
from collections import defaultdict
import random
import os
import csv

def read_dataset_robustly(dataset_file):
    """
    Parses the malformed dataset.tsv robustly. It reads the file line-by-line,
    and skips any row where the number of columns does not match the header.
    """
    records = []
    skipped_count = 0
    with open(dataset_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        if header_line.startswith('\ufeff'):
            header_line = header_line[1:]
        header = header_line.split('\t')
        num_expected_cols = len(header)
        
        for i, line in enumerate(f):
            if not line.strip():
                continue # Skip empty lines
            
            row_parts = line.strip().split('\t')
            
            if len(row_parts) == num_expected_cols:
                records.append(dict(zip(header, row_parts)))
            else:
                if skipped_count < 10: # To avoid flooding the console, show first 10 warnings
                    print(f"Warning: Skipping malformed row {i+2} in {dataset_file}. Expected {num_expected_cols}, found {len(row_parts)}.")
                skipped_count += 1
    
    if skipped_count > 0:
        print(f"Total malformed rows skipped: {skipped_count}")
    return records


def find_clusters(df_records, df_matches, df_unmatches=None):
    """
    マッチング情報からクラスターを検出する。
    df_unmatchesが提供された場合、unmatchペアを含むクラスターが形成されないようにマージを制限する。
    """
    if df_unmatches is None:
        # --- 従来の高速なグラフベースのクラスタリング ---
        print("No unmatch file provided. Using standard graph-based clustering.")
        adj = defaultdict(list)
        record_ids = set(df_records['id'].astype(str))

        for _, row in df_matches.iterrows():
            id1 = str(row['id1'])
            id2 = str(row['id2'])
            if id1 in record_ids and id2 in record_ids:
                adj[id1].append(id2)
                adj[id2].append(id1)

        clusters = {}
        visited = set()

        for record_id in record_ids:
            if record_id not in visited:
                cluster_id = record_id
                component = []
                q = [record_id]
                visited.add(record_id)
                head = 0
                while head < len(q):
                    u = q[head]; head += 1
                    component.append(u)
                    if u in adj:
                        for v in adj[u]:
                            if v not in visited:
                                visited.add(v)
                                q.append(v)
                for node in component:
                    clusters[node] = cluster_id
        return clusters

    # --- unmatch制約を考慮した、より正確なクラスタリング ---
    print("Unmatch file provided. Using constraint-based clustering...")
    unmatch_pairs = set()
    for _, row in df_unmatches.iterrows():
        id1, id2 = str(row['id1']), str(row['id2'])
        unmatch_pairs.add(tuple(sorted((id1, id2))))

    record_ids = [str(id) for id in df_records['id']]
    parent = {rid: rid for rid in record_ids}
    cluster_members = {rid: {rid} for rid in record_ids}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    for _, row in df_matches.iterrows():
        id1, id2 = str(row['id1']), str(row['id2'])
        if id1 not in parent or id2 not in parent:
            continue

        root1, root2 = find(id1), find(id2)
        if root1 == root2:
            continue

        c1_nodes = cluster_members[root1]
        c2_nodes = cluster_members[root2]
        if len(c1_nodes) > len(c2_nodes):
            c1_nodes, c2_nodes = c2_nodes, c1_nodes

        can_merge = all(tuple(sorted((n1, n2))) not in unmatch_pairs for n1 in c1_nodes for n2 in c2_nodes)
        
        if can_merge:
            # Union: merge smaller cluster into larger one
            larger_root, smaller_root = (root1, root2) if len(cluster_members[root1]) >= len(cluster_members[root2]) else (root2, root1)
            parent[smaller_root] = larger_root
            cluster_members[larger_root].update(cluster_members[smaller_root])
            del cluster_members[smaller_root]

    return {rid: find(rid) for rid in record_ids}


def convert_tsv_to_yml_with_matches(
    dataset_file, 
    match_file, 
    unmatch_file=None,
    partitions=None,
    attributes_to_include=None, 
    random_seed=None,
    order_by='random',
    training_partitions=0,
    top_n_for_training=0
):
    """
    データセットをクラスタリングし、--partitionで指定された定義に基づいて
    複数の排反なサブセットグループに一括で分割する。
    """
    try:
        # Replace pd.read_csv with the robust custom reader
        dataset_records = read_dataset_robustly(dataset_file)
        if not dataset_records:
            print("Error: No records were read from the dataset file. Aborting.")
            return
        df = pd.DataFrame(dataset_records)
        
        df_matches = pd.read_csv(match_file, sep='\t', quotechar='"')
        
        df_unmatches = None
        if unmatch_file:
            df_unmatches = pd.read_csv(unmatch_file, sep='\t')
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    record_id_to_cluster_id = find_clusters(df, df_matches, df_unmatches)
    
    if attributes_to_include is None:
        attributes_to_include = [col for col in df.columns if col not in ['id', 'original_id']]
        
    inf_attr = {col: 'TEXT_EN' for col in attributes_to_include}

    all_records_by_cluster = defaultdict(list)
    for _, row in df.iterrows():
        record_id_str = str(row['id'])
        cluster_id = record_id_to_cluster_id.get(record_id_str, record_id_str)
        
        record_data = row.to_dict()
        filtered_data = {k: str(v) for k, v in record_data.items() if k in attributes_to_include}
        
        record = {
            'id': str(uuid.uuid4()),
            'cluster_id': cluster_id,
            'data': filtered_data
        }
        all_records_by_cluster[cluster_id].append(record)

    # --- 一括での排反なサブセット分割処理 ---
    if not partitions:
        print("Error: At least one --partition must be specified.")
        return

    if random_seed is not None:
        random.seed(random_seed)
    
    all_cluster_ids = list(all_records_by_cluster.keys())
    master_cluster_idx = 0
    
    # --- クラスタの並べ替え/準備 ---
    if order_by == 'size' and training_partitions > 0:
        print("Using hybrid partitioning strategy...")
        
        multi_record_clusters = [cid for cid in all_cluster_ids if len(all_records_by_cluster[cid]) > 1]
        single_record_clusters = [cid for cid in all_cluster_ids if len(all_records_by_cluster[cid]) == 1]
        
        multi_record_clusters.sort(key=lambda cid: len(all_records_by_cluster[cid]), reverse=True)
        random.shuffle(single_record_clusters)

        if top_n_for_training > 0:
            print(f"  - Guaranteeing top {top_n_for_training} largest clusters for training.")
            guaranteed_clusters = multi_record_clusters[:top_n_for_training]
            remaining_clusters = multi_record_clusters[top_n_for_training:] + single_record_clusters
            random.shuffle(remaining_clusters)
            all_cluster_ids = guaranteed_clusters + remaining_clusters
        else:
             # 従来のハイブリッド戦略
             print("  - Prioritizing all multi-record clusters for training.")
             all_cluster_ids = multi_record_clusters + single_record_clusters
    
    else: # 通常の戦略
        if order_by == 'size':
            print("Ordering all clusters by size (descending)...")
            all_cluster_ids.sort(key=lambda cid: len(all_records_by_cluster[cid]), reverse=True)
        else:
            print("Ordering all clusters randomly...")
            random.shuffle(all_cluster_ids)

    # 指定された全てのパーティション定義をループ処理
    for i, partition_def in enumerate(partitions):
        try:
            prefix, count_str, size_str = partition_def.split(':')
            count = int(count_str)
            size = int(size_str)
        except ValueError:
            print(f"Error: Invalid partition format '{partition_def}'. Expected 'prefix:count:size'.")
            continue
        
        print(f"\nProcessing partition definition {i+1}: prefix='{prefix}', count={count}, size={size}")
        
        is_training_phase = order_by == 'size' and training_partitions > 0 and i < training_partitions

        # このフェーズ表示はデバッグに役立つので残す
        if is_training_phase:
            print(f"  -> Phase: Training")
        elif order_by == 'size' and training_partitions > 0 and i == training_partitions:
             print(f"  -> Phase: Testing (using remaining shuffled clusters)")


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
            write_yml_file(output_filename, inf_attr, current_subset_clusters)
            print(f"  - Created subset {j+1}: {sum(len(v) for v in current_subset_clusters.values())} records in {len(current_subset_clusters)} clusters.")


def write_yml_file(output_file, inf_attr, records_by_cluster):
    """YAMLファイル書き出し処理を共通化"""
    # --- 出力ディレクトリの存在確認と作成 ---
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert, cluster, and partition the dataset into multiple mutually exclusive subsets.')
    parser.add_argument('dataset_file', type=str, 
                        help='Path to the input dataset TSV file')
    parser.add_argument('match_file', type=str, 
                        help='Path to the input match TSV file')
    parser.add_argument('--unmatch_file', type=str,
                        help='Path to the unmatch TSV file for refined clustering.')
    parser.add_argument('--partition', action='append', required=True,
                        help='Define a group of subsets. Format: "prefix:count:size". Can be specified multiple times.')
    parser.add_argument('--attributes', type=str, 
                        help='Comma-separated list of attributes to include.')
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
    
    convert_tsv_to_yml_with_matches(
        args.dataset_file, 
        args.match_file, 
        unmatch_file=args.unmatch_file,
        partitions=args.partition,
        attributes_to_include=attributes,
        random_seed=args.random_seed,
        order_by=args.order_by,
        training_partitions=args.training_partitions,
        top_n_for_training=args.top_n_for_training
    )
