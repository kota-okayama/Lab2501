#!/usr/bin/env python3
import yaml
import argparse
from collections import defaultdict

def analyze_clusters(yml_file, show_size=None, show_details=False):
    """
    YMLファイルのクラスター構成を分析する
    
    Args:
        yml_file: 分析するYMLファイルのパス
        show_size: 表示するクラスターサイズ（Noneの場合は全て）
        show_details: 詳細情報を表示するかどうか
    """
    try:
        with open(yml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    records = data.get('records', {})
    
    if not records:
        print("No records found in the file.")
        return
    
    # クラスターサイズ別に分析
    cluster_sizes = defaultdict(list)
    total_records = 0
    
    for cluster_id, cluster_data in records.items():
        size = len(cluster_data)
        cluster_sizes[size].append((cluster_id, cluster_data))
        total_records += size
    
    print(f"File: {yml_file}")
    print(f"Total clusters: {len(records)}")
    print(f"Total records: {total_records}")
    print("-" * 50)
    
    # サイズ別統計
    for size in sorted(cluster_sizes.keys()):
        clusters = cluster_sizes[size]
        count = len(clusters)
        records_count = size * count
        print(f"Size {size}: {count} clusters ({records_count} records)")
        
        # 指定されたサイズの詳細表示
        if show_size is None or size == show_size:
            if show_details:
                print(f"  Clusters of size {size}:")
                for i, (cluster_id, cluster_data) in enumerate(clusters[:10]):  # 最初の10個まで表示
                    print(f"    [{i+1}] Cluster ID: {cluster_id}")
                    for j, record in enumerate(cluster_data):
                        record_id = record.get('id', 'N/A')
                        title = record.get('title', 'N/A')[:50] + "..." if len(record.get('title', '')) > 50 else record.get('title', 'N/A')
                        print(f"      Record {j+1}: ID={record_id}, Title={title}")
                    print()
                
                if len(clusters) > 10:
                    print(f"    ... and {len(clusters) - 10} more clusters of size {size}")
                print()
    
    print("-" * 50)
    
    # 検証計算
    calculated_total = sum(size * len(clusters) for size, clusters in cluster_sizes.items())
    print(f"Verification: {calculated_total} records (should match total above)")
    
    return cluster_sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze cluster composition in YML files")
    parser.add_argument('yml_file', help="Path to the YML file to analyze")
    parser.add_argument('--show_size', type=int, help="Show details for clusters of specific size")
    parser.add_argument('--details', action='store_true', help="Show detailed cluster contents")
    
    args = parser.parse_args()
    
    analyze_clusters(args.yml_file, args.show_size, args.details)
