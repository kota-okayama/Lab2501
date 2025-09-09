#!/usr/bin/env python3
import yaml
import argparse
import json

def extract_size2_records(yml_file, output_format='text'):
    """
    YMLファイルからサイズ2のクラスターに所属するレコードを全て抽出する
    
    Args:
        yml_file: 分析するYMLファイルのパス
        output_format: 出力形式 ('text', 'json', 'yaml')
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
    
    # サイズ2のクラスターを抽出
    size2_clusters = []
    size2_records = []
    
    for cluster_id, cluster_data in records.items():
        if len(cluster_data) == 2:
            size2_clusters.append((cluster_id, cluster_data))
            size2_records.extend(cluster_data)
    
    print(f"File: {yml_file}")
    print(f"Found {len(size2_clusters)} clusters of size 2")
    print(f"Total records in size 2 clusters: {len(size2_records)}")
    print("=" * 80)
    
    if output_format == 'text':
        for i, (cluster_id, cluster_data) in enumerate(size2_clusters, 1):
            print(f"\n[Cluster {i}] ID: {cluster_id}")
            print("-" * 50)
            for j, record in enumerate(cluster_data, 1):
                print(f"  Record {j}:")
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"    {key}: {value}")
                print()
    
    elif output_format == 'json':
        output_data = {
            'file': yml_file,
            'cluster_count': len(size2_clusters),
            'record_count': len(size2_records),
            'clusters': []
        }
        
        for cluster_id, cluster_data in size2_clusters:
            output_data['clusters'].append({
                'cluster_id': cluster_id,
                'records': cluster_data
            })
        
        print(json.dumps(output_data, indent=2, ensure_ascii=False))
    
    elif output_format == 'yaml':
        output_data = {
            'file': yml_file,
            'cluster_count': len(size2_clusters),
            'record_count': len(size2_records),
            'clusters': {}
        }
        
        for cluster_id, cluster_data in size2_clusters:
            output_data['clusters'][cluster_id] = cluster_data
        
        print(yaml.dump(output_data, allow_unicode=True, indent=2))
    
    return size2_clusters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract all records from size-2 clusters in YML files")
    parser.add_argument('yml_file', help="Path to the YML file to analyze")
    parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text',
                        help="Output format (default: text)")
    parser.add_argument('--limit', type=int, help="Limit number of clusters to display")
    
    args = parser.parse_args()
    
    extract_size2_records(args.yml_file, args.format)
