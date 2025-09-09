import yaml
import argparse
import os
import random
from collections import defaultdict
from datetime import datetime

def write_subset_yml(output_path, original_inf_attr, subset_records):
    """
    Calculates a new summary and writes a subset of records to a new YML file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Recalculate summary for the subset
    num_of_records = sum(len(cluster) for cluster in subset_records.values())
    num_of_pairs = defaultdict(int)
    for cluster in subset_records.values():
        num_of_pairs[len(cluster)] += 1
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    summary = {
        'creation_date': now,
        'update_date': now,
        'num_of_records': num_of_records,
        'num_of_pairs': dict(sorted(num_of_pairs.items())),
        'config_match': None,
        'config_mismatch': None,
    }

    output_data = {
        'summary': summary,
        'inf_attr': original_inf_attr,
        'records': subset_records
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, sort_keys=False, allow_unicode=True, indent=2)
        print(f"Successfully created subset: {output_path} ({num_of_records} records)")
    except Exception as e:
        print(f"Error writing YML file: {e}")

def split_yml(input_yml, subset_size, output_prefix, output_dir, random_seed, sort_by_size=False, 
              min_cluster_size=1, max_cluster_size=None, exclude_cluster_sizes=None, exclude_cluster_count=None):
    """
    Loads a YML file, shuffles or sorts clusters, and splits them 
    into multiple subsets of a specified approximate size.
    
    Args:
        sort_by_size: If True, sort clusters by size (largest first).
                     If False, shuffle clusters randomly (default).
        min_cluster_size: Minimum cluster size to include (default: 1).
        max_cluster_size: Maximum cluster size to include (default: None).
        exclude_cluster_sizes: List of cluster sizes to exclude (default: None).
        exclude_cluster_count: Number of largest clusters to exclude (default: None).
    """
    try:
        with open(input_yml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_yml}")
        return
    except Exception as e:
        print(f"Error reading or parsing YML file: {e}")
        return

    inf_attr = data.get('inf_attr', {})
    all_records_by_cluster = data.get('records', {})

    if not all_records_by_cluster:
        print("Warning: No records found in the input file.")
        return

    # Filter clusters based on various criteria
    original_cluster_count = len(all_records_by_cluster)
    
    # Apply size-based filtering
    filtered_records = {}
    for cid, cluster in all_records_by_cluster.items():
        cluster_size = len(cluster)
        
        # Check minimum size
        if cluster_size < min_cluster_size:
            continue
            
        # Check maximum size
        if max_cluster_size is not None and cluster_size > max_cluster_size:
            continue
            
        # Check excluded sizes
        if exclude_cluster_sizes and cluster_size in exclude_cluster_sizes:
            continue
            
        filtered_records[cid] = cluster
    
    # Apply cluster count exclusion (remove largest clusters)
    if exclude_cluster_count and exclude_cluster_count > 0:
        # Sort by size to identify largest clusters
        sorted_clusters = sorted(filtered_records.items(), key=lambda x: len(x[1]), reverse=True)
        excluded_count = min(exclude_cluster_count, len(sorted_clusters))
        # Keep all except the largest N clusters
        filtered_records = dict(sorted_clusters[excluded_count:])
        print(f"Excluded {excluded_count} largest clusters")
    
    cluster_ids = list(filtered_records.keys())
    
    print(f"Original clusters: {original_cluster_count}, After filtering: {len(cluster_ids)}")
    
    if not cluster_ids:
        print("Warning: No clusters found after filtering")
        return
    
    # Update the records dictionary to use filtered records
    all_records_by_cluster = filtered_records
    
    if sort_by_size:
        # Sort by cluster size (largest first)
        cluster_ids.sort(key=lambda cid: len(all_records_by_cluster[cid]), reverse=True)
        print(f"Sorting clusters by size (largest first). Total clusters: {len(cluster_ids)}")
    else:
        # Shuffle randomly for random distribution
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(cluster_ids)
        print(f"Shuffling clusters randomly (seed: {random_seed}). Total clusters: {len(cluster_ids)}")

    # Partition the clusters into subsets
    cluster_idx = 0
    subset_count = 1
    while cluster_idx < len(cluster_ids):
        current_subset_records = {}
        current_record_count = 0
        
        # Keep adding clusters until the subset size is met or exceeded
        while current_record_count < subset_size and cluster_idx < len(cluster_ids):
            cid = cluster_ids[cluster_idx]
            cluster_data = all_records_by_cluster[cid]
            current_subset_records[cid] = cluster_data
            current_record_count += len(cluster_data)
            cluster_idx += 1
        
        # Write the newly created subset to a file
        if current_subset_records:
            # Construct the full output path
            filename = f"{os.path.basename(output_prefix)}_{subset_count}.yml"
            output_path = os.path.join(output_dir, filename)
            
            write_subset_yml(output_path, inf_attr, current_subset_records)
            subset_count += 1
    
    print("\nSplitting complete.")


def merge_yml_files(file_paths, output_path, random_seed=42):
    """
    Merge multiple YML files into a single file.
    
    Args:
        file_paths: List of YML file paths to merge
        output_path: Output file path for merged YML
        random_seed: Random seed for reproducible cluster ID generation
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    merged_records = {}
    all_inf_attr = {}
    total_original_records = 0
    
    print(f"Merging {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        records = data.get('records', {})
        inf_attr = data.get('inf_attr', {})
        
        # Update inf_attr (merge attributes from all files)
        all_inf_attr.update(inf_attr)
        
        # Add records with prefixed cluster IDs to avoid conflicts
        for cluster_id, cluster_data in records.items():
            new_cluster_id = f"file{i}_{cluster_id}"
            merged_records[new_cluster_id] = cluster_data
            total_original_records += len(cluster_data)
            
        print(f"  - {file_path}: {len(records)} clusters, {sum(len(cluster) for cluster in records.values())} records")
    
    if not merged_records:
        print("Error: No records found in any input files")
        return
        
    # Write merged file
    write_subset_yml(output_path, all_inf_attr, merged_records)
    print(f"\nMerge complete: {len(merged_records)} clusters, {total_original_records} total records")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split a record.yml file into multiple smaller, cluster-aware subsets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_yml', type=str, nargs='?',
                        help="Path to the input .yml file to be split.")
    parser.add_argument('subset_size', type=int, nargs='?',
                        help="The approximate target number of records for each subset.")
    parser.add_argument('--output_prefix', type=str,
                        help="Prefix for the output files. \n(default: derived from input file, e.g., 'input.yml' -> 'input_part')")
    parser.add_argument('--output_dir', type=str,
                        help="Directory to save the output subset files. \n(default: same directory as the input file)")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducible shuffling (default: 42).")
    parser.add_argument('--sort_by_size', action='store_true',
                        help="Sort clusters by size (largest first) instead of random shuffling.")
    parser.add_argument('--min_cluster_size', type=int, default=1,
                        help="Minimum cluster size to include (default: 1, include all clusters).")
    parser.add_argument('--max_cluster_size', type=int, default=None,
                        help="Maximum cluster size to include (default: None, no upper limit).")
    parser.add_argument('--exclude_cluster_size', type=int, action='append',
                        help="Exclude clusters of specific size (can be used multiple times).")
    parser.add_argument('--exclude_cluster_count', type=int, default=None,
                        help="Exclude specified number of largest clusters.")
    parser.add_argument('--merge_files', nargs='+', 
                        help="Merge multiple YML files instead of splitting (provide multiple file paths).")

    args = parser.parse_args()

    # Check if merge mode is requested
    if args.merge_files:
        if not args.output_prefix:
            output_path = "merged_output.yml"
        else:
            output_path = f"{args.output_prefix}.yml"
            
        if args.output_dir:
            output_path = os.path.join(args.output_dir, output_path)
            
        merge_yml_files(args.merge_files, output_path, args.random_seed)
        exit(0)
    
    # Check required arguments for split mode
    if not args.input_yml or args.subset_size is None:
        parser.error("input_yml and subset_size are required for split mode (unless using --merge_files)")

    # Determine the output directory
    output_directory = args.output_dir if args.output_dir else os.path.dirname(args.input_yml)

    # If output_prefix is not provided, create a default one from the input filename
    if not args.output_prefix:
        base, _ = os.path.splitext(os.path.basename(args.input_yml))
        output_prefix = f"{base}_part"
    else:
        # Ensure prefix does not contain directory structure if output_dir is used
        output_prefix = os.path.basename(args.output_prefix)

    # Prepare exclude_cluster_sizes list
    exclude_cluster_sizes = args.exclude_cluster_size if args.exclude_cluster_size else None

    split_yml(args.input_yml, args.subset_size, output_prefix, output_directory, args.random_seed, 
              args.sort_by_size, args.min_cluster_size, args.max_cluster_size, 
              exclude_cluster_sizes, args.exclude_cluster_count)
