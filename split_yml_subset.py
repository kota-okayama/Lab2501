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

def split_yml(input_yml, subset_size, output_prefix, output_dir, random_seed):
    """
    Loads a YML file, shuffles its clusters, and splits them into multiple
    subsets of a specified approximate size.
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

    # Get a list of all cluster IDs and shuffle them for random distribution
    cluster_ids = list(all_records_by_cluster.keys())
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(cluster_ids)

    # Partition the shuffled clusters into subsets
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split a record.yml file into multiple smaller, cluster-aware subsets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_yml', type=str,
                        help="Path to the input .yml file to be split.")
    parser.add_argument('subset_size', type=int,
                        help="The approximate target number of records for each subset.")
    parser.add_argument('--output_prefix', type=str,
                        help="Prefix for the output files. \n(default: derived from input file, e.g., 'input.yml' -> 'input_part')")
    parser.add_argument('--output_dir', type=str,
                        help="Directory to save the output subset files. \n(default: same directory as the input file)")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducible shuffling (default: 42).")

    args = parser.parse_args()

    # Determine the output directory
    output_directory = args.output_dir if args.output_dir else os.path.dirname(args.input_yml)

    # If output_prefix is not provided, create a default one from the input filename
    if not args.output_prefix:
        base, _ = os.path.splitext(os.path.basename(args.input_yml))
        output_prefix = f"{base}_part"
    else:
        # Ensure prefix does not contain directory structure if output_dir is used
        output_prefix = os.path.basename(args.output_prefix)


    split_yml(args.input_yml, args.subset_size, output_prefix, output_directory, args.random_seed)
