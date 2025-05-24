#!/usr/bin/env python3
import json
import os
import sys
import argparse
import networkx as nx
from collections import defaultdict

# Add project root to sys.path to allow importing from data_processing
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from data_processing.load_yaml_data import load_bibliographic_data  # type: ignore


def format_record_details(record_data):
    """Helper function to format bibliographic details for JSON output."""
    if not record_data or not isinstance(record_data, dict):
        return "Details not available"

    title = record_data.get("bib1_title", "N/A")
    authors_list = record_data.get("bib1_author_list", [])
    if not authors_list and "bib1_author" in record_data:
        authors_list = [record_data["bib1_author"]]

    author_str = "/".join(authors_list) if authors_list else "N/A"
    publisher = record_data.get("bib1_publisher", "N/A")
    pub_date = record_data.get("bib1_pub_date", "N/A")

    return f"タイトル: {title}\\n著者: {author_str}\\n出版社: {publisher}\\n出版日: {pub_date}"


def main():
    parser = argparse.ArgumentParser(
        description="Format KNN graph JSON to include details and group by connected components."
    )
    parser.add_argument(
        "--input_knn_json", type=str, required=True, help="Path to the input KNN graph JSON (adjacency list format)."
    )
    parser.add_argument(
        "--ground_truth_yaml", type=str, required=True, help="Path to the YAML file containing bibliographic details."
    )
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the formatted JSON output.")

    args = parser.parse_args()

    print(f"Loading bibliographic data from {args.ground_truth_yaml}...")
    all_records_list = load_bibliographic_data(args.ground_truth_yaml)
    if not all_records_list:
        print("Failed to load bibliographic data. Cannot proceed.")
        return

    record_details_map = {str(rec["record_id"]): rec for rec in all_records_list}
    print(f"Loaded details for {len(record_details_map)} records.")

    print(f"Loading KNN graph from {args.input_knn_json}...")
    try:
        with open(args.input_knn_json, "r", encoding="utf-8") as f:
            knn_adj_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input KNN JSON file not found at {args.input_knn_json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_knn_json}")
        return

    G = nx.Graph()
    for source_node, neighbors in knn_adj_list.items():
        G.add_node(str(source_node))  # Ensure source node exists
        for neighbor_node in neighbors:
            G.add_edge(str(source_node), str(neighbor_node))

    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    output_data = {}
    print("Processing connected components as clusters...")
    for i, component in enumerate(nx.connected_components(G)):
        cluster_id = f"knn_cluster_{i+1}"
        output_data[cluster_id] = []
        for record_id_str in component:
            record_info = record_details_map.get(record_id_str)
            if record_info and "data" in record_info:
                formatted_details = format_record_details(record_info["data"])
            else:
                formatted_details = f"Details not found for {record_id_str}"
            output_data[cluster_id].append({"record_id": record_id_str, "details": formatted_details})

    print(f"Writing formatted KNN graph to {args.output_json}...")
    try:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("Successfully wrote formatted KNN graph.")
    except IOError as e:
        print(f"Error writing output JSON to {args.output_json}: {e}")


if __name__ == "__main__":
    main()
