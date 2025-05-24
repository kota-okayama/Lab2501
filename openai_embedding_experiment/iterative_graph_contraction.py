#!/usr/bin/env python3
import faiss
import numpy as np
import pickle
import os
import json
import time
import sys
import argparse
import networkx as nx
from collections import defaultdict
from itertools import combinations
import yaml

# Project root for importing other modules like load_yaml_data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from data_processing.load_yaml_data import load_bibliographic_data  # Assuming this is the correct path and function


def format_record_details(record_data):
    """Helper function to format bibliographic details for JSON output."""
    if not record_data or not isinstance(record_data, dict):
        return "Details not available"

    title = record_data.get("bib1_title", "N/A")
    # Try to get author_list, fallback to author string if not present
    authors_list = record_data.get("bib1_author_list", [])
    if not authors_list and "bib1_author" in record_data:  # Fallback to bib1_author
        authors_list = [record_data["bib1_author"]]

    author_str = "/".join(authors_list) if authors_list else "N/A"

    publisher = record_data.get("bib1_publisher", "N/A")
    pub_date = record_data.get("bib1_pub_date", "N/A")  # Assuming 'bib1_pub_date' is the key

    return f"タイトル: {title}\\n著者: {author_str}\\n出版社: {publisher}\\n出版日: {pub_date}"


def output_clusters_to_json(graph, record_details_map, output_filename, is_initial_graph=False):
    """Outputs graph clusters to a JSON file with record details."""
    output_data = {}
    processed_node_count = 0  # For logging problematic data

    if is_initial_graph:  # For initial KNN, connected components are clusters
        for i, component in enumerate(nx.connected_components(graph)):
            cluster_id = f"initial_knn_cluster_{i+1}"
            output_data[cluster_id] = []
            for node_id in component:  # node_id is an original record_id here
                # Ensure node_id is string for map lookup
                record_info = record_details_map.get(str(node_id))
                if record_info and "data" in record_info:
                    formatted_details = format_record_details(record_info["data"])
                else:
                    formatted_details = f"Details not found for {node_id}"
                output_data[cluster_id].append({"record_id": str(node_id), "details": formatted_details})
            processed_node_count += 1
    else:  # For contracted graph, each node is a cluster
        for i, node_id_in_graph in enumerate(graph.nodes()):
            cluster_id = f"contracted_cluster_{i+1}_node_{node_id_in_graph}"
            output_data[cluster_id] = []
            original_ids = graph.nodes[node_id_in_graph].get("initial_ids", frozenset())
            for original_record_id in original_ids:
                record_info = record_details_map.get(str(original_record_id))
                if record_info and "data" in record_info:
                    formatted_details = format_record_details(record_info["data"])
                else:
                    formatted_details = f"Details not found for {original_record_id}"

                output_data[cluster_id].append({"record_id": str(original_record_id), "details": formatted_details})
            processed_node_count += 1

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully wrote cluster details to {output_filename}")
    except IOError as e:
        print(f"Error writing cluster details to {output_filename}: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON for {output_filename}: {e}")
        # Log a snippet of the problematic data for debugging
        first_few_items = {k: v for idx, (k, v) in enumerate(output_data.items()) if idx < 2}
        print(
            "Problematic data snippet (first ~2 clusters):",
            json.dumps(first_few_items, ensure_ascii=False, indent=2, default=str)[:500],
        )


def load_ground_truth_pairs(record_yaml_path_arg):
    """Loads ground truth similar pairs from the YAML file."""
    if not os.path.exists(record_yaml_path_arg):
        print(f"Error: Ground truth YAML file not found: {record_yaml_path_arg}")
        return None
    gt_pairs = set()
    clusters_to_records = defaultdict(list)
    try:
        with open(record_yaml_path_arg, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        # Adjusted YAML parsing based on evaluate_knn_f1_openai.py
        possible_records_dict = {}
        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

        if isinstance(possible_records_dict, dict):
            for _cluster_key, items_in_cluster in possible_records_dict.items():
                # Skip metadata keys if necessary (like version, type, etc.)
                if _cluster_key in ["version", "type", "id", "summary", "inf_attr"]:
                    continue
                if isinstance(items_in_cluster, list):
                    for record_item in items_in_cluster:
                        if isinstance(record_item, dict) and "id" in record_item and "cluster_id" in record_item:
                            record_id = str(record_item["id"])
                            cluster_id = str(record_item["cluster_id"])
                            # We only care about records belonging to a non-orphan cluster for gt_pairs
                            if not cluster_id.startswith("gt_orphan_"):
                                clusters_to_records[cluster_id].append(record_id)
        else:
            print(f"Error: Structure of {record_yaml_path_arg} is not as expected for gt_pairs.")
            return None

        if not clusters_to_records:
            print(f"Error: Could not load cluster information for gt_pairs from {record_yaml_path_arg}.")
            return None

        for cluster_id, ids_in_cluster in clusters_to_records.items():
            if len(ids_in_cluster) >= 2:
                for pair_tuple in combinations(sorted(ids_in_cluster), 2):  # Sort for canonical representation
                    gt_pairs.add(pair_tuple)
        print(f"Loaded {len(gt_pairs)} ground truth similar pairs from {record_yaml_path_arg}")
        return gt_pairs
    except Exception as e:
        print(f"Error loading ground truth pairs: {e}")
        return None


def build_initial_knn_graph(embeddings_path, ids_path, k_neighbors, record_details_map, ground_truth_yaml_path):
    """Builds the initial K-NN graph using Faiss and NetworkX."""
    print(f"\nStep 1: Building initial K-NN graph (K={k_neighbors})...")
    if not os.path.exists(embeddings_path) or not os.path.exists(ids_path):
        print(f"Error: Embeddings file ({embeddings_path}) or IDs file ({ids_path}) not found.")
        return None

    try:
        record_embeddings = np.load(embeddings_path)
        with open(ids_path, "rb") as f:
            record_ids_ordered = pickle.load(f)
    except Exception as e:
        print(f"Error loading embedding data: {e}")
        return None

    if record_embeddings.ndim == 1:
        if record_embeddings.shape[0] > 0:
            record_embeddings = record_embeddings.reshape(1, -1)
        else:
            print("Error: Embeddings array is empty or malformed.")
            return None
    elif record_embeddings.ndim != 2:
        print(f"Error: Embeddings array has unexpected dimension: {record_embeddings.ndim}. Expected 2D.")
        return None

    num_records, dimension = record_embeddings.shape
    if len(record_ids_ordered) != num_records:
        print("Error: Mismatch between number of embeddings and IDs.")
        return None

    print(f"Loaded {num_records} records, dimension {dimension}.")

    actual_k = k_neighbors
    if actual_k >= num_records and num_records > 0:
        actual_k = num_records - 1

    if actual_k <= 0 and num_records > 0:
        print("Warning: K is too small for the number of records. Cannot build meaningful graph.")
        # Return an empty graph or a graph with nodes but no edges
        G = nx.Graph()
        for r_id in record_ids_ordered:
            G.add_node(str(r_id), initial_ids=frozenset([str(r_id)]))  # Store original ID(s)
        # Output this empty/node-only graph if needed
        output_filename = os.path.join(
            # Assuming args.output_dir is accessible or passed down
            # For now, using a fixed relative path for simplicity if args not available here.
            ".",
            f"initial_knn_clusters_k{k_neighbors}_empty.json",
        )
        # To make this robust, output_dir should be passed to this function or made globally accessible.
        # Fallback: if record_details_map is None, we can't add details.
        if record_details_map:
            output_clusters_to_json(G, record_details_map, output_filename, is_initial_graph=True)
        else:
            print(f"record_details_map not available to build_initial_knn_graph for empty graph output.")
        return G

    index = faiss.IndexFlatL2(dimension)
    index.add(record_embeddings)

    # Search for K+1 neighbors to exclude self if present
    num_to_search = min(actual_k + 1, num_records)
    distances, indices = index.search(record_embeddings, num_to_search)

    G = nx.Graph()
    # Initialize nodes with their original ID(s) as a frozenset attribute
    # This attribute will track the set of original record IDs a node represents after contraction
    for r_id in record_ids_ordered:
        G.add_node(str(r_id), initial_ids=frozenset([str(r_id)]))

    for i in range(num_records):
        source_node_id = str(record_ids_ordered[i])
        k_count = 0
        for j in range(indices.shape[1]):
            neighbor_original_idx = indices[i][j]
            if neighbor_original_idx == i:  # Skip self
                continue
            if neighbor_original_idx == -1:  # Should not happen with IndexFlatL2 unless K > N
                continue

            target_node_id = str(record_ids_ordered[neighbor_original_idx])
            # Add edge. NetworkX handles undirected graphs, so (A,B) is same as (B,A)
            G.add_edge(source_node_id, target_node_id, weight=distances[i][j])
            k_count += 1
            if k_count >= actual_k:
                break

    print(f"Initial K-NN graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Output initial K-NN graph clusters
    # Determine output directory more robustly
    # For now, assumes an "output_clusters" dir exists or is created by main()
    # This path might need to be passed if main's args.output_dir isn't global
    output_dir_path = "."  # Default to current if not specified otherwise
    # A better way would be to pass output_dir from main to this function.
    # Example: output_filename = os.path.join(passed_output_dir, f"initial_knn_clusters_k{k_neighbors}.json")

    # Attempt to get output_dir from a common place if not passed (e.g. global or from args if accessible)
    # This is a placeholder for robust path handling.
    try:
        # This is a hack, ideally output_dir comes from args passed down.
        # Trying to access args from a global context or via a shared object if set by main.
        # This part is problematic if build_initial_knn_graph is called outside main's direct scope
        # or if args object isn't passed.
        # For this edit, we'll assume it's called from main where args is defined.
        # A proper fix involves refactoring to pass 'output_dir' argument.
        script_args = sys.argv  # This is not ideal for library code.
        output_dir_arg_val = "."  # Default
        if "--output_dir" in script_args:
            output_dir_arg_val = script_args[script_args.index("--output_dir") + 1]
        else:  # Fallback to a default if not in args (e.g. if called programmatically without CLI args)
            if hasattr(main, "args_output_dir_for_build_graph"):  # A hypothetical way to pass it
                output_dir_arg_val = main.args_output_dir_for_build_graph

        # Ensure output_dir exists if we derived it here
        if not os.path.exists(output_dir_arg_val) and output_dir_arg_val != ".":
            os.makedirs(output_dir_arg_val)  # Create if doesn't exist and not current dir

        output_filename = os.path.join(output_dir_arg_val, f"initial_knn_clusters_k{k_neighbors}.json")
    except Exception:  # Fallback if arg parsing here is too complex/fails
        output_filename = f"initial_knn_clusters_k{k_neighbors}.json"  # In current dir

    if record_details_map:
        output_clusters_to_json(G, record_details_map, output_filename, is_initial_graph=True)
    else:
        print(f"record_details_map not available to build_initial_knn_graph for outputting initial clusters.")

    return G


# --- Placeholder for core logic ---
def calculate_edge_scores(graph):
    """Calculates Score(vi,vj) = |N(vi) ∩ N(vj)| for all edges in the graph."""
    edge_scores = {}
    for u, v in graph.edges():
        # Get neighbors of u and v. Convert to sets for intersection.
        # graph.neighbors(n) returns an iterator.
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))

        # The score is the number of common neighbors.
        # We should exclude u and v themselves if they are in neighbor sets by some chance,
        # though for simple graphs N(u) doesn't include u.
        # The intersection N(u) ∩ N(v) directly gives common neighbors.
        score = len(neighbors_u.intersection(neighbors_v))
        edge_scores[(u, v)] = score
    return edge_scores


def get_highest_score_edge(scores_dict):
    """Finds the edge with the highest score from a dictionary of edge scores."""
    if not scores_dict:
        return None

    # Find the edge (tuple of two nodes) with the maximum score.
    # max() on a dictionary's items will compare by value (the score).
    # The key for max will be a function that returns the value from the (key, value) pair.
    highest_score_edge = max(scores_dict.items(), key=lambda item: item[1])[0]
    # We could also store the score itself if needed: max_score = scores_dict[highest_score_edge]
    return highest_score_edge


def get_simulated_judgment(graph, current_node1, current_node2, ground_truth_pairs):
    """Simulates judgment for a pair of (potentially contracted) nodes based on ground truth.

    Checks if any original ID pair formed from the constituent original IDs of
    current_node1 and current_node2 exists in the ground_truth_pairs set.
    """
    if not graph.has_node(current_node1) or not graph.has_node(current_node2):
        print(f"Warning: One or both nodes {current_node1}, {current_node2} not in graph for judgment.")
        return False  # Or raise error

    original_ids_node1 = graph.nodes[current_node1].get("initial_ids")
    original_ids_node2 = graph.nodes[current_node2].get("initial_ids")

    if not original_ids_node1 or not original_ids_node2:
        print(f"Warning: initial_ids attribute missing for {current_node1} or {current_node2}.")
        return False  # Or raise error

    # Iterate through all combinations of original IDs from the two (potentially merged) nodes
    for id1_orig in original_ids_node1:
        for id2_orig in original_ids_node2:
            if id1_orig == id2_orig:  # Cannot form a pair with self if a node somehow contains same original id twice
                continue

            # Ensure canonical order for lookup in ground_truth_pairs (sorted tuple)
            # ground_truth_pairs should store pairs like (id_smaller, id_larger)
            eval_pair = tuple(sorted((str(id1_orig), str(id2_orig))))

            if eval_pair in ground_truth_pairs:
                return True  # Found a true match between original components

    return False  # No combination of original IDs formed a ground truth pair


def contract_nodes_in_graph(graph, node_to_keep, node_to_remove, discovered_true_pairs):
    """
    Contracts node_to_remove into node_to_keep in the graph.
    Updates the 'initial_ids' attribute of node_to_keep.
    Adds all newly formed original_id pairs to discovered_true_pairs.
    The graph is modified in place.
    """
    if not graph.has_node(node_to_keep) or not graph.has_node(node_to_remove):
        # This case should ideally be prevented by checks before calling
        print(f"Warning: Cannot contract. Node {node_to_keep} or {node_to_remove} not in graph.")
        return

    if node_to_keep == node_to_remove:
        # This case should also be prevented by checks (e.g. not picking self-loops for contraction)
        print(f"Warning: Attempting to contract a node {node_to_keep} with itself. Skipping.")
        return

    # Get initial_ids before contraction
    # Ensure they are frozensets as initialized
    ids_to_keep = graph.nodes[node_to_keep].get("initial_ids", frozenset())
    ids_to_remove = graph.nodes[node_to_remove].get("initial_ids", frozenset())

    # Add pairs formed between the two sets to discovered_true_pairs
    # These are the pairs that become "matched" by this contraction event.
    for id_k in ids_to_keep:
        for id_r in ids_to_remove:
            # initial_ids should contain original record IDs as strings
            # No need to check id_k != id_r if ids_to_keep and ids_to_remove are from distinct original entities
            # before this specific contraction. If they were already merged, this logic is more complex.
            # Assuming this function is called for two *currently distinct* graph nodes that are to be merged.
            discovered_true_pairs.add(tuple(sorted((str(id_k), str(id_r)))))

    # Manual contraction:
    new_initial_ids = ids_to_keep.union(ids_to_remove)

    # Re-target edges from node_to_remove to node_to_keep
    # Iterate over a copy of neighbors list as graph.adj[node_to_remove] will change during iteration
    # if graph.remove_node is called before finishing iteration over edges.
    # More robust: iterate over edges connected to node_to_remove.
    edges_to_rewire = list(graph.edges(node_to_remove, data=True))

    # Store attributes of the edge being contracted if needed for future logic
    # Example: contracted_edge_data = graph.get_edge_data(node_to_keep, node_to_remove)
    # This assumes the edge (node_to_keep, node_to_remove) is the one being processed.
    # If node_to_keep and node_to_remove were not directly connected, this would be None.

    # Check if node_to_remove exists before trying to remove it
    if graph.has_node(node_to_remove):
        graph.remove_node(node_to_remove)  # Remove the node first to handle its edges cleanly.
        # This also removes all incident edges of node_to_remove.
    else:
        # This could happen if node_to_remove was already merged due to processing another edge involving it
        # in a more complex scenario (e.g. if node_to_keep was also node_to_remove's neighbor via another path).
        print(f"Warning: Node {node_to_remove} was already removed from graph before explicit contraction step.")

    # Now, add back the rewired edges to node_to_keep
    for _, neighbor, edge_data in edges_to_rewire:
        if neighbor == node_to_remove:  # This was an edge to itself (self-loop on node_to_remove)
            continue
        if neighbor == node_to_keep:  # This was an edge between node_to_remove and node_to_keep
            # It's already removed by remove_node(node_to_remove).
            # We don't want to add a self-loop to node_to_keep unless intended.
            # The paper implies edge (vi,vj) is removed after processing. Here, it's part of contraction.
            continue  # Effectively, the edge (node_to_keep, node_to_remove) is consumed by the contraction.

        # If neighbor still exists (it wasn't node_to_remove itself)
        # Add edge from node_to_keep to this neighbor, preserving attributes
        # Check if an edge already exists to avoid parallel edges if G is not MultiGraph
        # (though add_edge updates attributes if it exists)
        if graph.has_node(neighbor):  # Ensure neighbor wasn't also removed (e.g. if it was node_to_remove)
            graph.add_edge(node_to_keep, neighbor, **edge_data)
        # else:
        # print(f"  Skipping rewire to {neighbor} as it's no longer in graph (possibly {node_to_remove}).")

    # Update the initial_ids of the kept node
    if graph.has_node(node_to_keep):  # Should always be true
        graph.nodes[node_to_keep]["initial_ids"] = new_initial_ids
    # else:
    # This would be a critical error, node_to_keep should always exist.
    # print(f"CRITICAL ERROR: node_to_keep {node_to_keep} does not exist after removing {node_to_remove}")
    # print(f"  Contracted {node_to_remove} into {node_to_keep}. New initial_ids for {node_to_keep}: {graph.nodes[node_to_keep]['initial_ids']}")


def main():
    parser = argparse.ArgumentParser(description="Iterative graph contraction for entity resolution.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to .npy embeddings.")
    parser.add_argument("--ids_path", type=str, required=True, help="Path to .pkl record IDs.")
    parser.add_argument("--ground_truth_yaml", type=str, required=True, help="Path to ground truth YAML.")
    parser.add_argument("--k_neighbors", type=int, default=5, help="K for initial K-NN graph.")
    parser.add_argument(
        "--output_dir", type=str, default="output_clusters", help="Directory to save cluster JSON files."
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Maximum number of iterations for the contraction loop."
    )

    args = parser.parse_args()
    # For build_initial_knn_graph to access output_dir if not passed directly (example of less ideal sharing)
    # setattr(main, 'args_output_dir_for_build_graph', args.output_dir) # Not standard, better to pass args

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("Starting Iterative Graph Contraction Process...")
    print(f"Parameters: K={args.k_neighbors}")
    print(f"  Embeddings: {args.embeddings_path}")
    print(f"  IDs: {args.ids_path}")
    print(f"  Ground Truth: {args.ground_truth_yaml}")
    print(f"  Output Directory: {args.output_dir}")
    if args.max_iterations:
        print(f"  Max Iterations: {args.max_iterations}")

    # Load bibliographic details
    print(f"Loading bibliographic data from {args.ground_truth_yaml} for details...")
    all_records_list = load_bibliographic_data(args.ground_truth_yaml)
    if not all_records_list:
        print("Failed to load bibliographic data. Details will be missing in output. Exiting.")
        return  # Exit if critical data for output is missing.

    record_details_map = {str(rec["record_id"]): rec for rec in all_records_list}
    print(f"Loaded details for {len(record_details_map)} records.")

    # Load ground truth pairs
    ground_truth_pairs = load_ground_truth_pairs(args.ground_truth_yaml)
    if ground_truth_pairs is None:
        print("Failed to load ground truth pairs. Exiting.")
        return

    # Build initial K-NN graph (and output its clusters)
    # Pass args.output_dir to build_initial_knn_graph for robust path handling
    current_graph = build_initial_knn_graph(
        args.embeddings_path,
        args.ids_path,
        args.k_neighbors,
        record_details_map,
        args.ground_truth_yaml,  # This was for record_details_map, now passed directly
        # output_dir=args.output_dir # Ideal way if function signature is changed
    )
    # The build_initial_knn_graph has been modified to attempt to use sys.argv or a hypothetical shared attribute for output_dir.
    # A cleaner way is to modify its signature: build_initial_knn_graph(..., output_dir_path)
    # and pass args.output_dir directly. For now, relying on the internal fallback.

    if current_graph is None or current_graph.number_of_nodes() == 0:
        print("Failed to build initial K-NN graph or graph is empty. Exiting.")
        return

    # --- Main Iterative Loop ---
    print("\nStep 2: Starting iterative contraction loop...")
    discovered_true_pairs = set()  # To store unique original ID pairs that have been effectively merged
    iteration_count = 0

    while current_graph.number_of_edges() > 0:
        if args.max_iterations and iteration_count >= args.max_iterations:
            print(f"Reached max iterations ({args.max_iterations}). Stopping.")
            break
        iteration_count += 1
        # Suppressing per-iteration print for cleaner logs, unless debugging.
        # print(
        #     f"\nIteration {iteration_count}, Edges: {current_graph.number_of_edges()}, Nodes: {current_graph.number_of_nodes()}"
        # )

        # 1. Calculate scores for all current edges
        edge_scores = calculate_edge_scores(current_graph)
        if not edge_scores:
            print("No more edges with calculable scores. Stopping.")
            break

        # 2. Select edge with the highest score
        # Ensure a consistent way to pick if multiple edges have the same max score
        # For now, max() behavior is fine. The edge is (u,v)
        best_edge_to_process = get_highest_score_edge(edge_scores)
        if best_edge_to_process is None:
            print("No best edge found. Stopping.")  # Should be caught by `if not edge_scores` generally
            break

        # Ensure consistent node order for contraction (e.g. smaller ID first, or based on some attribute)
        # This can help in reproducibility if scores are tied. For now, using the order from get_highest_score_edge.
        # node1, node2 = sorted(list(best_edge_to_process)) # Ensure list for sort, then unpack
        node1, node2 = best_edge_to_process[0], best_edge_to_process[1]

        # Safety check: ensure nodes are still in graph. They should be if edge_scores are fresh.
        if not current_graph.has_node(node1) or not current_graph.has_node(node2):
            print(
                f"  Warning: Edge ({node1}, {node2}) selected, but one or both nodes no longer in graph. This might indicate stale scores or concurrent modification issues. Skipping this edge."
            )
            # If this happens often, consider re-calculating scores or managing graph state more carefully.
            current_graph.remove_edge(node1, node2)  # Remove the problematic edge
            continue  # Or break, if this state is unexpected and critical

        # print(f"  Processing edge: ({node1}, {node2}) with score {edge_scores[best_edge_to_process]}")

        simulated_match = get_simulated_judgment(current_graph, node1, node2, ground_truth_pairs)

        if simulated_match:
            # print(f"  Match found (simulated) between current nodes {node1} and {node2}. Contracting...")
            # contract_nodes_in_graph modifies current_graph and discovered_true_pairs in place
            contract_nodes_in_graph(
                current_graph, node1, node2, discovered_true_pairs
            )  # node1 is kept, node2 is removed
        else:
            # print(f"  No match (simulated) between {node1} and {node2}. Removing edge...")
            if current_graph.has_edge(node1, node2):
                current_graph.remove_edge(node1, node2)
            # else:
            # print(f"  Warning: Edge ({node1}, {node2}) was to be removed but not found. Already processed?")

    print(f"\nLoop finished after {iteration_count} iterations.")
    print(f"Final graph: {current_graph.number_of_nodes()} nodes, {current_graph.number_of_edges()} edges.")

    contracted_output_filename = os.path.join(
        args.output_dir, f"contracted_clusters_k{args.k_neighbors}_iter{iteration_count}.json"
    )
    output_clusters_to_json(current_graph, record_details_map, contracted_output_filename, is_initial_graph=False)

    # --- Evaluation ---
    print("\nStep 3: Evaluating results...")
    # discovered_true_pairs now contains all pairs of *original* IDs that were deemed to be
    # in the same cluster due to one or more contraction operations.

    tp = len(discovered_true_pairs.intersection(ground_truth_pairs))
    fp = len(discovered_true_pairs - ground_truth_pairs)
    fn = len(ground_truth_pairs - discovered_true_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # tp + fn is len(ground_truth_pairs)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Total Ground Truth Pairs: {len(ground_truth_pairs)}")
    print(f"  Total Discovered Merged Pairs: {len(discovered_true_pairs)}")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  Recall:               {recall:.4f}")
    print(f"  F1 Score:             {f1:.4f}")

    print("\nIterative Graph Contraction Process Finished.")


if __name__ == "__main__":
    main()
