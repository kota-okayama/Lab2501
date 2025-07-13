import csv
import yaml
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

def load_llm_evaluations(filepath, score_column_name):
    """LLMの評価結果CSVを読み込み、ペアとその類似度スコアを辞書で返す"""
    pair_scores = {}
    if not os.path.exists(filepath):
        print(f"エラー: LLM評価結果ファイルが見つかりません: {filepath}")
        return pair_scores

    try:
        with open(filepath, "r", newline="", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                try:
                    id1 = row["record_id_1"]
                    id2 = row["record_id_2"]
                    score_str = row.get(score_column_name)

                    if score_str is None or score_str.lower() == "none" or score_str == "":
                        score = 0.0
                    else:
                        score = float(score_str)

                    key = tuple(sorted((id1, id2)))
                    pair_scores[key] = score
                except ValueError:
                    print(f"警告: スコアの数値変換に失敗しました: {score_str} (行: {row})")
                    continue
                except KeyError:
                    print(f"警告: CSVに必要なキーが見つかりません (record_id_1, record_id_2, or {score_column_name})")
                    continue
    except Exception as e:
        print(f"LLM評価結果ファイル ({filepath}) の読み込み中にエラー: {e}")

    print(f"{len(pair_scores)} 件のペア評価を {filepath} からロードしました。")
    return pair_scores

def load_record_clusters(yaml_path):
    """record.yml を読み込み、record_id と cluster_id の対応辞書を返す"""
    record_to_cluster = {}
    bib_data = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        return record_to_cluster, bib_data

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        key = 'records' if 'records' in data else 'clusters'
        for cluster_id, records in data[key].items():
            for record in records:
                record_id = record.get('id') or record.get('record_id')
                record_to_cluster[str(record_id)] = str(cluster_id)
                bib_data[str(record_id)] = record.get('data', record)

    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラーが発生しました: {e}")

    print(f"{len(record_to_cluster)} 件のレコードとクラスタIDの対応を {yaml_path} からロードしました。")
    return record_to_cluster, bib_data

def find_inconsistent_triangles(pair_scores, record_to_cluster, num_triangles_to_select):
    """非一貫性のある三角形を見つけ、スコア順にソートして返す"""
    all_record_ids = set()
    for id1, id2 in pair_scores.keys():
        all_record_ids.add(id1)
        all_record_ids.add(id2)

    adj = defaultdict(set)
    for id1, id2 in pair_scores.keys():
        adj[id1].add(id2)
        adj[id2].add(id1)

    inconsistent_triangles_data = []
    sorted_nodes = sorted(list(all_record_ids))

    print("三角形の列挙と非一貫性スコアの計算を開始します...")
    for u_node in tqdm(sorted_nodes, desc="ノード処理中"):
        u_neighbors = list(adj[u_node])
        for i in range(len(u_neighbors)):
            for j in range(i + 1, len(u_neighbors)):
                v_node = u_neighbors[i]
                w_node = u_neighbors[j]

                # v-w 間のエッジが存在するか確認 (三角形の成立)
                if w_node in adj[v_node]:
                    key_uv = tuple(sorted((u_node, v_node)))
                    key_vw = tuple(sorted((v_node, w_node)))
                    key_wu = tuple(sorted((u_node, w_node)))

                    p_uv = pair_scores.get(key_uv, 0)
                    p_vw = pair_scores.get(key_vw, 0)
                    p_wu = pair_scores.get(key_wu, 0)

                    inconsistency = p_uv * p_vw * (1 - p_wu) + p_vw * p_wu * (1 - p_uv) + p_wu * p_uv * (1 - p_vw)

                    c_u, c_v, c_w = record_to_cluster.get(u_node), record_to_cluster.get(v_node), record_to_cluster.get(w_node)
                    true_uv = c_u == c_v if c_u and c_v else None
                    true_vw = c_v == c_w if c_v and c_w else None
                    true_wu = c_u == c_w if c_u and c_w else None

                    inconsistent_triangles_data.append({
                        "triangle": tuple(sorted((u_node, v_node, w_node))),
                        "inconsistency_score": inconsistency,
                        "p_uv": p_uv, "p_vw": p_vw, "p_wu": p_wu,
                        "true_uv": true_uv, "true_vw": true_vw, "true_wu": true_wu,
                        "c_u": c_u, "c_v": c_v, "c_w": c_w,
                    })

    # 重複する三角形を削除
    unique_triangles = {d['triangle']: d for d in inconsistent_triangles_data}
    sorted_triangles = sorted(unique_triangles.values(), key=lambda x: x["inconsistency_score"], reverse=True)
    
    print(f"\n計算が完了したユニークな三角形の数: {len(sorted_triangles)}")
    
    return sorted_triangles[:num_triangles_to_select]

def save_results(selected_triangles, output_dir, base_filename, pair_scores, record_to_cluster):
    """分析結果をCSVファイルに保存する"""
    if not selected_triangles:
        print("矛盾三角形は見つかりませんでした。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # --- 矛盾三角形リストの保存 ---
    output_triangles_path = os.path.join(output_dir, f"{base_filename}_inconsistent_triangles.csv")
    try:
        with open(output_triangles_path, "w", newline="", encoding="utf-8") as outfile:
            fieldnames = [
                "triangle_node1", "triangle_node2", "triangle_node3",
                "inconsistency_score", "p_edge12", "p_edge23", "p_edge31",
                "true_edge12", "true_edge23", "true_edge31",
                "cluster_id1", "cluster_id2", "cluster_id3",
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in selected_triangles:
                nodes = data["triangle"]
                # 辺とノードの対応を合わせる
                p_uv = pair_scores[tuple(sorted((nodes[0], nodes[1])))]
                p_vw = pair_scores[tuple(sorted((nodes[1], nodes[2])))]
                p_wu = pair_scores[tuple(sorted((nodes[0], nodes[2])))]
                
                writer.writerow({
                    "triangle_node1": nodes[0], "triangle_node2": nodes[1], "triangle_node3": nodes[2],
                    "inconsistency_score": data["inconsistency_score"],
                    "p_edge12": p_uv, "p_edge23": p_vw, "p_edge31": p_wu,
                    "true_edge12": data["true_uv"], "true_edge23": data["true_vw"], "true_edge31": data["true_wu"],
                    "cluster_id1": data["c_u"], "cluster_id2": data["c_v"], "cluster_id3": data["c_w"],
                })
        print(f"矛盾三角形の詳細は {output_triangles_path} に保存されました。")
    except Exception as e:
        print(f"エラー: 矛盾三角形のCSVファイル書き込み中にエラー: {e}")

    # --- レビュー対象ペアリストの保存 ---
    review_pairs = set()
    for data in selected_triangles:
        n = data["triangle"]
        review_pairs.add(tuple(sorted((n[0], n[1]))))
        review_pairs.add(tuple(sorted((n[1], n[2]))))
        review_pairs.add(tuple(sorted((n[0], n[2]))))

    output_review_path = os.path.join(output_dir, f"{base_filename}_review_candidate_pairs.csv")
    try:
        with open(output_review_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["record_id_1", "record_id_2", "llm_similarity_score", "cluster_id_1", "cluster_id_2", "is_same_cluster_gt"])
            for id1, id2 in sorted(list(review_pairs)):
                score = pair_scores.get(tuple(sorted((id1, id2))), "N/A")
                c1, c2 = record_to_cluster.get(id1, "N/A"), record_to_cluster.get(id2, "N/A")
                same_cluster = c1 == c2 if c1 != "N/A" and c2 != "N/A" else None
                writer.writerow([id1, id2, score, c1, c2, same_cluster])
        print(f"レビュー対象ペアリストは {output_review_path} に保存されました。({len(review_pairs)}ペア)")
    except Exception as e:
        print(f"エラー: レビュー対象ペアのCSVファイル書き込み中にエラー: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLM評価結果から推移律に矛盾する三角形を検出するスクリプト")
    parser.add_argument("--input-csv", required=True, help="ペアごとのLLM評価結果が記載されたCSVファイル")
    parser.add_argument("--ground-truth-yaml", required=True, help="正解クラスタ情報が記載されたYAMLファイル")
    parser.add_argument("--score-column", required=True, help="類似度スコアとして使用する列名 (例: score_after)")
    parser.add_argument("--output-dir", default=".", help="出力ファイルを保存するディレクトリ")
    parser.add_argument("--num-triangles", type=int, default=100, help="出力する矛盾三角形の最大数")

    args = parser.parse_args()

    print("矛盾検出処理を開始します...")
    
    pair_scores = load_llm_evaluations(args.input_csv, args.score_column)
    record_to_cluster, _ = load_record_clusters(args.ground_truth_yaml)

    if not pair_scores:
        print("LLM評価スコアがロードできなかったため、処理を終了します。")
        return

    selected_triangles = find_inconsistent_triangles(pair_scores, record_to_cluster, args.num_triangles)
    
    # 出力用のベースファイル名を生成
    base_filename = os.path.basename(args.input_csv).replace("_details.csv", "")
    base_filename = f"{base_filename}_{args.score_column}"

    save_results(selected_triangles, args.output_dir, base_filename, pair_scores, record_to_cluster)

    print("\n処理が完了しました。")

if __name__ == "__main__":
    main()
