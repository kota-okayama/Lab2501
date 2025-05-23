import csv
import json
import os
import time
import yaml
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)
import networkx as nx
from openai import OpenAI
import argparse
from collections import defaultdict

# --- グローバル設定 ---
# スクリプトのベースディレクトリ
SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 想定されるプロジェクトルート (siamese_model_pytorch ディレクトリの親)
PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(SCRIPT_BASE_DIR, ".."))

# グローバル変数として書誌データを保持
BIB_DATA = {}
# グローバル変数として正解クラスタIDを保持 (record_id -> cluster_id)
GROUND_TRUTH_CLUSTERS = {}
# キャッシュデータ
CACHE_DATA = {}
# OpenAI APIクライアント
OPENAI_CLIENT = None


# --- キャッシュ管理関数 (evaluate_finetuning_performance.py より) ---
def load_cache(cache_file_path):
    global CACHE_DATA
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "r", encoding="utf-8") as f:
                CACHE_DATA = json.load(f)
            print(f"{len(CACHE_DATA)} 件のキャッシュを {cache_file_path} からロードしました。")
        except json.JSONDecodeError as e:
            print(
                f"エラー: キャッシュファイル {cache_file_path} のJSON形式が正しくありません: {e}。新規キャッシュで開始します。"
            )
            CACHE_DATA = {}
        except Exception as e:
            print(
                f"エラー: キャッシュファイル {cache_file_path} の読み込み中に予期せぬエラー: {e}。新規キャッシュで開始します。"
            )
            CACHE_DATA = {}
    else:
        print(f"キャッシュファイル {cache_file_path} は見つかりませんでした。新規に作成されます。")


def save_cache(cache_file_path):
    global CACHE_DATA
    try:
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(CACHE_DATA, f, ensure_ascii=False, indent=4)
        print(f"{len(CACHE_DATA)} 件のキャッシュを {cache_file_path} に保存しました。")
    except Exception as e:
        print(f"エラー: キャッシュファイル {cache_file_path} への書き込み中にエラー: {e}")


# --- データ読み込み関数 (evaluate_finetuning_performance.py より調整) ---
def load_bib_data_and_gt_clusters(yaml_path):
    global BIB_DATA, GROUND_TRUTH_CLUSTERS
    BIB_DATA = {}
    GROUND_TRUTH_CLUSTERS = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        return False
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            if isinstance(possible_records_dict, dict):
                for _, value_list in possible_records_dict.items():
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])
                                BIB_DATA[record_id_str] = record.get("data", record)  # dataキーがなければrecord全体

                                if "cluster_id" in record:
                                    GROUND_TRUTH_CLUSTERS[record_id_str] = str(record["cluster_id"])
                                else:
                                    GROUND_TRUTH_CLUSTERS[record_id_str] = f"gt_orphan_{record_id_str}"
        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。")
            return False
        print(
            f"{len(BIB_DATA)} 件の書誌データ、{len(GROUND_TRUTH_CLUSTERS)} 件の正解クラスタIDを {yaml_path} からロードしました。"
        )
        return True
    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
        return False
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラー: {e}")
        return False


# --- ペア抽出関数 (extract_llm_pairs.py より) ---
def extract_unique_pairs_from_knn_graph(knn_graph_file_path):
    unique_pairs = set()
    try:
        with open(knn_graph_file_path, "r", encoding="utf-8") as f:
            knn_graph_data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: K近傍グラフファイルが見つかりません: {knn_graph_file_path}")
        return unique_pairs
    except json.JSONDecodeError:
        print(f"エラー: K近傍グラフファイルのJSON形式が正しくありません: {knn_graph_file_path}")
        return unique_pairs

    # 入力形式が隣接リスト辞書の場合
    if isinstance(knn_graph_data, dict) and all(isinstance(v, list) for v in knn_graph_data.values()):
        for record_id, neighbors in knn_graph_data.items():
            for neighbor_id in neighbors:
                if record_id == neighbor_id:
                    continue
                pair = tuple(sorted((str(record_id), str(neighbor_id))))
                unique_pairs.add(pair)
    # 入力形式が node-link 形式の場合 (networkx.node_link_data の出力)
    elif isinstance(knn_graph_data, dict) and "nodes" in knn_graph_data and "links" in knn_graph_data:
        # この形式の場合、linksがエッジリストそのもの
        for link in knn_graph_data["links"]:
            source_id = str(link.get("source"))
            target_id = str(link.get("target"))
            if source_id == target_id or not source_id or not target_id or source_id == "None" or target_id == "None":
                continue
            pair = tuple(sorted((source_id, target_id)))
            unique_pairs.add(pair)
    else:
        print(
            f"エラー: K近傍グラフファイル {knn_graph_file_path} の形式が不明です。隣接リストまたはnode-link形式を期待します。"
        )

    print(f"{len(unique_pairs)} 組のユニークな評価対象ペアを {knn_graph_file_path} から抽出しました。")
    return unique_pairs


# --- 書誌情報取得関数 (evaluate_finetuning_performance.py より) ---
def get_record_details_for_prompt(record_id):
    if not BIB_DATA:
        return "情報取得エラー: BIB_DATA未ロード"
    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    # 'data' サブキーがあるか、または bib_details が直接データか確認
    actual_data = bib_details.get("data", bib_details) if isinstance(bib_details, dict) else {}
    if not isinstance(actual_data, dict):  # bib_detailsが文字列などの場合
        return f"レコードID {record_id} の書誌情報が不正な形式です。"

    title = actual_data.get("title", actual_data.get("bib1_title", "タイトル不明"))
    authors_list = actual_data.get("author", actual_data.get("bib1_author", []))
    authors_str = ""
    if isinstance(authors_list, list):
        authors_str = ", ".join(filter(None, authors_list)) if authors_list else "著者不明"
    elif isinstance(authors_list, str):
        authors_str = authors_list if authors_list else "著者不明"
    else:
        authors_str = "著者不明"

    publisher = actual_data.get("publisher", actual_data.get("bib1_publisher", "出版社不明"))
    pubdate = actual_data.get("pubdate", actual_data.get("bib1_pubdate", "出版日不明"))
    details = f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"
    return details


# --- LLM呼び出し関連関数 (evaluate_finetuning_performance.py より) ---
def get_llm_evaluation_for_pair(record_id_1, record_id_2, model_id, cache_file_path):
    global CACHE_DATA, OPENAI_CLIENT
    cache_key = f"{record_id_1}_{record_id_2}_{model_id}"
    error_msg = None

    if cache_key in CACHE_DATA:
        cached_item = CACHE_DATA[cache_key]
        if isinstance(cached_item, dict) and "is_similar" in cached_item and "score" in cached_item:
            # print(f"  キャッシュヒット: ペア ({record_id_1}, {record_id_2}), モデル {model_id}") # 詳細ログ
            return cached_item["is_similar"], cached_item["score"], None
        else:
            print(f"警告: キャッシュキー {cache_key} のデータ形式が不正です。APIを呼び出します。")

    if not OPENAI_CLIENT:
        return None, None, "OpenAI APIクライアントが初期化されていません。"

    bib_info_1 = get_record_details_for_prompt(record_id_1)
    bib_info_2 = get_record_details_for_prompt(record_id_2)

    if "情報取得エラー" in bib_info_1 or "書誌情報なし" in bib_info_1 or "不正な形式" in bib_info_1:
        return None, None, f"レコード {record_id_1} の情報取得に失敗: {bib_info_1}"
    if "情報取得エラー" in bib_info_2 or "書誌情報なし" in bib_info_2 or "不正な形式" in bib_info_2:
        return None, None, f"レコード {record_id_2} の情報取得に失敗: {bib_info_2}"

    system_prompt = (
        "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\n"
        "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\n"
        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
    )
    user_prompt = (
        f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\n\n"
        f"書誌情報1:\n{bib_info_1}\n\n"
        f"書誌情報2:\n{bib_info_2}\n\n"
        "これらは同一の文献ですか？\n回答:"
    )

    try:
        time.sleep(0.1)  # レート制限対策
        completion = OPENAI_CLIENT.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        response_text = completion.choices[0].message.content.strip()

        lines = response_text.split("\n")
        is_similar_str = lines[0].strip().lower() if lines else ""
        similarity_score_str = ""
        score_keyword = "類似度スコア:"
        for line in lines:
            if score_keyword in line:
                similarity_score_str = line.split(score_keyword)[-1].strip().rstrip(".")  # 末尾のピリオドも除去
                break

        parsed_is_similar = None
        if "はい" in is_similar_str:
            parsed_is_similar = True
        elif "いいえ" in is_similar_str:
            parsed_is_similar = False
        else:  # 応答全体から探す
            if "はい" in response_text.lower() and "いいえ" not in response_text.lower():
                parsed_is_similar = True
            elif "いいえ" in response_text.lower() and "はい" not in response_text.lower():
                parsed_is_similar = False

        parsed_similarity_score = None
        if similarity_score_str:
            try:
                parsed_similarity_score = float(similarity_score_str)
                if not (0.0 <= parsed_similarity_score <= 1.0):
                    print(
                        f"警告: ペア ({record_id_1}, {record_id_2}) スコア '{parsed_similarity_score}' が範囲外。応答: {response_text}"
                    )
                    # parsed_similarity_score = max(0.0, min(1.0, parsed_similarity_score)) # 範囲内に丸めるかNoneにするか
            except ValueError:
                print(
                    f"警告: ペア ({record_id_1}, {record_id_2}) スコア変換失敗: '{similarity_score_str}'。応答: {response_text}"
                )

        if parsed_is_similar is None and parsed_similarity_score is not None:
            parsed_is_similar = parsed_similarity_score >= 0.5
        elif parsed_is_similar is not None and parsed_similarity_score is None:
            parsed_similarity_score = 1.0 if parsed_is_similar else 0.0

        if parsed_is_similar is None:
            return None, None, f"LLM応答から判定抽出不可。応答: {response_text}"
        # スコアがNoneでも判定があれば進めるように変更。スコアは後で補完される。
        # if parsed_similarity_score is None: return parsed_is_similar, None, f"LLM応答からスコア抽出不可。応答: {response_text}"

        CACHE_DATA[cache_key] = {
            "is_similar": parsed_is_similar,
            "score": (
                parsed_similarity_score if parsed_similarity_score is not None else (1.0 if parsed_is_similar else 0.0)
            ),
        }
        return parsed_is_similar, parsed_similarity_score, None

    except Exception as e:
        error_msg = f"OpenAI APIエラー (ペア: {record_id_1}, {record_id_2}, モデル: {model_id}): {e}"
        print(error_msg)
        # import traceback; traceback.print_exc() # 詳細デバッグ用
        return None, None, error_msg


# --- 評価指標計算関連関数 (evaluate_finetuning_performance.py より) ---
def evaluate_model_on_pairs(model_id, pairs_to_evaluate, cache_file_path):
    predictions, ground_truths, predicted_positive_pairs, llm_scores, errors, processed_pairs = [], [], [], [], [], []

    print(f"\nモデル '{model_id}' で {len(pairs_to_evaluate)} ペアの評価を開始します...")
    for i, (r_id1, r_id2) in enumerate(pairs_to_evaluate):
        processed_pairs.append((r_id1, r_id2))
        gt_cluster1 = GROUND_TRUTH_CLUSTERS.get(r_id1)
        gt_cluster2 = GROUND_TRUTH_CLUSTERS.get(r_id2)
        is_truly_similar = False
        if (
            gt_cluster1
            and gt_cluster2
            and not str(gt_cluster1).startswith("gt_orphan_")
            and not str(gt_cluster2).startswith("gt_orphan_")
            and gt_cluster1 == gt_cluster2
        ):
            is_truly_similar = True
        ground_truths.append(is_truly_similar)

        llm_is_similar, llm_score, error_msg = get_llm_evaluation_for_pair(r_id1, r_id2, model_id, cache_file_path)

        if error_msg:
            errors.append(((r_id1, r_id2), error_msg))
            predictions.append(False)  # エラー時はNegativeとみなす（またはNoneにして後で処理）
            llm_scores.append(0.0)  # エラー時はスコア0（またはNone）
            print(
                f"  ペア ({r_id1}, {r_id2}) 評価エラー: {error_msg[:200]}..."
            )  # エラーメッセージが長い場合があるので一部表示
        else:
            predictions.append(llm_is_similar)
            # llm_scoreがNoneの場合のフォールバック (get_llm_evaluation_for_pair内で処理されるようになったので基本的には不要だが念のため)
            current_score = llm_score if llm_score is not None else (1.0 if llm_is_similar else 0.0)
            llm_scores.append(current_score)
            if llm_is_similar:
                predicted_positive_pairs.append((r_id1, r_id2))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs_to_evaluate)} ペア処理完了...")
            save_cache(cache_file_path)
    print(f"モデル '{model_id}' での評価完了。エラー: {len(errors)}件。")
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "predicted_positive_pairs": predicted_positive_pairs,
        "llm_scores": llm_scores,
        "errors": errors,
        "processed_pairs": processed_pairs,
    }


def calculate_pairwise_metrics(ground_truths, predictions, model_name=""):
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average="binary", zero_division=0
    )
    print(f"\n--- {model_name} ペアワイズ評価指標 ---")
    print(f"  適合率 (Precision): {precision:.4f}")
    print(f"  再現率 (Recall):    {recall:.4f}")
    print(f"  F1スコア:           {f1:.4f}")
    return {"precision": precision, "recall": recall, "f1_score": f1}


def form_predicted_clusters(positive_pairs, all_record_ids):
    graph = nx.Graph()
    graph.add_nodes_from(all_record_ids)
    graph.add_edges_from(positive_pairs)
    predicted_cluster_map = {}
    cluster_label_counter = 0
    for component_nodes in nx.connected_components(graph):
        for node in component_nodes:
            predicted_cluster_map[node] = cluster_label_counter
        cluster_label_counter += 1
    for record_id in all_record_ids:
        if record_id not in predicted_cluster_map:
            predicted_cluster_map[record_id] = cluster_label_counter
            cluster_label_counter += 1
    return predicted_cluster_map


def calculate_clustering_metrics(true_cluster_map, pred_cluster_map, all_record_ids, model_name=""):
    true_labels = [true_cluster_map.get(rid, f"missing_gt_{rid}") for rid in all_record_ids]
    pred_labels = [pred_cluster_map.get(rid, -1) for rid in all_record_ids]
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method="arithmetic")
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    print(f"\n--- {model_name} クラスタリング評価指標 ---")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Completeness: {completeness:.4f}")
    print(f"  V-measure: {v_measure:.4f}")
    return {"ari": ari, "nmi": nmi, "homogeneity": homogeneity, "completeness": completeness, "v_measure": v_measure}


def format_clusters_with_details(predicted_clusters_map, bib_data_dict):
    grouped_clusters = defaultdict(list)
    for record_id, cluster_label in predicted_clusters_map.items():
        details_str = get_record_details_for_prompt(record_id)
        grouped_clusters[str(cluster_label)].append({"record_id": record_id, "details": details_str})

    sorted_grouped_clusters = {}
    for label in sorted(
        grouped_clusters.keys(), key=lambda x: int(x) if x.isdigit() else x
    ):  # 数値ラベルなら数値順ソート
        sorted_records = sorted(grouped_clusters[label], key=lambda x: x["record_id"])
        sorted_grouped_clusters[str(label)] = sorted_records
    return sorted_grouped_clusters


# --- メイン処理 ---
def main(args):
    global OPENAI_CLIENT

    # 出力ディレクトリの準備
    os.makedirs(args.output_dir, exist_ok=True)

    # キャッシュファイルのパス設定とロード
    cache_file_path = os.path.join(args.output_dir, "llm_api_cache.json")
    load_cache(cache_file_path)

    try:
        OPENAI_CLIENT = OpenAI()
    except Exception as e:
        print(f"OpenAI APIクライアントの初期化に失敗: {e}。環境変数 OPENAI_API_KEY を確認してください。")
        return

    print(f"評価処理を開始します。モデル: {args.model_id}")

    if not load_bib_data_and_gt_clusters(args.bib_yaml_path):
        return

    evaluation_pairs_set = extract_unique_pairs_from_knn_graph(args.knn_graph_path)
    if not evaluation_pairs_set:
        print("K近傍グラフから評価ペアを抽出できませんでした。処理を終了します。")
        return

    evaluation_pairs_list = sorted(list(evaluation_pairs_set))  # 順序を固定

    # 評価対象ペアに含まれる全ユニークなレコードIDのセットを取得
    all_record_ids_in_pairs = set()
    for r_id1, r_id2 in evaluation_pairs_list:
        all_record_ids_in_pairs.add(r_id1)
        all_record_ids_in_pairs.add(r_id2)
    all_record_ids_list = sorted(list(all_record_ids_in_pairs))

    # LLM評価実行
    results = evaluate_model_on_pairs(args.model_id, evaluation_pairs_list, cache_file_path)

    # ペアワイズ評価
    pairwise_metrics = calculate_pairwise_metrics(results["ground_truths"], results["predictions"], args.model_id)

    # クラスタリング評価
    pred_clusters_map = form_predicted_clusters(results["predicted_positive_pairs"], all_record_ids_list)
    clustering_metrics = calculate_clustering_metrics(
        GROUND_TRUTH_CLUSTERS, pred_clusters_map, all_record_ids_list, args.model_id
    )

    # 詳細結果のDataFrame作成とCSV保存
    results_df_data = []
    for i, (r_id1, r_id2) in enumerate(results["processed_pairs"]):
        error_for_pair = next((err_msg for pair_ids, err_msg in results["errors"] if pair_ids == (r_id1, r_id2)), None)
        results_df_data.append(
            {
                "record_id_1": r_id1,
                "record_id_2": r_id2,
                "ground_truth_similar": results["ground_truths"][i],
                "predicted_similar": results["predictions"][i],
                "llm_score": results["llm_scores"][i],
                "error": error_for_pair,
            }
        )
    detailed_results_df = pd.DataFrame(results_df_data)
    detailed_csv_path = os.path.join(args.output_dir, f"detailed_results_{args.model_id.replace(':', '_')}.csv")
    try:
        detailed_results_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n詳細な評価結果を {detailed_csv_path} に保存しました。")
    except Exception as e:
        print(f"エラー: 詳細な評価結果のCSV保存に失敗しました: {e}")

    # パフォーマンスレポートの作成と保存
    report_path = os.path.join(args.output_dir, f"performance_report_{args.model_id.replace(':', '_')}.txt")
    report_content = f"""# LLM評価レポート ({args.model_id})
日付: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 評価設定
- K近傍グラフ: {args.knn_graph_path}
- 書誌データYAML: {args.bib_yaml_path}
- LLMモデルID: {args.model_id}
- 評価ペア数: {len(evaluation_pairs_list)} (K近傍グラフから抽出)
- 評価対象レコード数 (ペア内ユニーク): {len(all_record_ids_in_pairs)}

## ペアワイズ評価
- 適合率 (Precision): {pairwise_metrics['precision']:.4f}
- 再現率 (Recall):    {pairwise_metrics['recall']:.4f}
- F1スコア:           {pairwise_metrics['f1_score']:.4f}
- 判定エラー数:       {len(results['errors'])}

## クラスタリング評価
- Adjusted Rand Index (ARI): {clustering_metrics['ari']:.4f}
- Normalized Mutual Information (NMI): {clustering_metrics['nmi']:.4f}
- Homogeneity:  {clustering_metrics['homogeneity']:.4f}
- Completeness: {clustering_metrics['completeness']:.4f}
- V-measure:    {clustering_metrics['v_measure']:.4f}
"""
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"パフォーマンスレポートを {report_path} に保存しました。")
    except Exception as e:
        print(f"エラー: パフォーマンスレポートの保存に失敗しました: {e}")

    # 予測クラスタ情報をJSONファイルに保存
    pred_clusters_json_path = os.path.join(
        args.output_dir, f"predicted_clusters_{args.model_id.replace(':', '_')}.json"
    )
    try:
        if pred_clusters_map:
            formatted_clusters = format_clusters_with_details(pred_clusters_map, BIB_DATA)
            with open(pred_clusters_json_path, "w", encoding="utf-8") as f:
                json.dump(formatted_clusters, f, ensure_ascii=False, indent=4)
            print(f"予測クラスタ情報を {pred_clusters_json_path} に保存しました。")
    except Exception as e:
        print(f"エラー: 予測クラスタ情報のJSON保存に失敗しました: {e}")

    print("\n評価処理が完了しました。")
    save_cache(cache_file_path)  # 最後にキャッシュを保存


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K近傍グラフから抽出したペアをLLMで評価し、性能を測定するスクリプト。")
    # --- 入力関連引数 ---
    parser.add_argument(
        "--knn_graph_path",
        type=str,
        required=True,
        help="K近傍グラフのJSONファイルパス (例: pipeline-output/knn_graph_eval/scenarioC_graph_output.json)",
    )
    parser.add_argument(
        "--bib_yaml_path",
        type=str,
        required=True,
        help="書誌情報と正解クラスタIDが含まれるYAMLファイルパス (例: benchmark/bib_japan_20241024/1k/record.yml)",
    )
    parser.add_argument(
        "--model_id_1",  # ここを確認
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="評価に使用するOpenAIモデルのID (1つ目、デフォルト: gpt-4o-mini-2024-07-18)",
    )
    parser.add_argument(
        "--model_id_2",  # ここを確認
        type=str,
        default=None,
        help="評価に使用するOpenAIモデルのID (2つ目、オプション、指定すると比較モード)",
    )
    # --- 出力関連引数 ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT_ASSUMED, "pipeline-output", "llm_evaluation_results"),
        help=f"評価結果（レポート、CSV、キャッシュ等）を保存するディレクトリ (デフォルト: PROJECT_ROOT/pipeline-output/llm_evaluation_results/)",
    )
    # --- オプション ---
    parser.add_argument(
        "--limit_pairs",
        type=int,
        default=None,
        help="評価するペア数を制限する (デバッグ用、指定しない場合は全ペア評価)",
    )

    parsed_args = parser.parse_args()

    # output_dir が絶対パスでない場合は、PROJECT_ROOT_ASSUMED からの相対パスとみなすこともできるが、
    # ここでは素直に指定されたパスを使用する。
    # デフォルト値は PROJECT_ROOT を基準にしているので、そのまま使えるはず。

    # もし --limit_pairs が指定された場合の処理を main 関数内に追加するか、
    # evaluation_pairs_list を作る際にスライスする。
    # ここでは main 関数に渡す前に処理する方がシンプルかもしれない。
    # (ただし、現状のコードでは main 関数内で evaluation_pairs_list を作成しているので、
    # main 関数内で --limit_pairs を見てスライスするのが自然)

    main(parsed_args)
