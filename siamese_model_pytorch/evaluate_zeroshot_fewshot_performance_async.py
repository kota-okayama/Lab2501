import asyncio
# import aiohttp  <- 不要になったのでコメントアウト
import time
import json
import pickle
import yaml
import csv
import os
import argparse
import itertools
import hashlib
import networkx as nx
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure
)
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


# --- グローバル変数 ---
BIB_DATA = {}  # {record_id: bib_details_dict}
GROUND_TRUTH_CLUSTERS = {}  # {record_id: cluster_id}
AVAILABLE_FIELDS = []
CACHE_DATA = {}  # {cache_key: {"is_similar": bool, "score": float}}
CACHE_FILE = "llm_evaluation_cache.pkl"

# APIレート制限パラメータ
MAX_CONCURRENT_REQUESTS = 20  # 同時リクエスト最大数
REQUESTS_PER_MINUTE = 3000    # 1分間の最大リクエスト数
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE  # リクエスト間の最小間隔

# デフォルト設定
DEFAULT_MODEL_ID = "gpt-4o-mini-2024-07-18"


def get_prompts(data_type):
    """データタイプに応じたプロンプトのエンティティと情報を返す"""
    prompt_map = {
        "bib": {"entity": "文献", "info": "書誌情報"},
        "music": {"entity": "楽曲", "info": "楽曲情報"},
        "person": {"entity": "人物", "info": "人物情報"},
        "walmart_amazon_product": {"entity": "製品", "info": "製品情報"},
        "wdc_product": {"entity": "製品", "info": "製品情報"},
        "unknown": {"entity": "レコード", "info": "情報"}
    }
    if data_type not in prompt_map:
        # フォールバック
        return prompt_map["unknown"]
    return prompt_map[data_type]


def load_cache():
    """キャッシュファイルを読み込む（PickleとJSONの両方をサポート）"""
    global CACHE_DATA
    CACHE_DATA = {}
    
    # 既存のJSONキャッシュを読み込み
    json_cache_path = (
        "openai_embedding_experiment/evaluation_results/llm_api_cache.json"
    )
    if os.path.exists(json_cache_path):
        try:
            with open(json_cache_path, 'r', encoding='utf-8') as f:
                json_cache = json.load(f)
            CACHE_DATA.update(json_cache)
            print(f"JSONキャッシュを読み込みました: {len(json_cache)} 件")
        except Exception as e:
            print(f"JSONキャッシュファイルの読み込みに失敗: {e}")
    
    # Pickleキャッシュを読み込み（JSONキャッシュと統合）
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                pickle_cache = pickle.load(f)
            # Pickleキャッシュで上書き（より新しいデータと仮定）
            CACHE_DATA.update(pickle_cache)
            print(f"Pickleキャッシュを読み込みました: {len(pickle_cache)} 件")
        except Exception as e:
            print(f"Pickleキャッシュファイルの読み込みに失敗: {e}")
    
    if not CACHE_DATA:
        print("キャッシュファイルが見つかりません。新規作成します。")
    else:
        print(f"総キャッシュ件数: {len(CACHE_DATA)} 件")


def save_cache(pbar=None):
    """キャッシュをPickleとJSONの両方の形式でファイルに保存する"""
    global CACHE_DATA

    # 1. Pickle形式で保存 (高速)
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(CACHE_DATA, f)
        message = f"Pickleキャッシュを保存しました: {len(CACHE_DATA)} 件"
        if pbar:
            pbar.write(message)
        else:
            print(message)
    except Exception as e:
        error_message = f"Pickleキャッシュファイルの保存に失敗: {e}"
        if pbar:
            pbar.write(error_message)
        else:
            print(error_message)

    # 2. JSON形式で保存 (可読性)
    json_cache_path = "openai_embedding_experiment/evaluation_results/llm_api_cache.json"
    try:
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(json_cache_path), exist_ok=True)
        with open(json_cache_path, 'w', encoding='utf-8') as f:
            json.dump(CACHE_DATA, f, indent=4, ensure_ascii=False)
        message = f"JSONキャッシュを保存しました: {len(CACHE_DATA)} 件"
        if pbar:
            pbar.write(message)
        else:
            print(message)
    except Exception as e:
        error_message = f"JSONキャッシュファイルの保存に失敗: {e}"
        if pbar:
            pbar.write(error_message)
        else:
            print(error_message)


def load_bib_data_and_gt_clusters(yaml_path):
    """
    YAMLファイルから書誌データと正解クラスタ情報を読み込む
    """
    global BIB_DATA, GROUND_TRUTH_CLUSTERS, AVAILABLE_FIELDS
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAMLファイルが見つかりません: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    AVAILABLE_FIELDS = list(data.get("inf_attr", {}).keys())
    
    BIB_DATA = {}
    GROUND_TRUTH_CLUSTERS = {}
    
    # YAMLファイルの構造に応じて処理を分岐
    if 'records' in data:
        # 新しい形式: records キーを使用
        orphan_counter = 0
        
        for cluster_id, records in data['records'].items():
            cluster_id_str = str(cluster_id)  # 一貫性のため文字列に統一
            for record in records:
                record_id = record['id']  # 'id' フィールドを使用
                # dataフィールドから書誌情報を取得し、record_idを追加
                bib_record = record['data'].copy()
                bib_record['record_id'] = record_id
                BIB_DATA[record_id] = bib_record
                
                if len(records) == 1:
                    # 単一レコードのクラスタは孤立ノードとして特別扱い
                    GROUND_TRUTH_CLUSTERS[record_id] = (
                        f"gt_orphan_{orphan_counter}"
                    )
                    orphan_counter += 1
                else:
                    GROUND_TRUTH_CLUSTERS[record_id] = cluster_id_str
    elif 'clusters' in data:
        # 旧形式: clusters キーを使用
        orphan_counter = 0
        
        for cluster_id, records in data['clusters'].items():
            cluster_id_str = str(cluster_id)  # 一貫性のため文字列に統一
            for record in records:
                record_id = record['record_id']
                BIB_DATA[record_id] = record
                
                if len(records) == 1:
                    # 単一レコードのクラスタは孤立ノードとして特別扱い
                    GROUND_TRUTH_CLUSTERS[record_id] = (
                        f"gt_orphan_{orphan_counter}"
                    )
                    orphan_counter += 1
                else:
                    GROUND_TRUTH_CLUSTERS[record_id] = cluster_id_str
    else:
        raise ValueError("YAMLファイルに 'records' または 'clusters' キーがありません")
    
    print(f"書誌データを読み込みました: {len(BIB_DATA)} レコード")
    print(f"正解クラスタ情報を読み込みました: {len(GROUND_TRUTH_CLUSTERS)} レコード")
    
    # クラスタサイズの分布を表示
    cluster_sizes = {}
    for cluster_id in GROUND_TRUTH_CLUSTERS.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    
    size_distribution = {}
    for size in cluster_sizes.values():
        size_distribution[size] = size_distribution.get(size, 0) + 1
    
    print("クラスタサイズ分布:")
    for size in sorted(size_distribution.keys()):
        print(f"  サイズ{size}: {size_distribution[size]}クラスタ")


def load_evaluation_pairs(csv_path):
    """
    CSVファイルから評価用ペアを読み込む
    """
    pairs = []
    all_record_ids = set()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # ヘッダーを読み飛ばす
        
        for row in reader:
            if len(row) >= 2:
                record_id_1, record_id_2 = row[0].strip(), row[1].strip()
                pairs.append((record_id_1, record_id_2))
                all_record_ids.add(record_id_1)
                all_record_ids.add(record_id_2)
    
    print(f"評価ペアを読み込みました: {len(pairs)} ペア ({len(all_record_ids)} ユニークレコード)")
    return pairs, all_record_ids


def get_record_details_for_prompt(record_id):
    """
    レコードIDから詳細情報を取得してプロンプト形式で返す (汎用版)
    """
    global BIB_DATA, AVAILABLE_FIELDS
    
    record_details = BIB_DATA.get(str(record_id))
    if not record_details:
        return f"レコードID {record_id} の情報なし"

    parts = []
    # YAMLのinf_attrで指定されたフィールドを使う。なければ全フィールド。
    if AVAILABLE_FIELDS:
        fields_to_use = AVAILABLE_FIELDS
    else:
        fields_to_use = list(record_details.keys())
    
    for field in fields_to_use:
        if value := record_details.get(field):
            parts.append(f"{field}: {value}")
            
    if not parts:
        return f"レコードID {record_id} に利用可能な情報なし"
        
    return "\n".join(parts)


class RateLimiter:
    """非同期処理用のレート制限クラス"""
    
    def __init__(self, max_requests_per_minute=3000):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """リクエスト許可を取得（必要に応じて待機）"""
        async with self.lock:
            now = time.time()
            # 1分以上古いリクエスト履歴を削除
            one_minute_ago = now - 60.0
            self.request_times = [
                t for t in self.request_times if t > one_minute_ago
            ]
            
            if len(self.request_times) >= self.max_requests_per_minute:
                # レート制限に達している場合は待機
                oldest_request = min(self.request_times)
                sleep_time = 60.0 - (now - oldest_request) + 0.1  # 少し余裕を持たせる
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.time())


async def get_llm_evaluation_for_pair_async(
    client, record_id_1, record_id_2, model_id,
    rate_limiter, system_prompt, few_shot_examples=None, no_cache=False,
    data_type="unknown"
):
    """
    非同期でLLM評価を実行する関数 (openaiライブラリ使用)
    """
    global CACHE_DATA

    is_fewshot = few_shot_examples is not None

    if is_fewshot:
        # Few-shot評価の場合：プロンプト内容のハッシュをキーに含める
        prompt_content_str = system_prompt
        prompt_content_str += json.dumps(
            few_shot_examples, sort_keys=True, ensure_ascii=False
        )
        prompt_hash = hashlib.sha256(
            prompt_content_str.encode('utf-8')
        ).hexdigest()[:16]
        cache_key = (
            f"{record_id_1}_{record_id_2}_{model_id}_{data_type}_{prompt_hash}"
        )
    else:
        # Zero-shot評価の場合：finetuning評価スクリプトとロジックを統一
        if data_type == 'bib':
            cache_key = f"{record_id_1}_{record_id_2}_{model_id}"
        else:
            cache_key = f"{record_id_1}_{record_id_2}_{model_id}_{data_type}"

    # キャッシュチェック
    if not no_cache and cache_key in CACHE_DATA:
        cached_item = CACHE_DATA[cache_key]
        if (isinstance(cached_item, dict) and
                "is_similar" in cached_item and "score" in cached_item):
            return cached_item["is_similar"], cached_item["score"], None

    # 書誌情報取得
    bib_info_1 = get_record_details_for_prompt(record_id_1)
    bib_info_2 = get_record_details_for_prompt(record_id_2)
    
    if "情報取得エラー" in bib_info_1 or "情報なし" in bib_info_1:
        return None, None, f"レコード {record_id_1} の情報取得に失敗: {bib_info_1}"
    if "情報取得エラー" in bib_info_2 or "情報なし" in bib_info_2:
        return None, None, f"レコード {record_id_2} の情報取得に失敗: {bib_info_2}"

    messages = [{"role": "system", "content": system_prompt}]

    # Few-shotの例をメッセージに追加
    if few_shot_examples:
        messages.extend(few_shot_examples)

    data_type_info = get_prompts(data_type)
    info_name = data_type_info['info']
    entity_name = data_type_info['entity']

    user_prompt = (
        f"以下の2つの{info_name}が、実質的に同一の{entity_name}を指しているかどうかを判断してください。\n\n"
        f"{info_name}1:\n{bib_info_1}\n\n"
        f"{info_name}2:\n{bib_info_2}\n\n"
        f"これらは同一の{entity_name}ですか？\n回答:"
    )
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        # レート制限を適用
        await rate_limiter.acquire()
        
        # OpenAI API呼び出し（公式ライブラリ使用）
        completion = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        response_text = completion.choices[0].message.content.strip()
        
        # レスポンスのパース
        lines = response_text.split("\n")
        is_similar_str = ""
        similarity_score_str = ""
        
        if lines:
            is_similar_str = lines[0].strip().lower()
        
        score_keyword = "類似度スコア:"
        for line in lines:
            if score_keyword in line:
                similarity_score_str = line.split(score_keyword)[-1].strip()
                break
        
        # パース処理
        parsed_is_similar = None
        if "はい" in is_similar_str:
            parsed_is_similar = True
        elif "いいえ" in is_similar_str:
            parsed_is_similar = False
        else:
            # 応答全体から探す
            if "はい" in response_text and "いいえ" not in response_text:
                parsed_is_similar = True
            elif "いいえ" in response_text and "はい" not in response_text:
                parsed_is_similar = False
        
        parsed_similarity_score = None
        if similarity_score_str:
            try:
                parsed_similarity_score = float(similarity_score_str)
                if not (0.0 <= parsed_similarity_score <= 1.0):
                    print(
                        f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコア "
                        f"'{parsed_similarity_score}' が範囲外です。"
                    )
            except ValueError:
                print(
                    f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコアが"
                    f"数値に変換できません: '{similarity_score_str}'"
                )
        
        # フォールバック処理
        if parsed_is_similar is None and parsed_similarity_score is not None:
            parsed_is_similar = parsed_similarity_score >= 0.5
        elif parsed_is_similar is not None and parsed_similarity_score is None:
            parsed_similarity_score = 1.0 if parsed_is_similar else 0.0
        
        if parsed_is_similar is None:
            return None, None, f"LLMの応答から判定を抽出できませんでした。応答: {response_text}"
        if parsed_similarity_score is None:
            return (
                parsed_is_similar,
                None,
                f"LLMの応答から類似度スコアを抽出できませんでした。応答: {response_text}"
            )
        
        # キャッシュに保存
        if not no_cache:
            CACHE_DATA[cache_key] = {
                "is_similar": parsed_is_similar,
                "score": parsed_similarity_score,
            }
        
        return parsed_is_similar, parsed_similarity_score, None
        
    except Exception as e:
        # openai.APIErrorから詳細なエラー情報を取得
        if hasattr(e, 'message'):
            error_details = e.message
        elif hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_details = e.response.text
        else:
            error_details = str(e)
            
        error_msg = (
            f"API呼び出し中にエラーが発生 (ペア: {record_id_1}, {record_id_2}): "
            f"{error_details}"
        )
        return None, None, error_msg


async def evaluate_model_on_pairs_async(
    model_id,
    pairs_to_evaluate,
    all_record_ids_in_pairs,
    api_key,
    system_prompt,
    few_shot_examples=None,
    no_cache=False,
    data_type="unknown"
):
    """
    非同期でモデル評価を実行する関数
    """
    predictions = []
    ground_truths = []
    predicted_positive_pairs = []
    llm_scores = []
    errors = []
    processed_pairs = []
    
    mode = "Few-shot" if few_shot_examples else "Zero-shot"
    print(
        f"\nモデル '{model_id}' ({mode}) で "
        f"{len(pairs_to_evaluate)} ペアの非同期評価を開始します..."
    )
    
    # レート制限器とHTTPセッションを初期化
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    client = AsyncOpenAI(api_key=api_key)

    async def evaluate_single_pair(pair_info):
        async with semaphore:
            r_id1, r_id2, index = pair_info
            
            # 正解ラベルの決定
            gt_cluster1 = GROUND_TRUTH_CLUSTERS.get(r_id1)
            gt_cluster2 = GROUND_TRUTH_CLUSTERS.get(r_id2)
            
            is_truly_similar = (
                gt_cluster1 is not None and
                gt_cluster2 is not None and
                not gt_cluster1.startswith("gt_orphan_") and
                not gt_cluster2.startswith("gt_orphan_") and
                gt_cluster1 == gt_cluster2
            )
            
            # LLM評価実行
            llm_is_similar, llm_score, error_msg = (
                await get_llm_evaluation_for_pair_async(
                    client, r_id1, r_id2, model_id,
                    rate_limiter, system_prompt, few_shot_examples,
                    no_cache=no_cache, data_type=data_type
                )
            )
            
            if error_msg:
                return {
                    'index': index,
                    'pair': (r_id1, r_id2),
                    'error': error_msg,
                    'is_valid_result': False
                }

            return {
                'index': index,
                'pair': (r_id1, r_id2),
                'ground_truth': is_truly_similar,
                'prediction': llm_is_similar,
                'score': llm_score,
                'error': None,
                'is_positive': llm_is_similar,
                'is_valid_result': True
            }
    
    # ペア情報にインデックスを追加
    pair_info_list = [
        (pair[0], pair[1], i) for i, pair in enumerate(pairs_to_evaluate)
    ]
    
    # バッチサイズを設定してメモリ使用量を制御
    batch_size = MAX_CONCURRENT_REQUESTS * 2
    
    # 進捗バーを初期化（改行せずに更新）
    desc = f"評価中 ({model_id} - {mode})"
    with tqdm(total=len(pair_info_list), desc=desc,
              unit="ペア", leave=True, ncols=100) as pbar:
        for i in range(0, len(pair_info_list), batch_size):
            batch = pair_info_list[i:i + batch_size]
            
            # バッチを並列実行
            tasks = [evaluate_single_pair(pair_info) for pair_info in batch]
            batch_results = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # 結果を整理
            batch_task_errors = 0
            for result in batch_results:
                pbar.update(1)
                if isinstance(result, Exception):
                    batch_task_errors += 1
                    continue
                
                if not result.get('is_valid_result', False):
                    if result.get('error'):
                        errors.append((result['pair'], result['error']))
                    continue
                
                processed_pairs.append(result['pair'])
                ground_truths.append(result['ground_truth'])
                predictions.append(result['prediction'])
                llm_scores.append(result['score'])
                
                if result['is_positive']:
                    predicted_positive_pairs.append(result['pair'])
            
            # 進捗バーにエラー情報を表示
            if batch_task_errors > 0:
                pbar.set_description(
                    f"{desc} [タスクエラー: {batch_task_errors}件]"
                )
            if len(errors) > 0:
                pbar.set_postfix_str(f"APIエラー: {len(errors)}件")
            
            # 進捗保存
            if not no_cache and i % (batch_size * 3) == 0:  # 3バッチごとにキャッシュ保存
                save_cache(pbar)
    
    print(f"モデル '{model_id}' ({mode}) での非同期評価完了。エラー: {len(errors)}件。")
    
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "predicted_positive_pairs": predicted_positive_pairs,
        "llm_scores": llm_scores,
        "errors": errors,
        "processed_pairs": processed_pairs,
    }


def calculate_pairwise_metrics(ground_truths, predictions, model_name=""):
    """ペアごとの評価指標と混合行列を計算する"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average="binary", zero_division=0
    )
    
    cm = confusion_matrix(ground_truths, predictions, labels=[False, True])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1 and len(set(ground_truths)) == 1:
        label_val = list(set(ground_truths))[0]
        if label_val is False:
            tn, fp, fn, tp = cm.item(), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm.item()
    else:
        print(f"警告: {model_name} の混合行列が予期せぬ形状です。")
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    print(f"\n--- {model_name} ペアワイズ評価指標 ---")
    print("  混合行列 (Positive: 一致ペア, Negative: 不一致ペア):")
    print("  +----------------+-----------------+-----------------+")
    print(
        f"  | {'Ground Truth':^14} | {'Predicted: Pos':^15} "
        f"| {'Predicted: Neg':^15} |"
    )
    print("  +================+=================+=================+")
    print(f"  | {'Positive':<14} | TP: {tp:<12d} | FN: {fn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  | {'Negative':<14} | FP: {fp:<12d} | TN: {tn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  適合率 (Precision): {precision:.4f} (TP / (TP + FP))")
    print(f"  再現率 (Recall):    {recall:.4f} (TP / (TP + FN))")
    print(f"  F1スコア:           {f1:.4f}")
    
    return {
        "precision": precision, "recall": recall, "f1_score": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def form_predicted_clusters(positive_pairs, all_record_ids):
    """LLMが「はい」と判定したペアから予測クラスタを形成する"""
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


def calculate_clustering_metrics(
    true_cluster_map, pred_cluster_map, all_record_ids, model_name=""
):
    """クラスタリング評価指標を計算する"""
    true_labels = [
        true_cluster_map.get(rid, f"missing_gt_{rid}")
        for rid in all_record_ids
    ]
    pred_labels = [pred_cluster_map.get(rid, -1) for rid in all_record_ids]
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )
    
    print(f"\n--- {model_name} クラスタリング評価指標 ---")
    print(f"  調整ランド指数 (ARI): {ari:.4f}")
    print(f"  正規化相互情報量 (NMI): {nmi:.4f}")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Completeness: {completeness:.4f}")
    print(f"  V-measure: {v_measure:.4f}")
    
    return {
        "ari": ari, "nmi": nmi, "homogeneity": homogeneity,
        "completeness": completeness, "v_measure": v_measure
    }


def sanitize_model_name_for_filename(model_name):
    """ファイル名に使用できるようにモデル名をサニタイズする"""
    return model_name.replace('/', '_').replace(':', '_').replace(' ', '_')


def format_clusters_with_details(predicted_clusters, bib_data_dict):
    """
    予測されたクラスタ情報を、レコード詳細を含めて整形する。
    """
    grouped_clusters = {}
    for record_id, cluster_label in predicted_clusters.items():
        if cluster_label not in grouped_clusters:
            grouped_clusters[cluster_label] = []

        details_str = get_record_details_for_prompt(record_id)
        grouped_clusters[cluster_label].append(
            {"record_id": record_id, "details": details_str}
        )
    
    sorted_grouped_clusters = {}
    for label in sorted(grouped_clusters.keys()):
        sorted_records = sorted(
            grouped_clusters[label], key=lambda x: x["record_id"]
        )
        sorted_grouped_clusters[str(label)] = sorted_records
    return sorted_grouped_clusters


def load_few_shot_prompt_and_examples(file_path, max_examples=None):
    """
    JSONファイルからFew-shot用のシステムプロンプトとお手本を読み込む
    """
    if not file_path or not os.path.exists(file_path):
        print("Few-shotデータファイルが見つからないため、Zero-shot評価のみ行います。")
        return None, None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        system_prompt = data.get("system_prompt")
        fewshot_examples = data.get("fewshot_examples")

        if not system_prompt or not fewshot_examples:
            print(
                f"警告: {file_path} に 'system_prompt' または "
                f"'fewshot_examples' が見つかりません。"
            )
            return None, None
            
        # Limit the number of examples if specified
        if max_examples is not None and max_examples > 0:
            # Each example is a user/assistant pair, 
            # so we take max_examples * 2 messages
            num_messages = max_examples * 2
            if len(fewshot_examples) > num_messages:
                fewshot_examples = fewshot_examples[:num_messages]
                print(f"Few-shotのお手本を最初の {max_examples} 件に制限しました。")
            
        print(f"Few-shotデータを読み込みました: {len(fewshot_examples) // 2} 例")
        return system_prompt, fewshot_examples
    except json.JSONDecodeError:
        print(f"警告: {file_path} のJSONパースに失敗しました。")
        return None, None
    except Exception as e:
        print(f"警告: Few-shotデータファイル({file_path})の読み込み中にエラー: {e}")
        return None, None


async def main(args):
    # OpenAI APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY 環境変数が設定されていません。")
        return

    # --- 引数とパスの整合性チェック ---
    path_str_for_check = (
        f"{args.ground_truth_yaml.lower()}_{args.pairs_csv.lower()}"
    )
    inferred_type = None
    if 'person' in path_str_for_check:
        inferred_type = 'person'
    elif 'music' in path_str_for_check:
        inferred_type = 'music'
    elif 'bib' in path_str_for_check:
        inferred_type = 'bib'

    if inferred_type and inferred_type != args.data_type:
        print("\n" + "="*80)
        print("【警告】引数とファイルパスのデータタイプが一致しない可能性があります。")
        print(f"  - 引数 (--data_type): {args.data_type}")
        print(f"  - ファイルパスから推測: {inferred_type}")
        print(f"  - YAML: {args.ground_truth_yaml}")
        print(f"  - CSV: {args.pairs_csv}")
        print("意図しないプロンプトやキャッシュが使用され、評価結果が不正確になる恐れがあります。")
        print("="*80 + "\n")
        # 続行するが、ユーザーは警告に気づくはず
    
    # キャッシュと書誌データの読み込み
    if not args.no_cache:
        load_cache()
    else:
        print("キャッシュを使用しないモードで実行します。")
    load_bib_data_and_gt_clusters(args.ground_truth_yaml)
    
    # 評価ペアの読み込み
    pairs_to_evaluate, all_record_ids_in_pairs = load_evaluation_pairs(
        args.pairs_csv
    )
    
    # 評価ペアを制限する
    if args.limit is not None and args.limit > 0:
        print(f"評価ペアを最初の {args.limit} 件に制限します。")
        if len(pairs_to_evaluate) > args.limit:
            pairs_to_evaluate = pairs_to_evaluate[:args.limit]
            # 制限後のペアに存在するIDのみを再計算
            limited_ids = set()
            for r1, r2 in pairs_to_evaluate:
                limited_ids.add(r1)
                limited_ids.add(r2)
            all_record_ids_in_pairs = limited_ids
        else:
            print(
                f"ペア数({len(pairs_to_evaluate)})が指定された上限({args.limit})より"
                "少ないため、制限は適用されません。"
            )

    all_record_ids_list = sorted(list(all_record_ids_in_pairs))

    # データタイプに応じたプロンプト情報を取得
    data_type_info = get_prompts(args.data_type)

    # Few-shotデータの読み込み
    few_shot_system_prompt, few_shot_examples = (
        load_few_shot_prompt_and_examples(
            args.few_shot_data, max_examples=args.max_fewshot_examples
        )
    )
    
    print("\n===== Zero-shot vs Few-shot モデル性能比較評価 =====")
    print(f"評価モデル: {args.model_id}")
    if few_shot_examples:
        print(f"Few-shotデータ: {args.few_shot_data}")
    print(f"最大同時リクエスト数: {MAX_CONCURRENT_REQUESTS}")
    print(f"1分間の最大リクエスト数: {REQUESTS_PER_MINUTE}")
    
    # Zero-shot評価
    print("\n===== Zero-shot性能評価 =====")
    # Zero-shot用の基本プロンプトを動的に生成
    info_name = data_type_info['info']
    entity_name = data_type_info['entity']
    zeroshot_system_prompt = (
        f"あなたは2つの{info_name}が実質的に同一の{entity_name}を指すか"
        "どうかを判断する専門家です。\n"
        f"まず、2つの{info_name}が同一の{entity_name}と思われる場合は「はい」、"
        "そうでない場合は「いいえ」で明確に回答してください。\n"
        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から"
        "1.0（完全に同一）の範囲で提示してください。\n"
        "あなたの判断は次のルールに厳密に従う必要があります：\n"
        " - 類似度スコアが0.5以上の場合、回答は必ず「はい」にしてください。\n"
        " - 類似度スコアが0.5未満の場合、回答は必ず「いいえ」にしてください。"
    )
    print("--- 使用するZero-shotシステムプロンプト ---")
    print(zeroshot_system_prompt)
    print("----------------------------------------")

    start_time_zeroshot = time.time()
    results_zeroshot = await evaluate_model_on_pairs_async(
        args.model_id, pairs_to_evaluate, all_record_ids_in_pairs, api_key,
        system_prompt=zeroshot_system_prompt,
        few_shot_examples=None, no_cache=args.no_cache,
        data_type=args.data_type
    )
    end_time_zeroshot = time.time()
    
    pairwise_metrics_zeroshot = calculate_pairwise_metrics(
        results_zeroshot["ground_truths"], 
        results_zeroshot["predictions"], 
        f"{args.model_id} (Zero-shot)"
    )
    
    pred_clusters_zeroshot = form_predicted_clusters(
        results_zeroshot["predicted_positive_pairs"], 
        all_record_ids_list
    )
    
    clustering_metrics_zeroshot = calculate_clustering_metrics(
        GROUND_TRUTH_CLUSTERS, 
        pred_clusters_zeroshot, 
        all_record_ids_list, 
        f"{args.model_id} (Zero-shot)"
    )
    
    # Few-shot評価
    results_fewshot = None
    pairwise_metrics_fewshot = {}
    clustering_metrics_fewshot = {}
    end_time_fewshot = start_time_zeroshot  # 初期化
    if few_shot_examples:
        print("\n===== Few-shot性能評価 =====")
        print("--- 使用するFew-shotシステムプロンプト ---")
        print(few_shot_system_prompt)
        print("---------------------------------------")
        start_time_fewshot = time.time()
        results_fewshot = await evaluate_model_on_pairs_async(
            args.model_id, pairs_to_evaluate, all_record_ids_in_pairs, api_key,
            system_prompt=few_shot_system_prompt,
            few_shot_examples=few_shot_examples, no_cache=args.no_cache,
            data_type=args.data_type
        )
        end_time_fewshot = time.time()
    
        pairwise_metrics_fewshot = calculate_pairwise_metrics(
            results_fewshot["ground_truths"], 
            results_fewshot["predictions"], 
            f"{args.model_id} (Few-shot)"
        )
    
        pred_clusters_fewshot = form_predicted_clusters(
            results_fewshot["predicted_positive_pairs"], 
            all_record_ids_list
        )
    
        clustering_metrics_fewshot = calculate_clustering_metrics(
            GROUND_TRUTH_CLUSTERS, 
            pred_clusters_fewshot, 
            all_record_ids_list, 
            f"{args.model_id} (Few-shot)"
        )

    # 全ペア推論評価
    print("\n===== 全ペア推論評価 =====")
    all_record_ids_global = sorted(list(BIB_DATA.keys()))
    
    if len(all_record_ids_global) >= 2:
        pred_clusters_all_scope_zeroshot = form_predicted_clusters(
            results_zeroshot["predicted_positive_pairs"], all_record_ids_global
        )
        if results_fewshot:
            pred_clusters_all_scope_fewshot = form_predicted_clusters(
                results_fewshot["predicted_positive_pairs"],
                all_record_ids_global
            )

        all_pairs_true_labels = []
        all_pairs_pred_labels_zeroshot = []
        all_pairs_pred_labels_fewshot = []
        
        num_total_pairs = (
            len(all_record_ids_global) * (len(all_record_ids_global) - 1) // 2
        )
        print(
            f"全ペア推論評価: {len(all_record_ids_global)}C2 = {num_total_pairs} "
            "ペアのラベルを生成中..."
        )
        
        # 全ペア推論用の進捗バー（改行せずに更新）
        pair_combinations = list(
            itertools.combinations(all_record_ids_global, 2)
        )
        for r_id1, r_id2 in tqdm(
            pair_combinations, desc="全ペア推論", unit="ペア", leave=True, ncols=100
        ):
            # 正解ラベル
            gt_c1 = GROUND_TRUTH_CLUSTERS.get(str(r_id1))
            gt_c2 = GROUND_TRUTH_CLUSTERS.get(str(r_id2))
            is_truly_similar = (
                gt_c1 is not None and gt_c2 is not None and 
                not str(gt_c1).startswith("gt_orphan_") and 
                not str(gt_c2).startswith("gt_orphan_") and 
                str(gt_c1) == str(gt_c2)
            )
            all_pairs_true_labels.append(is_truly_similar)
            
            # Zero-shotモデルの予測ラベル
            pred_c1_z = pred_clusters_all_scope_zeroshot.get(str(r_id1))
            pred_c2_z = pred_clusters_all_scope_zeroshot.get(str(r_id2))
            all_pairs_pred_labels_zeroshot.append(
                pred_c1_z is not None and pred_c1_z == pred_c2_z
            )
            
            # Few-shotモデルの予測ラベル
            if results_fewshot:
                pred_c1_f = pred_clusters_all_scope_fewshot.get(str(r_id1))
                pred_c2_f = pred_clusters_all_scope_fewshot.get(str(r_id2))
                all_pairs_pred_labels_fewshot.append(
                    pred_c1_f is not None and pred_c1_f == pred_c2_f
                )
        
        pairwise_metrics_all_zeroshot = calculate_pairwise_metrics(
            all_pairs_true_labels,
            all_pairs_pred_labels_zeroshot,
            f"{args.model_id} (Zero-shot, 全ペア推論)"
        )
        if results_fewshot:
            pairwise_metrics_all_fewshot = calculate_pairwise_metrics(
                all_pairs_true_labels,
                all_pairs_pred_labels_fewshot,
                f"{args.model_id} (Few-shot, 全ペア推論)"
            )
    else:
        pairwise_metrics_all_zeroshot = {
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "precision": 0, "recall": 0, "f1_score": 0
        }
    if not results_fewshot:
        pairwise_metrics_all_fewshot = {
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "precision": 0, "recall": 0, "f1_score": 0
        }

    # 詳細結果のCSV作成
    results_df_data = []
    for i, (r_id1, r_id2) in enumerate(results_zeroshot["processed_pairs"]):
        row = {
            "record_id_1": r_id1,
            "record_id_2": r_id2,
            "ground_truth_similar": results_zeroshot["ground_truths"][i],
            "predicted_similar_zeroshot": results_zeroshot["predictions"][i],
            "score_zeroshot": results_zeroshot["llm_scores"][i],
            "error_zeroshot": next((
                err[1] for pair_ids, err in results_zeroshot["errors"]
                if pair_ids == (r_id1, r_id2)
            ), None),
        }
        if results_fewshot and i < len(results_fewshot["predictions"]):
            row["predicted_similar_fewshot"] = (
                results_fewshot["predictions"][i]
            )
            row["score_fewshot"] = results_fewshot["llm_scores"][i]
            row["error_fewshot"] = next((
                err[1] for pair_ids, err in results_fewshot["errors"]
                if pair_ids == (r_id1, r_id2)
            ), None)
        else:
            row["predicted_similar_fewshot"] = None
            row["score_fewshot"] = None
            row["error_fewshot"] = None
        results_df_data.append(row)
    
    # --- 出力先ディレクトリの準備 ---
    pairs_csv_path = args.pairs_csv
    # CSVファイルがあるディレクトリの親階層に "evaluation_results" を作成
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(pairs_csv_path)), "evaluation_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n結果は次のディレクトリに保存されます: {output_dir}")

    # --- ファイル名の生成 ---
    model_sanitized = sanitize_model_name_for_filename(args.model_id)
    pairs_basename = os.path.basename(args.pairs_csv).replace('.csv', '')
    base_filename = (
        f"eval_{pairs_basename}_model-{model_sanitized}"
    )

    # few-shotデータファイル名を出力ファイル名に含める
    if few_shot_examples and args.few_shot_data:
        fewshot_basename = os.path.basename(
            args.few_shot_data
        ).replace('.json', '')
        base_filename += f"_fewshot-{fewshot_basename}"

    # --- 詳細結果の保存 ---
    detailed_csv_filename = os.path.join(
        output_dir, f"{base_filename}_details.csv"
    )
    detailed_results_df = pd.DataFrame(results_df_data)
    detailed_results_df.to_csv(
        detailed_csv_filename, index=False, encoding="utf-8-sig"
    )
    print(f"\n詳細な評価結果を {detailed_csv_filename} に保存しました。")

    # --- パフォーマンスレポートの生成 ---
    avg_time_zeroshot = (
        (end_time_zeroshot - start_time_zeroshot) / len(pairs_to_evaluate)
    )

    report_content = f"""# Zero-shot vs Few-shot 性能評価レポート
日付: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 評価対象
- モデル: {args.model_id}
- データ: {args.ground_truth_yaml}
- ペアリスト: {args.pairs_csv} ({len(pairs_to_evaluate)} ペア)
- 全レコード数: {len(all_record_ids_global)}\n"""
    if few_shot_examples:
        report_content += f"- Few-shotデータ: {args.few_shot_data}\n"

    report_content += (
        "\n## 処理時間\n"
        f"- Zero-shot評価時間: {end_time_zeroshot - start_time_zeroshot:.2f}秒\n"
        f"- 平均処理時間（Zero-shot）: {avg_time_zeroshot:.3f}秒/ペア\n"
    )
    if few_shot_examples:
        avg_time_fewshot = (
            (end_time_fewshot - start_time_fewshot) / len(pairs_to_evaluate)
        )
        report_content += (
            f"- Few-shot評価時間: {end_time_fewshot - start_time_fewshot:.2f}秒\n"
            f"- 平均処理時間（Few-shot）: {avg_time_fewshot:.3f}秒/ペア\n"
        )

    report_content += "\n## K近傍ペア評価\n"
    # Zero-shotレポート
    report_content += f"### Zero-shotモデル ({args.model_id})\n"
    report_content += "- 混合行列:\n"
    report_content += (
        "    予測ラベル     |  Predicted: Positive | Predicted: Negative\n"
    )
    report_content += (
        "  ----------------|----------------------|----------------------\n"
    )
    tp = pairwise_metrics_zeroshot.get('tp', 0)
    fn = pairwise_metrics_zeroshot.get('fn', 0)
    fp = pairwise_metrics_zeroshot.get('fp', 0)
    tn = pairwise_metrics_zeroshot.get('tn', 0)
    report_content += f"  Actual: Positive  | TP: {tp:<18d} | FN: {fn:<18d}\n"
    report_content += f"  Actual: Negative  | FP: {fp:<18d} | TN: {tn:<18d}\n"
    precision = pairwise_metrics_zeroshot.get('precision', 0)
    recall = pairwise_metrics_zeroshot.get('recall', 0)
    f1 = pairwise_metrics_zeroshot.get('f1_score', 0)
    report_content += (
        f"- 適合率: {precision:.4f}, 再現率: {recall:.4f}, F1: {f1:.4f}\n"
    )
    report_content += f"- エラー数: {len(results_zeroshot['errors'])}\n"

    # Zero-shotクラスタリングレポート
    ari = clustering_metrics_zeroshot.get('ari', 0)
    nmi = clustering_metrics_zeroshot.get('nmi', 0)
    homogeneity = clustering_metrics_zeroshot.get('homogeneity', 0)
    completeness = clustering_metrics_zeroshot.get('completeness', 0)
    v_measure = clustering_metrics_zeroshot.get('v_measure', 0)
    report_content += "\n### クラスタリング評価（Zero-shot）\n"
    report_content += f"- ARI: {ari:.4f}\n"
    report_content += f"- NMI: {nmi:.4f}\n"
    report_content += (
        f"- Homogeneity: {homogeneity:.4f}, Completeness: {completeness:.4f}, "
        f"V-measure: {v_measure:.4f}\n"
    )

    if few_shot_examples:
        # Few-shotレポート
        report_content += f"\n### Few-shotモデル ({args.model_id})\n"
        report_content += "- 混合行列:\n"
        report_content += (
            "    予測ラベル     |  Predicted: Positive | Predicted: Negative\n"
        )
        report_content += (
            "  ----------------|----------------------|----------------------\n"
        )
        tp = pairwise_metrics_fewshot.get('tp', 0)
        fn = pairwise_metrics_fewshot.get('fn', 0)
        fp = pairwise_metrics_fewshot.get('fp', 0)
        tn = pairwise_metrics_fewshot.get('tn', 0)
        report_content += f"  Actual: Positive  | TP: {tp:<18d} | FN: {fn:<18d}\n"
        report_content += f"  Actual: Negative  | FP: {fp:<18d} | TN: {tn:<18d}\n"
        precision = pairwise_metrics_fewshot.get('precision', 0)
        recall = pairwise_metrics_fewshot.get('recall', 0)
        f1 = pairwise_metrics_fewshot.get('f1_score', 0)
        report_content += (
            f"- 適合率: {precision:.4f}, 再現率: {recall:.4f}, F1: {f1:.4f}\n"
        )
        report_content += f"- エラー数: {len(results_fewshot['errors'])}\n"

        # Few-shotクラスタリングレポート
        ari = clustering_metrics_fewshot.get('ari', 0)
        nmi = clustering_metrics_fewshot.get('nmi', 0)
        homogeneity = clustering_metrics_fewshot.get('homogeneity', 0)
        completeness = clustering_metrics_fewshot.get('completeness', 0)
        v_measure = clustering_metrics_fewshot.get('v_measure', 0)
        report_content += "\n### クラスタリング評価（Few-shot）\n"
        report_content += f"- ARI: {ari:.4f}\n"
        report_content += f"- NMI: {nmi:.4f}\n"
        report_content += (
            f"- Homogeneity: {homogeneity:.4f}, "
            f"Completeness: {completeness:.4f}, V-measure: {v_measure:.4f}\n"
        )

    report_content += "\n## 全ペア推論評価\n"

    # 全ペア Zero-shot レポート
    report_content += f"### Zero-shotモデル ({args.model_id})\n"
    report_content += "- 混合行列:\n"
    report_content += (
        "    予測ラベル     |  Predicted: Positive | Predicted: Negative\n"
    )
    report_content += (
        "  ----------------|----------------------|----------------------\n"
    )
    tp = pairwise_metrics_all_zeroshot.get('tp', 0)
    fn = pairwise_metrics_all_zeroshot.get('fn', 0)
    fp = pairwise_metrics_all_zeroshot.get('fp', 0)
    tn = pairwise_metrics_all_zeroshot.get('tn', 0)
    report_content += f"  Actual: Positive  | TP: {tp:<18d} | FN: {fn:<18d}\n"
    report_content += f"  Actual: Negative  | FP: {fp:<18d} | TN: {tn:<18d}\n"
    precision = pairwise_metrics_all_zeroshot.get('precision', 0)
    recall = pairwise_metrics_all_zeroshot.get('recall', 0)
    f1 = pairwise_metrics_all_zeroshot.get('f1_score', 0)
    report_content += (
        f"- 適合率: {precision:.4f}, 再現率: {recall:.4f}, F1: {f1:.4f}\n"
    )


    if few_shot_examples:
        f1_improvement = (pairwise_metrics_fewshot.get('f1_score', 0) -
                          pairwise_metrics_zeroshot.get('f1_score', 0))
        ari_imp = (clustering_metrics_fewshot.get('ari', 0) -
                   clustering_metrics_zeroshot.get('ari', 0))
        all_f1_imp = (pairwise_metrics_all_fewshot.get('f1_score', 0) -
                      pairwise_metrics_all_zeroshot.get('f1_score', 0))
        
        # 全ペア Few-shot レポート
        report_content += f"\n### Few-shotモデル ({args.model_id})\n"
        report_content += "- 混合行列:\n"
        report_content += (
            "    予測ラベル     |  Predicted: Positive | Predicted: Negative\n"
        )
        report_content += (
            "  ----------------|----------------------|----------------------\n"
        )
        tp = pairwise_metrics_all_fewshot.get('tp', 0)
        fn = pairwise_metrics_all_fewshot.get('fn', 0)
        fp = pairwise_metrics_all_fewshot.get('fp', 0)
        tn = pairwise_metrics_all_fewshot.get('tn', 0)
        report_content += f"  Actual: Positive  | TP: {tp:<18d} | FN: {fn:<18d}\n"
        report_content += f"  Actual: Negative  | FP: {fp:<18d} | TN: {tn:<18d}\n"
        precision = pairwise_metrics_all_fewshot.get('precision', 0)
        recall = pairwise_metrics_all_fewshot.get('recall', 0)
        f1 = pairwise_metrics_all_fewshot.get('f1_score', 0)
        report_content += (
            f"- 適合率: {precision:.4f}, 再現率: {recall:.4f}, F1: {f1:.4f}\n"
        )

        report_content += "\n## 改善度\n"
        report_content += f"- F1スコア改善: {f1_improvement:+.4f}\n"
        report_content += f"- ARI改善: {ari_imp:+.4f}\n"
        report_content += f"- 全ペアF1改善: {all_f1_imp:+.4f}\n"
    
    # --- レポートの保存 ---
    report_filename = os.path.join(output_dir, f"{base_filename}_report.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"パフォーマンスレポートを {report_filename} に保存しました。")

    # --- クラスタ詳細の保存 ---
    try:
        formatted_clusters_zeroshot = format_clusters_with_details(
            pred_clusters_zeroshot, BIB_DATA
        )
        filename_zeroshot_detailed = os.path.join(
            output_dir, f"{base_filename}_clusters_zeroshot.json"
        )
        with open(filename_zeroshot_detailed, "w", encoding="utf-8") as f:
            json.dump(
                formatted_clusters_zeroshot, f, ensure_ascii=False, indent=4
            )
        print(f"Zero-shot予測クラスタ詳細を {filename_zeroshot_detailed} に保存しました。")
        
        if few_shot_examples:
            formatted_clusters_fewshot = format_clusters_with_details(
                pred_clusters_fewshot, BIB_DATA
            )
            filename_fewshot_detailed = os.path.join(
                output_dir, f"{base_filename}_clusters_fewshot.json"
            )
            with open(filename_fewshot_detailed, "w", encoding="utf-8") as f:
                json.dump(
                    formatted_clusters_fewshot, f, ensure_ascii=False, indent=4
                )
            print(f"Few-shot予測クラスタ詳細を {filename_fewshot_detailed} に保存しました。")
    except Exception as e:
        print(f"エラー: 詳細な予測クラスタ情報のJSON保存に失敗: {e}")
    
    # 最終的にキャッシュを保存
    if not args.no_cache:
        save_cache()
    
    print("\n===== 評価完了 =====")
    total_time = (end_time_zeroshot - start_time_zeroshot)
    if few_shot_examples:
        total_time += (end_time_fewshot - start_time_fewshot)
    print(f"総処理時間: {total_time:.2f}秒")
    if few_shot_examples:
        f1_improvement = (
            pairwise_metrics_fewshot.get('f1_score', 0) -
            pairwise_metrics_zeroshot.get('f1_score', 0)
        )
        print(f"F1スコア改善: {f1_improvement:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shotとFew-shotのLLM性能を非同期で比較評価するスクリプト"
    )
    parser.add_argument(
        "--pairs_csv", required=True, help="評価ペアのCSVファイルパス"
    )
    parser.add_argument(
        "--ground_truth_yaml", required=True, help="正解データのYAMLファイルパス"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["bib", "music", "person", "walmart_amazon_product", "wdc_product"],
        help="データの種類 (プロンプト生成に利用)"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=DEFAULT_MODEL_ID,
        help=f"評価対象のモデルID。デフォルト: {DEFAULT_MODEL_ID}"
    )
    parser.add_argument(
        "--few_shot_data",
        type=str,
        default=None,
        help=(
            "Few-shot学習用のJSONファイルパス。"
            "指定しない場合はZero-shotのみ評価。"
        )
    )
    parser.add_argument(
        '--max_fewshot_examples',
        type=int,
        default=None,
        help='Few-shot学習で使用するお手本の最大数。指定しない場合はすべて使用します。'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='キャッシュを無効にして実行します。'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='評価するペアの最大数を制限します。テスト用に最初のNペアのみを処理します。'
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=20, help="最大同時リクエスト数"
    )
    parser.add_argument(
        "--requests_per_minute",
        type=int,
        default=3000,
        help="1分間の最大リクエスト数"
    )
    
    args = parser.parse_args()
    
    # グローバル設定を更新
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    REQUESTS_PER_MINUTE = args.requests_per_minute
    REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE
    
    # 非同期実行
    asyncio.run(main(args))