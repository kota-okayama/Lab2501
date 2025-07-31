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
CACHE_DATA = {}  # {cache_key: {"is_similar": bool, "score": float}}
CACHE_FILE = "llm_evaluation_cache.pkl"

# APIレート制限パラメータ
MAX_CONCURRENT_REQUESTS = 20  # 同時リクエスト最大数
REQUESTS_PER_MINUTE = 3000    # 1分間の最大リクエスト数
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE  # リクエスト間の最小間隔

# デフォルト設定
DEFAULT_MODEL_ID_BEFORE_FINETUNING = "gpt-4o-mini"
DEFAULT_MODEL_ID_AFTER_FINETUNING = "ft:gpt-4o-mini-2024-07-18:your-org:model-name:suffix"


def load_cache():
    """キャッシュファイルを読み込む（PickleとJSONの両方をサポート）"""
    global CACHE_DATA
    CACHE_DATA = {}
    
    # 既存のJSONキャッシュを読み込み
    json_cache_path = "openai_embedding_experiment/evaluation_results/llm_api_cache.json"
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
    """キャッシュをファイルに保存する"""
    global CACHE_DATA
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(CACHE_DATA, f)
        
        message = f"キャッシュを保存しました: {len(CACHE_DATA)} 件"
        if pbar:
            pbar.write(message)
    except Exception as e:
        error_message = f"キャッシュファイルの保存に失敗: {e}"
        if pbar:
            pbar.write(error_message)
        else:
            print(error_message)


def load_bib_data_and_gt_clusters(yaml_path):
    """
    YAMLファイルから書誌データと正解クラスタ情報を読み込む
    """
    global BIB_DATA, GROUND_TRUTH_CLUSTERS
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAMLファイルが見つかりません: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
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
                    GROUND_TRUTH_CLUSTERS[record_id] = f"gt_orphan_{orphan_counter}"
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
                    GROUND_TRUTH_CLUSTERS[record_id] = f"gt_orphan_{orphan_counter}"
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
        header = next(reader, None)  # ヘッダー行を読み飛ばす
        
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
    レコードIDから書誌情報を取得してプロンプト形式で返す（動的フィールド対応）
    """
    global BIB_DATA
    
    if record_id not in BIB_DATA:
        # このメッセージは、load_bib_data_and_gt_clustersが正しく動作していれば通常表示されないはず
        return f"レコードID {record_id} の書誌情報が内部データ構造(BIB_DATA)に見つかりません"
    
    record = BIB_DATA[record_id]
    details = []
    
    # レコードのすべてのキーと値をループ処理
    for field, value in record.items():
        # 'record_id'フィールドは表示しない
        if field == 'record_id':
            continue
        
        # 値が文字列で、かつ空でない場合にのみ追加
        if isinstance(value, str) and value.strip():
            # 'bib1_'プレフィックスを削除して整形（存在する場合）
            field_name = field.replace('bib1_', '').title()
            details.append(f"{field_name}: {value.strip()}")

    if not details:
        # BIB_DATAには存在するが、中身が空の場合
        return f"レコードID {record_id} には表示可能な書誌情報フィールドがありません"
    
    return f"Record ID: {record_id}\n" + "\n".join(details)


def get_prompts(data_type):
    """データタイプに応じたプロンプトを返す"""
    if data_type == "bib":
        system_prompt = (
            "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\n"
            "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。\n"
            "あなたの判断は次のルールに厳密に従う必要があります：\n"
            " - 類似度スコアが0.5以上の場合、回答は必ず「はい」にしてください。\n"
            " - 類似度スコアが0.5未満の場合、回答は必ず「いいえ」にしてください。\n"
        )
        user_prompt_template = (
            "以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\n\n"
            "情報1:\n{info_1}\n\n"
            "情報2:\n{info_2}\n\n"
            "これらは同一の文献ですか？\n回答:"
        )
    elif data_type == "music":
        system_prompt = (
            "あなたは2つの楽曲情報が実質的に同一の楽曲を指すかどうかを判断する専門家です。\n"
            "まず、2つの楽曲情報が同一の楽曲と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。\n"
            "あなたの判断は次のルールに厳密に従う必要があります：\n"
            " - 類似度スコアが0.5以上の場合、回答は必ず「はい」にしてください。\n"
            " - 類似度スコアが0.5未満の場合、回答は必ず「いいえ」にしてください。\n"
        )
        user_prompt_template = (
            "以下の2つの楽曲情報が、実質的に同一の楽曲を指しているかどうかを判断してください。\n\n"
            "情報1:\n{info_1}\n\n"
            "情報2:\n{info_2}\n\n"
            "これらは同一の楽曲ですか？\n回答:"
        )
    elif data_type == "person":
        system_prompt = (
            "あなたは2つの人物情報が実質的に同一の人物を指すかどうかを判断する専門家です。\n"
            "まず、2つの人物情報が同一の人物と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。\n"
            "あなたの判断は次のルールに厳密に従う必要があります：\n"
            " - 類似度スコアが0.5以上の場合、回答は必ず「はい」にしてください。\n"
            " - 類似度スコアが0.5未満の場合、回答は必ず「いいえ」にしてください。\n"
        )
        user_prompt_template = (
            "以下の2つの人物情報が、実質的に同一の人物を指しているかどうかを判断してください。\n\n"
            "情報1:\n{info_1}\n\n"
            "情報2:\n{info_2}\n\n"
            "これらは同一の人物ですか？\n回答:"
        )
    else:
        raise ValueError(f"未知のデータタイプです: {data_type}")

    return system_prompt, user_prompt_template


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
            self.request_times = [t for t in self.request_times if now - t < 60.0]
            
            if len(self.request_times) >= self.max_requests_per_minute:
                # レート制限に達している場合は待機
                oldest_request = min(self.request_times)
                sleep_time = 60.0 - (now - oldest_request) + 0.1  # 少し余裕を持たせる
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.time())


async def get_llm_evaluation_for_pair_async(client, record_id_1, record_id_2, model_id, rate_limiter, data_type):
    """
    非同期でLLM評価を実行する関数 (openaiライブラリ使用)
    """
    global CACHE_DATA
    # data_typeが'bib'の場合はキーに含めず、それ以外は含める
    if data_type == 'bib':
        cache_key = f"{record_id_1}_{record_id_2}_{model_id}"
    else:
        cache_key = f"{record_id_1}_{record_id_2}_{model_id}_{data_type}"
    
    # キャッシュチェック
    if cache_key in CACHE_DATA:
        cached_item = CACHE_DATA[cache_key]
        if isinstance(cached_item, dict) and "is_similar" in cached_item and "score" in cached_item:
            return cached_item["is_similar"], cached_item["score"], None
    
    # 書誌情報取得
    info_1 = get_record_details_for_prompt(record_id_1)
    info_2 = get_record_details_for_prompt(record_id_2)
    
    if "見つかりません" in info_1 or "フィールドがありません" in info_1:
        return None, None, f"レコード {record_id_1} の情報取得に失敗: {info_1}"
    if "見つかりません" in info_2 or "フィールドがありません" in info_2:
        return None, None, f"レコード {record_id_2} の情報取得に失敗: {info_2}"
    
    system_prompt, user_prompt_template = get_prompts(data_type)
    user_prompt = user_prompt_template.format(info_1=info_1, info_2=info_2)
    
    try:
        # レート制限を適用
        await rate_limiter.acquire()
        
        # OpenAI API呼び出し（公式ライブラリ使用）
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
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
                    print(f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコア '{parsed_similarity_score}' が範囲外です。")
            except ValueError:
                print(f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコアが数値に変換できません: '{similarity_score_str}'")
        
        # フォールバック処理
        if parsed_is_similar is None and parsed_similarity_score is not None:
            parsed_is_similar = parsed_similarity_score >= 0.5
        elif parsed_is_similar is not None and parsed_similarity_score is None:
            parsed_similarity_score = 1.0 if parsed_is_similar else 0.0
        
        if parsed_is_similar is None:
            return None, None, f"LLMの応答から判定を抽出できませんでした。応答: {response_text}"
        if parsed_similarity_score is None:
            return parsed_is_similar, None, f"LLMの応答から類似度スコアを抽出できませんでした。応答: {response_text}"
        
        # キャッシュに保存
        CACHE_DATA[cache_key] = {"is_similar": parsed_is_similar, "score": parsed_similarity_score}
        
        return parsed_is_similar, parsed_similarity_score, None
        
    except Exception as e:
        # openai.APIErrorから詳細なエラー情報を取得
        if hasattr(e, 'message'):
            error_details = e.message
        elif hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_details = e.response.text
        else:
            error_details = str(e)
            
        error_msg = f"API呼び出し中にエラーが発生 (ペア: {record_id_1}, {record_id_2}): {error_details}"
        return None, None, error_msg


async def evaluate_model_on_pairs_async(model_id, pairs_to_evaluate, all_record_ids_in_pairs, api_key, data_type):
    """
    非同期でモデル評価を実行する関数
    """
    predictions = []
    ground_truths = []
    predicted_positive_pairs = []
    llm_scores = []
    errors = []
    processed_pairs = []
    
    print(f"\nモデル '{model_id}' で {len(pairs_to_evaluate)} ペアの非同期評価を開始します...")
    
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
            
            is_truly_similar = False
            if (gt_cluster1 is not None and gt_cluster2 is not None and 
                not gt_cluster1.startswith("gt_orphan_") and 
                not gt_cluster2.startswith("gt_orphan_") and 
                gt_cluster1 == gt_cluster2):
                is_truly_similar = True
            
            # LLM評価実行
            llm_is_similar, llm_score, error_msg = await get_llm_evaluation_for_pair_async(
                client, r_id1, r_id2, model_id, rate_limiter, data_type
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
    pair_info_list = [(pair[0], pair[1], i) for i, pair in enumerate(pairs_to_evaluate)]
    
    # バッチサイズを設定してメモリ使用量を制御
    batch_size = MAX_CONCURRENT_REQUESTS * 2
    
    # 進捗バーを初期化（改行せずに更新）
    with tqdm(total=len(pair_info_list), desc=f"評価中 ({model_id})", unit="ペア", leave=True, ncols=100) as pbar:
        for i in range(0, len(pair_info_list), batch_size):
            batch = pair_info_list[i:i + batch_size]
            
            # バッチを並列実行
            tasks = [evaluate_single_pair(pair_info) for pair_info in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
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
                pbar.set_description(f"評価中 ({model_id}) [タスクエラー: {batch_task_errors}件]")
            if len(errors) > 0:
                pbar.set_postfix_str(f"APIエラー: {len(errors)}件")
            
            # 進捗保存
            if i % (batch_size * 3) == 0:  # 3バッチごとにキャッシュ保存
                save_cache(pbar)
    
    print(f"モデル '{model_id}' での非同期評価完了。エラー: {len(errors)}件。")
    
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
        if label_val == False:
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
    print(f"  | {'Ground Truth':^14} | {'Predicted: Pos':^15} | {'Predicted: Neg':^15} |")
    print("  +================+=================+=================+")
    print(f"  | {'Positive':<14} | TP: {tp:<12d} | FN: {fn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  | {'Negative':<14} | FP: {fp:<12d} | TN: {tn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  適合率 (Precision): {precision:.4f} (TP / (TP + FP))")
    print(f"  再現率 (Recall):    {recall:.4f} (TP / (TP + FN))")
    print(f"  F1スコア:           {f1:.4f}")
    
    return {"precision": precision, "recall": recall, "f1_score": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


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


def calculate_clustering_metrics(true_cluster_map, pred_cluster_map, all_record_ids, model_name=""):
    """クラスタリング評価指標を計算する"""
    true_labels = [true_cluster_map.get(rid, f"missing_gt_{rid}") for rid in all_record_ids]
    pred_labels = [pred_cluster_map.get(rid, -1) for rid in all_record_ids]
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)
    
    print(f"\n--- {model_name} クラスタリング評価指標 ---")
    print(f"  調整ランド指数 (ARI): {ari:.4f}")
    print(f"  正規化相互情報量 (NMI): {nmi:.4f}")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Completeness: {completeness:.4f}")
    print(f"  V-measure: {v_measure:.4f}")
    
    return {"ari": ari, "nmi": nmi, "homogeneity": homogeneity, "completeness": completeness, "v_measure": v_measure}


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
        grouped_clusters[cluster_label].append({"record_id": record_id, "details": details_str})
    
    sorted_grouped_clusters = {}
    for label in sorted(grouped_clusters.keys()):
        sorted_records = sorted(grouped_clusters[label], key=lambda x: x["record_id"])
        sorted_grouped_clusters[str(label)] = sorted_records
    return sorted_grouped_clusters


async def main(args):
    # OpenAI APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY 環境変数が設定されていません。")
        return
    
    # キャッシュと書誌データの読み込み
    load_cache()
    load_bib_data_and_gt_clusters(args.ground_truth_yaml)
    
    # 評価ペアの読み込み
    pairs_to_evaluate, all_record_ids_in_pairs = load_evaluation_pairs(args.pairs_csv)
    all_record_ids_list = sorted(list(all_record_ids_in_pairs))
    
    print("\n===== ファインチューニング前後のモデル性能比較評価 =====")
    print(f"ファインチューニング前のモデル: {args.model_before_ft}")
    print(f"ファインチューニング後のモデル: {args.model_after_ft}")
    print(f"最大同時リクエスト数: {MAX_CONCURRENT_REQUESTS}")
    print(f"1分間の最大リクエスト数: {REQUESTS_PER_MINUTE}")
    
    # ファインチューニング前のモデル評価
    print("\n===== ファインチューニング「前」のモデル性能評価 =====")
    start_time_before = time.time()
    results_before = await evaluate_model_on_pairs_async(
        args.model_before_ft, pairs_to_evaluate, all_record_ids_in_pairs, api_key, args.data_type
    )
    end_time_before = time.time()
    
    pairwise_metrics_before = calculate_pairwise_metrics(
        results_before["ground_truths"], 
        results_before["predictions"], 
        args.model_before_ft
    )
    
    pred_clusters_before = form_predicted_clusters(
        results_before["predicted_positive_pairs"], 
        all_record_ids_list
    )
    
    clustering_metrics_before = calculate_clustering_metrics(
        GROUND_TRUTH_CLUSTERS, 
        pred_clusters_before, 
        all_record_ids_list, 
        args.model_before_ft
    )
    
    # ファインチューニング後のモデル評価
    print("\n===== ファインチューニング「後」のモデル性能評価 =====")
    start_time_after = time.time()
    results_after = await evaluate_model_on_pairs_async(
        args.model_after_ft, pairs_to_evaluate, all_record_ids_in_pairs, api_key, args.data_type
    )
    end_time_after = time.time()
    
    pairwise_metrics_after = calculate_pairwise_metrics(
        results_after["ground_truths"], 
        results_after["predictions"], 
        args.model_after_ft
    )
    
    pred_clusters_after = form_predicted_clusters(
        results_after["predicted_positive_pairs"], 
        all_record_ids_list
    )
    
    clustering_metrics_after = calculate_clustering_metrics(
        GROUND_TRUTH_CLUSTERS, 
        pred_clusters_after, 
        all_record_ids_list, 
        args.model_after_ft
    )
    
    # 全ペア推論評価
    print("\n===== 全ペア推論評価 =====")
    all_record_ids_global = sorted(list(BIB_DATA.keys()))
    
    if len(all_record_ids_global) >= 2:
        pred_clusters_all_scope_before = form_predicted_clusters(
            results_before["predicted_positive_pairs"], all_record_ids_global
        )
        pred_clusters_all_scope_after = form_predicted_clusters(
            results_after["predicted_positive_pairs"], all_record_ids_global
        )
        
        all_pairs_true_labels = []
        all_pairs_pred_labels_before = []
        all_pairs_pred_labels_after = []
        
        num_total_pairs = len(all_record_ids_global) * (len(all_record_ids_global) - 1) // 2
        print(f"全ペア推論評価: {len(all_record_ids_global)}C2 = {num_total_pairs} ペアのラベルを生成中...")
        
        # 全ペア推論用の進捗バー（改行せずに更新）
        pair_combinations = list(itertools.combinations(all_record_ids_global, 2))
        for r_id1, r_id2 in tqdm(pair_combinations, desc="全ペア推論", unit="ペア", leave=True, ncols=100):
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
            
            # ファインチューニング前モデルの予測ラベル
            pred_c1_b = pred_clusters_all_scope_before.get(str(r_id1))
            pred_c2_b = pred_clusters_all_scope_before.get(str(r_id2))
            all_pairs_pred_labels_before.append(pred_c1_b is not None and pred_c1_b == pred_c2_b)
            
            # ファインチューニング後モデルの予測ラベル
            pred_c1_a = pred_clusters_all_scope_after.get(str(r_id1))
            pred_c2_a = pred_clusters_all_scope_after.get(str(r_id2))
            all_pairs_pred_labels_after.append(pred_c1_a is not None and pred_c1_a == pred_c2_a)
        
        pairwise_metrics_all_before = calculate_pairwise_metrics(
            all_pairs_true_labels, all_pairs_pred_labels_before, f"{args.model_before_ft} (全ペア推論)"
        )
        pairwise_metrics_all_after = calculate_pairwise_metrics(
            all_pairs_true_labels, all_pairs_pred_labels_after, f"{args.model_after_ft} (全ペア推論)"
        )
    else:
        pairwise_metrics_all_before = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "precision": 0, "recall": 0, "f1_score": 0}
        pairwise_metrics_all_after = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "precision": 0, "recall": 0, "f1_score": 0}
    
    # 詳細結果のCSV作成
    results_df_data = []
    for i, (r_id1, r_id2) in enumerate(results_before["processed_pairs"]):
        results_df_data.append({
            "record_id_1": r_id1,
            "record_id_2": r_id2,
            "ground_truth_similar": results_before["ground_truths"][i],
            "predicted_similar_before": results_before["predictions"][i],
            "score_before": results_before["llm_scores"][i],
            "error_before": next((err[1] for pair_ids, err in results_before["errors"] if pair_ids == (r_id1, r_id2)), None),
            "predicted_similar_after": results_after["predictions"][i] if i < len(results_after["predictions"]) else None,
            "score_after": results_after["llm_scores"][i] if i < len(results_after["llm_scores"]) else None,
            "error_after": next((err[1] for pair_ids, err in results_after["errors"] if pair_ids == (r_id1, r_id2)), None),
        })
    
    # --- 出力先ディレクトリの準備 ---
    pairs_csv_path = args.pairs_csv
    # CSVファイルがあるディレクトリの親階層に "evaluation_results" を作成
    output_dir = os.path.join(os.path.dirname(os.path.dirname(pairs_csv_path)), "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n結果は次のディレクトリに保存されます: {output_dir}")

    # --- ファイル名の生成 ---
    model_before_ft_sanitized = sanitize_model_name_for_filename(args.model_before_ft)
    model_after_ft_sanitized = sanitize_model_name_for_filename(args.model_after_ft)
    base_filename = f"eval_async_{os.path.basename(args.pairs_csv).replace('.csv', '')}_before-{model_before_ft_sanitized}_after-{model_after_ft_sanitized}"

    # --- 詳細結果の保存 ---
    detailed_csv_filename = os.path.join(output_dir, f"{base_filename}_details.csv")
    detailed_results_df = pd.DataFrame(results_df_data)
    detailed_results_df.to_csv(detailed_csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n詳細な評価結果を {detailed_csv_filename} に保存しました。")

    # --- パフォーマンスレポートの生成 ---
    report_content = f"""# ファインチューニング性能評価レポート（非同期版）
日付: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 評価対象
- 書誌データ: {args.ground_truth_yaml}
- K近傍ペアリスト: {args.pairs_csv} ({len(pairs_to_evaluate)} ペア)
- 全レコード数: {len(all_record_ids_global)}

## 処理時間
- ファインチューニング前モデル評価時間: {end_time_before - start_time_before:.2f}秒
- ファインチューニング後モデル評価時間: {end_time_after - start_time_after:.2f}秒
- 平均処理時間（前）: {(end_time_before - start_time_before) / len(pairs_to_evaluate):.3f}秒/ペア
- 平均処理時間（後）: {(end_time_after - start_time_after) / len(pairs_to_evaluate):.3f}秒/ペア

## K近傍ペア評価
### ファインチューニング前モデル ({args.model_before_ft})
- 混合行列:
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_before['tp']:<18d} | FN: {pairwise_metrics_before['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_before['fp']:<18d} | TN: {pairwise_metrics_before['tn']:<18d}
- 適合率: {pairwise_metrics_before['precision']:.4f}, 再現率: {pairwise_metrics_before['recall']:.4f}, F1: {pairwise_metrics_before['f1_score']:.4f}
- エラー数: {len(results_before['errors'])}

### クラスタリング評価（前）
- ARI: {clustering_metrics_before['ari']:.4f}
- NMI: {clustering_metrics_before['nmi']:.4f}
- Homogeneity: {clustering_metrics_before['homogeneity']:.4f}, Completeness: {clustering_metrics_before['completeness']:.4f}, V-measure: {clustering_metrics_before['v_measure']:.4f}

### ファインチューニング後モデル ({args.model_after_ft})
- 混合行列:
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_after['tp']:<18d} | FN: {pairwise_metrics_after['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_after['fp']:<18d} | TN: {pairwise_metrics_after['tn']:<18d}
- 適合率: {pairwise_metrics_after['precision']:.4f}, 再現率: {pairwise_metrics_after['recall']:.4f}, F1: {pairwise_metrics_after['f1_score']:.4f}
- エラー数: {len(results_after['errors'])}

### クラスタリング評価（後）
- ARI: {clustering_metrics_after['ari']:.4f}
- NMI: {clustering_metrics_after['nmi']:.4f}
- Homogeneity: {clustering_metrics_after['homogeneity']:.4f}, Completeness: {clustering_metrics_after['completeness']:.4f}, V-measure: {clustering_metrics_after['v_measure']:.4f}

## 全ペア推論評価
### ファインチューニング前モデル ({args.model_before_ft})
- 混合行列:
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_all_before['tp']:<18d} | FN: {pairwise_metrics_all_before['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_all_before['fp']:<18d} | TN: {pairwise_metrics_all_before['tn']:<18d}
- 適合率: {pairwise_metrics_all_before['precision']:.4f}, 再現率: {pairwise_metrics_all_before['recall']:.4f}, F1: {pairwise_metrics_all_before['f1_score']:.4f}

### ファインチューニング後モデル ({args.model_after_ft})
- 混合行列:
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_all_after['tp']:<18d} | FN: {pairwise_metrics_all_after['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_all_after['fp']:<18d} | TN: {pairwise_metrics_all_after['tn']:<18d}
- 適合率: {pairwise_metrics_all_after['precision']:.4f}, 再現率: {pairwise_metrics_all_after['recall']:.4f}, F1: {pairwise_metrics_all_after['f1_score']:.4f}

## 改善度
- F1スコア改善: {pairwise_metrics_after['f1_score'] - pairwise_metrics_before['f1_score']:+.4f}
- ARI改善: {clustering_metrics_after['ari'] - clustering_metrics_before['ari']:+.4f}
- 全ペアF1改善: {pairwise_metrics_all_after['f1_score'] - pairwise_metrics_all_before['f1_score']:+.4f}
"""
    
    # --- レポートの保存 ---
    report_filename = os.path.join(output_dir, f"{base_filename}_report.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"パフォーマンスレポートを {report_filename} に保存しました。")

    # --- クラスタ詳細の保存 ---
    try:
        formatted_clusters_before = format_clusters_with_details(pred_clusters_before, BIB_DATA)
        filename_before_detailed = os.path.join(output_dir, f"{base_filename}_clusters_before.json")
        with open(filename_before_detailed, "w", encoding="utf-8") as f:
            json.dump(formatted_clusters_before, f, ensure_ascii=False, indent=4)
        print(f"ファインチューニング前予測クラスタ詳細を {filename_before_detailed} に保存しました。")
        
        formatted_clusters_after = format_clusters_with_details(pred_clusters_after, BIB_DATA)
        filename_after_detailed = os.path.join(output_dir, f"{base_filename}_clusters_after.json")
        with open(filename_after_detailed, "w", encoding="utf-8") as f:
            json.dump(formatted_clusters_after, f, ensure_ascii=False, indent=4)
        print(f"ファインチューニング後予測クラスタ詳細を {filename_after_detailed} に保存しました。")
    except Exception as e:
        print(f"エラー: 詳細な予測クラスタ情報のJSON保存に失敗: {e}")
    
    # 最終的にキャッシュを保存
    save_cache()
    
    print("\n===== 評価完了 =====")
    print(f"総処理時間: {(end_time_after - start_time_before):.2f}秒")
    print(f"F1スコア改善: {pairwise_metrics_after['f1_score'] - pairwise_metrics_before['f1_score']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ファインチューニング前後のLLM性能を非同期で評価するスクリプト")
    parser.add_argument("--pairs_csv", required=True, help="評価ペアのCSVファイルパス")
    parser.add_argument("--ground_truth_yaml", required=True, help="正解データのYAMLファイルパス")
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["bib", "music", "person"],
        help="評価対象のデータの種類 (bib, music, person)"
    )
    parser.add_argument(
        "--model_before_ft", 
        type=str, 
        default=DEFAULT_MODEL_ID_BEFORE_FINETUNING,
        help=f"ファインチューニング前のモデルID。デフォルト: {DEFAULT_MODEL_ID_BEFORE_FINETUNING}"
    )
    parser.add_argument("--model_after_ft", required=True, help="ファインチューニング後のモデルID (必須)")
    parser.add_argument("--max_concurrent", type=int, default=20, help="最大同時リクエスト数")
    parser.add_argument("--requests_per_minute", type=int, default=3000, help="1分間の最大リクエスト数")
    
    args = parser.parse_args()
    
    # グローバル設定を更新
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    REQUESTS_PER_MINUTE = args.requests_per_minute
    REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE
    
    # 非同期実行
    asyncio.run(main(args))