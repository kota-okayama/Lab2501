import asyncio
# import aiohttp  <- 不要になったのでコメントアウト
import time
import json
import pickle
import yaml
import os
import argparse
import itertools
import networkx as nx
# import pandas as pd
from sklearn.metrics import (
    # precision_recall_fscore_support,
    # confusion_matrix,
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
DEFAULT_MODEL_ID_AFTER_FINETUNING = (
    "ft:gpt-4o-mini-2024-07-18:your-org:model-name:suffix"
)


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
                    gt_cluster_val = f"gt_orphan_{orphan_counter}"
                    GROUND_TRUTH_CLUSTERS[record_id] = gt_cluster_val
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
                    gt_cluster_val = f"gt_orphan_{orphan_counter}"
                    GROUND_TRUTH_CLUSTERS[record_id] = gt_cluster_val
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


def load_evaluation_candidates(json_path):
    """
    JSONファイルから評価用候補リストを読み込む
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSONファイルが見つかりません: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        candidate_sets = json.load(f)

    all_record_ids = set()
    # 候補セットが辞書であることを確認
    if not isinstance(candidate_sets, dict):
        raise TypeError(
            "JSONファイルはアンカーIDをキー、"
            "候補IDリストを値とする辞書形式である必要があります。"
        )
    for anchor_id, candidate_ids in candidate_sets.items():
        all_record_ids.add(anchor_id)
        # 候補リストがリスト形式であることを確認
        if isinstance(candidate_ids, list):
            all_record_ids.update(candidate_ids)
        else:
            print(
                f"警告: アンカー {anchor_id} の候補がリスト形式ではありません。"
                "スキップします。"
            )

    print(
        f"評価セットを読み込みました: {len(candidate_sets)} アンカー "
        f"({len(all_record_ids)} ユニークレコード)"
    )
    return candidate_sets, all_record_ids


def get_record_details_for_prompt(record_id):
    """
    レコードIDから書誌情報を取得してプロンプト形式で返す（動的フィールド対応）
    """
    global BIB_DATA
    
    if record_id not in BIB_DATA:
        # このメッセージは、load_bib_data_and_gt_clustersが正しく動作していれば
        # 通常表示されないはず
        return (
            f"レコードID {record_id} の書誌情報が内部データ構造(BIB_DATA)に"
            "見つかりません"
        )
    
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


def get_prompts(data_type, strategy="matching"):
    """データタイプと戦略に応じたプロンプンプトを返す"""
    if strategy == "selecting":
        system_prompt = (
            "あなたはエンティティマッチングを行うAIアシストです。"
            "あなたの唯一のタスクは、与えられたレコードと最も一致する候補をリストから選び、"
            "その番号を `[番号]` の形式で出力することです。\n"
            "**重要: いかなる説明や追加のテキストも生成してはなりません。**\n"
            "一致するものが複数ある場合は、それぞれをブラケットで囲んでください（例: `[1][5]`）。\n"
            "一致するものがない場合は `[0]` とだけ出力してください。\n"
            "あなたの応答は、必ず `[<number>]` という形式のみで構成されなければなりません。"
        )
        user_prompt_template = (
            "以下の指定されたレコードと、候補リスト中のレコードを比較してください。\n"
            "指定されたレコードと同一のエンティティを指すレコードを候補リストから**すべて**選び、"
            "その番号を `[番号]` 形式で答えてください。\n"
            "該当するものがない場合は `[0]` と回答してください。\n\n"
            "指定されたレコード:\n{anchor_record}\n\n"
            "候補リスト:\n{candidate_list}\n\n"
            "回答（`[番号]`形式のみ）:"
        )
        return system_prompt, user_prompt_template

    if data_type == "bib":
        system_prompt = (
            "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\n"
            "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、"
            "そうでない場合は「いいえ」で明確に回答してください。\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から"
            "1.0（完全に同一）の範囲で提示してください。\n"
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
            "まず、2つの楽曲情報が同一の楽曲と思われる場合は「はい」、"
            "そうでない場合は「いいえ」で明確に回答してください。\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から"
            "1.0（完全に同一）の範囲で提示してください。\n"
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
            "まず、2つの人物情報が同一の人物と思われる場合は「はい」、"
            "そうでない場合は「いいえ」で明確に回答してください。\n"
            "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から"
            "1.0（完全に同一）の範囲で提示してください。\n"
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
            one_minute_ago = now - 60.0
            self.request_times = [
                t for t in self.request_times if t > one_minute_ago
            ]
            
            if len(self.request_times) >= self.max_requests_per_minute:
                # レート制限に達している場合は待機
                oldest_request = min(self.request_times)
                sleep_time = 60.0 - (now - oldest_request) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.time())


async def get_llm_evaluation_for_selection_async(
    client, anchor_id, candidate_ids, model_id, rate_limiter, data_type
):
    """
    非同期でLLM評価を実行する関数 (selecting戦略)
    """
    global CACHE_DATA
    # sorting candidate_ids to ensure cache key is consistent
    sorted_candidate_ids_str = "_".join(sorted(candidate_ids))
    if data_type == 'bib':
        cache_key = f"select_{anchor_id}_{sorted_candidate_ids_str}_{model_id}"
    else:
        cache_key = (
            f"select_{anchor_id}_{sorted_candidate_ids_str}_{model_id}_"
            f"{data_type}"
        )
    
    # キャッシュチェック
    if cache_key in CACHE_DATA:
        cached_item = CACHE_DATA[cache_key]
        # selecting戦略のキャッシュ形式は { "selected_index": int } を想定
        if isinstance(cached_item, dict) and "selected_index" in cached_item:
            return [cached_item["selected_index"]], None
        # 新しいキャッシュ形式 { "selected_indices": [int] } をサポート
        if isinstance(cached_item, dict) and "selected_indices" in cached_item:
            return cached_item["selected_indices"], None

    # アンカーと候補の書誌情報を取得
    anchor_record_details = get_record_details_for_prompt(anchor_id)
    if ("見つかりません" in anchor_record_details or
            "フィールドがありません" in anchor_record_details):
        return None, f"アンカー {anchor_id} の情報取得に失敗: {anchor_record_details}"

    candidate_list_str = []
    for i, candidate_id in enumerate(candidate_ids):
        details = get_record_details_for_prompt(candidate_id)
        if "見つかりません" in details or "フィールドがありません" in details:
            return None, f"候補 {candidate_id} の情報取得に失敗: {details}"
        candidate_list_str.append(f"[{i+1}] {details}")
    
    candidate_list_formatted = "\n\n".join(candidate_list_str)

    system_prompt, user_prompt_template = get_prompts(
        data_type, strategy="selecting"
    )
    user_prompt = user_prompt_template.format(
        anchor_record=anchor_record_details,
        candidate_list=candidate_list_formatted
    )
    
    try:
        # レート制限を適用
        await rate_limiter.acquire()
        
        # OpenAI API呼び出し
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10  # e.g., "[10]"
        )
        response_text = completion.choices[0].message.content.strip()
        
        # レスポンスのパース
        import re
        matches = re.findall(r'\[(\d+)\]', response_text)
        if matches:
            # 抽出した番号が '0' の場合は、空リストではなく [-1] を返す
            if '0' in matches and len(matches) == 1:
                selected_indices = [-1]
            else:
                # '0' を除外し、1-based index to 0-based index に変換
                selected_indices = [
                    int(m) - 1 for m in matches if m != '0'
                ]
            
            CACHE_DATA[cache_key] = {"selected_indices": selected_indices}
            return selected_indices, None
        else:
            error_msg = (
                f"LLMの応答から選択番号を抽出できませんでした。"
                f"応答: {response_text}"
            )
            return None, error_msg
            
    except Exception as e:
        if hasattr(e, 'message'):
            error_details = e.message
        elif hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_details = e.response.text
        else:
            error_details = str(e)
            
        error_msg = (
            f"API呼び出し中にエラーが発生 (アンカー: {anchor_id}): {error_details}"
        )
        return None, error_msg


async def evaluate_model_on_selections_async(
    model_id, candidate_sets, all_record_ids_in_candidates,
    api_key, data_type
):
    """
    非同期でモデル評価を実行する関数 (selecting戦略)
    """
    predictions = []
    ground_truths = []
    # `predicted_positive_pairs` はクラスタリングのために維持
    predicted_positive_pairs = []
    errors = []
    processed_anchors = []
    
    print(
        f"\nモデル '{model_id}' で {len(candidate_sets)} アンカーの"
        "非同期評価を開始します..."
    )
    
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    client = AsyncOpenAI(api_key=api_key)

    async def evaluate_single_selection(anchor_info):
        async with semaphore:
            anchor_id, candidate_ids, index = anchor_info
            
            # LLM評価実行
            selected_indices, error_msg = await get_llm_evaluation_for_selection_async(
                client, anchor_id, candidate_ids, model_id,
                rate_limiter, data_type
            )
            
            if error_msg:
                return {
                    'index': index,
                    'anchor_id': anchor_id,
                    'error': error_msg,
                    'is_valid_result': False
                }
            
            # 正解ラベルの決定
            anchor_gt_cluster = GROUND_TRUTH_CLUSTERS.get(anchor_id)
            true_match_candidate_ids = set()
            is_truly_positive_exists = False
            if anchor_gt_cluster and not anchor_gt_cluster.startswith(
                "gt_orphan_"
            ):
                for cand_id in candidate_ids:
                    cand_gt_cluster = GROUND_TRUTH_CLUSTERS.get(cand_id)
                    if (cand_gt_cluster and
                            anchor_gt_cluster == cand_gt_cluster):
                        true_match_candidate_ids.add(cand_id)
                        is_truly_positive_exists = True

            # 予測が正解かどうかの判定と、予測ペアの収集
            prediction_correct = False
            selected_candidate_ids = []
            
            # `selected_indices` が None でなく、空でもなく、[-1] でもない場合
            if selected_indices and selected_indices[0] != -1:
                for idx in selected_indices:
                    if 0 <= idx < len(candidate_ids):
                        selected_id = candidate_ids[idx]
                        selected_candidate_ids.append(selected_id)
                        
                        # 予測ペアは正解・不正解を問わず追加 (バグ修正)
                        pair = tuple(sorted((anchor_id, selected_id)))
                        predicted_positive_pairs.append(pair)
                        
                        # 予測が正解セットに含まれているかチェック
                        if selected_id in true_match_candidate_ids:
                            prediction_correct = True

            # 最初の正解候補のみをレポート用に選択（なければNone）
            first_true_match = next(iter(true_match_candidate_ids), None)

            return {
                'index': index,
                'anchor_id': anchor_id,
                'ground_truth_exists': is_truly_positive_exists,
                'prediction_correct': prediction_correct,
                'selected_candidate_id': ", ".join(selected_candidate_ids) if selected_candidate_ids else "None",
                'true_match_candidate_id': first_true_match,
                'error': None,
                'is_valid_result': True
            }

    anchor_info_list = [
        (anchor_id, cands, i)
        for i, (anchor_id, cands) in enumerate(candidate_sets.items())
    ]
    
    batch_size = MAX_CONCURRENT_REQUESTS * 2
    
    desc = f"評価中 ({model_id})"
    with tqdm(total=len(anchor_info_list), desc=desc,
              unit="アンカー", leave=True, ncols=100) as pbar:
        for i in range(0, len(anchor_info_list), batch_size):
            batch = anchor_info_list[i:i + batch_size]
            
            tasks = [evaluate_single_selection(info) for info in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_task_errors = 0
            for result in batch_results:
                pbar.update(1)
                if isinstance(result, Exception):
                    batch_task_errors += 1
                    continue
                
                if not result.get('is_valid_result', False):
                    if result.get('error'):
                        errors.append((result['anchor_id'], result['error']))
                        # この行を追加してエラー内容を即時表示
                        pbar.write(f"エラー発生 (アンカー: {result['anchor_id']}): {result['error']}")
                    continue
                
                processed_anchors.append(result['anchor_id'])
                ground_truths.append(result['ground_truth_exists'])
                predictions.append(result['prediction_correct'])
            
            if batch_task_errors > 0:
                pbar.set_description(f"{desc} [タスクエラー: {batch_task_errors}件]")
            if len(errors) > 0:
                pbar.set_postfix_str(f"APIエラー: {len(errors)}件")
            
            if i % (batch_size * 3) == 0:
                save_cache(pbar)
    
    print(f"モデル '{model_id}' での非同期評価完了。エラー: {len(errors)}件。")
    
    return {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "predicted_positive_pairs": list(set(predicted_positive_pairs)),
        "errors": errors,
        "processed_anchors": processed_anchors,
    }


def calculate_pairwise_metrics(ground_truths, predictions, model_name=""):
    """ペアごとの評価指標と混合行列を計算する"""
    # Note: For selecting strategy, this is not a standard pairwise metric.
    # It's "Accuracy of selection for anchors that have a match".
    # - ground_truths: a true match exists for the anchor
    # - predictions: the model selected the correct match
    # TP: Correctly selected a true match.
    # FN: Failed to select a true match that existed.
    # FP: Selected a wrong candidate.
    # TN: Correctly selected [0] when no match existed.

    tp = sum(p and g for p, g in zip(predictions, ground_truths))
    fn = sum(not p and g for p, g in zip(predictions, ground_truths))
    fp = sum(p and not g for p, g in zip(predictions, ground_truths))
    tn = sum(not p and not g for p, g in zip(predictions, ground_truths))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * (precision * recall) / (precision + recall)
          if (precision + recall) > 0 else 0)
    
    print(f"\n--- {model_name} Selecting戦略 評価指標 ---")
    print("  解釈:")
    print("  - Ground Truth: アンカーに対応する正解マッチが候補内に存在するか")
    print("  - Prediction:   モデルがその正解マッチを正しく選択したか")
    print("  +----------------+-----------------+-----------------+")
    print(f"  | {'':^14} | {'Predicted: Yes':^15} | {'Predicted: No':^15} |")
    print("  +================+=================+=================+")
    print(f"  | {'GT: Yes':<14} | TP: {tp:<12d} | FN: {fn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  | {'GT: No':<14} | FP: {fp:<12d} | TN: {tn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  適合率 (Precision): {precision:.4f} (TP / (TP + FP))")
    print(f"  再現率 (Recall):    {recall:.4f} (TP / (TP + FN))")
    print(f"  F1スコア:           {f1:.4f}")

    metrics = {
        "precision": precision, "recall": recall, "f1_score": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }
    return metrics


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
    homogeneity, completeness, v_measure = (
        homogeneity_completeness_v_measure(true_labels, pred_labels)
    )
    
    print(f"\n--- {model_name} クラスタリング評価指標 ---")
    print(f"  調整ランド指数 (ARI): {ari:.4f}")
    print(f"  正規化相互情報量 (NMI): {nmi:.4f}")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Completeness: {completeness:.4f}")
    print(f"  V-measure: {v_measure:.4f}")
    
    metrics = {
        "ari": ari, "nmi": nmi, "homogeneity": homogeneity,
        "completeness": completeness, "v_measure": v_measure
    }
    return metrics


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


async def main(args):
    # OpenAI APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY 環境変数が設定されていません。")
        return
    
    # キャッシュと書誌データの読み込み
    load_cache()
    load_bib_data_and_gt_clusters(args.ground_truth_yaml)
    
    # 評価候補の読み込み
    candidate_sets, all_record_ids_in_candidates = (
        load_evaluation_candidates(args.candidates_json)
    )
    all_record_ids_list = sorted(list(all_record_ids_in_candidates))
    
    print("\n===== ファインチューニング前後のモデル性能比較評価 (selecting戦略) =====")
    print(f"ファインチューニング前のモデル: {args.model_before_ft}")
    print(f"ファインチューニング後のモデル: {args.model_after_ft}")
    print(f"最大同時リクエスト数: {MAX_CONCURRENT_REQUESTS}")
    print(f"1分間の最大リクエスト数: {REQUESTS_PER_MINUTE}")
    
    # ファインチューニング前のモデル評価
    print("\n===== ファインチューニング「前」のモデル性能評価 =====")
    start_time_before = time.time()
    results_before = await evaluate_model_on_selections_async(
        args.model_before_ft, candidate_sets,
        all_record_ids_in_candidates, api_key, args.data_type
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
    results_after = await evaluate_model_on_selections_async(
        args.model_after_ft, candidate_sets,
        all_record_ids_in_candidates, api_key, args.data_type
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
        
        num_total_pairs = (
            len(all_record_ids_global) * (len(all_record_ids_global) - 1) // 2
        )
        print(
            f"全ペア推論評価: {len(all_record_ids_global)}C2 = "
            f"{num_total_pairs} ペアのラベルを生成中..."
        )
        
        # 全ペア推論用の進捗バー
        pair_combinations = list(
            itertools.combinations(all_record_ids_global, 2)
        )
        desc = "全ペア推論"
        for r_id1, r_id2 in tqdm(pair_combinations, desc=desc,
                                 unit="ペア", leave=True, ncols=100):
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
            pred_b_similar = (pred_c1_b is not None and pred_c1_b == pred_c2_b)
            all_pairs_pred_labels_before.append(pred_b_similar)
            
            # ファインチューニング後モデルの予測ラベル
            pred_c1_a = pred_clusters_all_scope_after.get(str(r_id1))
            pred_c2_a = pred_clusters_all_scope_after.get(str(r_id2))
            pred_a_similar = (pred_c1_a is not None and pred_c1_a == pred_c2_a)
            all_pairs_pred_labels_after.append(pred_a_similar)
        
        model_name_before = f"{args.model_before_ft} (全ペア推論)"
        pairwise_metrics_all_before = calculate_pairwise_metrics(
            all_pairs_true_labels, all_pairs_pred_labels_before,
            model_name_before
        )
        model_name_after = f"{args.model_after_ft} (全ペア推論)"
        pairwise_metrics_all_after = calculate_pairwise_metrics(
            all_pairs_true_labels, all_pairs_pred_labels_after,
            model_name_after
        )
    else:
        empty_metrics = {
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
            "precision": 0, "recall": 0, "f1_score": 0
        }
        pairwise_metrics_all_before = empty_metrics
        pairwise_metrics_all_after = empty_metrics
    
    # 詳細結果のCSV作成 (To be implemented properly)
    # results_df_data = []
    
    # --- 出力先ディレクトリの準備 ---
    candidates_json_path = args.candidates_json
    # JSONファイルがあるディレクトリの親階層に "evaluation_results" を作成
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(candidates_json_path)),
        "evaluation_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n結果は次のディレクトリに保存されます: {output_dir}")

    # --- ファイル名の生成 ---
    model_before_ft_sanitized = sanitize_model_name_for_filename(
        args.model_before_ft
    )
    model_after_ft_sanitized = sanitize_model_name_for_filename(
        args.model_after_ft
    )
    base_filename = (
        f"eval_async_{os.path.basename(args.candidates_json).replace('.json', '')}_"
        f"before-{model_before_ft_sanitized}_"
        f"after-{model_after_ft_sanitized}"
    )

    # --- パフォーマンスレポートの生成 ---
    report_content = f"""# ファインチューニング性能評価レポート（selecting戦略）
日付: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 評価対象
- 書誌データ: {args.ground_truth_yaml}
- 候補リスト: {args.candidates_json} ({len(candidate_sets)} アンカー)
- 全レコード数: {len(all_record_ids_global)}

## 処理時間
- ファインチューニング前モデル評価時間: {end_time_before - start_time_before:.2f}秒
- ファインチューニング後モデル評価時間: {end_time_after - start_time_after:.2f}秒
- 平均処理時間（前）: {(end_time_before - start_time_before) / len(candidate_sets) if len(candidate_sets) > 0 else 0:.3f}秒/アンカー
- 平均処理時間（後）: {(end_time_after - start_time_after) / len(candidate_sets) if len(candidate_sets) > 0 else 0:.3f}秒/アンカー

## Selecting戦略 評価
### ファインチューニング前モデル ({args.model_before_ft})
- 混合行列:
    (解釈: GT=正解マッチが存在, Predicted=正解マッチを選択)
    予測      |  Predicted: Yes | Predicted: No
  ------------|-----------------|-----------------
  GT: Yes     | TP: {pairwise_metrics_before['tp']:<15d} | FN: {pairwise_metrics_before['fn']:<15d}
  GT: No      | FP: {pairwise_metrics_before['fp']:<15d} | TN: {pairwise_metrics_before['tn']:<15d}
- 適合率: {pairwise_metrics_before['precision']:.4f}, 再現率: {pairwise_metrics_before['recall']:.4f}, F1: {pairwise_metrics_before['f1_score']:.4f}
- エラー数: {len(results_before['errors'])}

### クラスタリング評価（前）
- ARI: {clustering_metrics_before['ari']:.4f}
- NMI: {clustering_metrics_before['nmi']:.4f}
- Homogeneity: {clustering_metrics_before['homogeneity']:.4f}, Completeness: {clustering_metrics_before['completeness']:.4f}, V-measure: {clustering_metrics_before['v_measure']:.4f}

### ファインチューニング後モデル ({args.model_after_ft})
- 混合行列:
    (解釈: GT=正解マッチが存在, Predicted=正解マッチを選択)
    予測      |  Predicted: Yes | Predicted: No
  ------------|-----------------|-----------------
  GT: Yes     | TP: {pairwise_metrics_after['tp']:<15d} | FN: {pairwise_metrics_after['fn']:<15d}
  GT: No      | FP: {pairwise_metrics_after['fp']:<15d} | TN: {pairwise_metrics_after['tn']:<15d}
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
        formatted_clusters_before = format_clusters_with_details(
            pred_clusters_before, BIB_DATA
        )
        filename_before = os.path.join(
            output_dir, f"{base_filename}_clusters_before.json"
        )
        with open(filename_before, "w", encoding="utf-8") as f:
            json.dump(
                formatted_clusters_before, f, ensure_ascii=False, indent=4
            )
        print(
            f"ファインチューニング前予測クラスタ詳細を {filename_before} に保存しました。"
        )
        
        formatted_clusters_after = format_clusters_with_details(
            pred_clusters_after, BIB_DATA
        )
        filename_after = os.path.join(
            output_dir, f"{base_filename}_clusters_after.json"
        )
        with open(filename_after, "w", encoding="utf-8") as f:
            json.dump(
                formatted_clusters_after, f, ensure_ascii=False, indent=4
            )
        print(
            f"ファインチューニング後予測クラスタ詳細を {filename_after} に保存しました。"
        )
    except Exception as e:
        print(f"エラー: 詳細な予測クラスタ情報のJSON保存に失敗: {e}")
    
    # 最終的にキャッシュを保存
    save_cache()
    
    print("\n===== 評価完了 =====")
    total_time = end_time_after - start_time_before
    print(f"総処理時間: {total_time:.2f}秒")
    f1_improvement = (
        pairwise_metrics_after['f1_score'] -
        pairwise_metrics_before['f1_score']
    )
    print(f"F1スコア改善: {f1_improvement:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ファインチューニング前後のLLM性能をselecting戦略で非同期評価するスクリプト"
    )
    parser.add_argument(
        "--candidates_json",
        required=True,
        help="評価候補のJSONファイルパス (KNNグラフ)"
    )
    parser.add_argument(
        "--ground_truth_yaml",
        required=True,
        help="正解データのYAMLファイルパス"
    )
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
        help=(
            "ファインチューニング前のモデルID。デフォルト: "
            f"{DEFAULT_MODEL_ID_BEFORE_FINETUNING}"
        )
    )
    parser.add_argument(
        "--model_after_ft",
        required=True,
        help="ファインチューニング後のモデルID (必須)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=20,
        help="最大同時リクエスト数"
    )
    parser.add_argument(
        "--requests_per_minute",
        type=int, default=3000,
        help="1分間の最大リクエスト数"
    )
    
    args = parser.parse_args()
    
    # グローバル設定を更新
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    REQUESTS_PER_MINUTE = args.requests_per_minute
    REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE
    
    # 非同期実行
    asyncio.run(main(args))