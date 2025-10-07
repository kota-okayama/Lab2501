#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llamaを使用したエンティティマッチング評価のテストスクリプト

OpenAIの代わりにOllamaを使用してLLMによるエンティティマッチングを実行します。
"""

import asyncio
import aiohttp
import time
import pickle
import yaml
import os
import argparse
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm.asyncio import tqdm
import re

# --- グローバル変数 ---
BIB_DATA = {}
GROUND_TRUTH_CLUSTERS = {}
CACHE_DATA = {}
CACHE_FILE = "llama_evaluation_cache.pkl"

# Ollama設定
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
MAX_CONCURRENT_REQUESTS = 5


def load_cache():
    """キャッシュファイルを読み込む"""
    global CACHE_DATA
    CACHE_DATA = {}
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                CACHE_DATA = pickle.load(f)
            print(f"キャッシュを読み込みました: {len(CACHE_DATA)} 件")
        except Exception as e:
            print(f"キャッシュファイルの読み込みに失敗: {e}")
    else:
        print("新しいキャッシュファイルを作成します。")


def save_cache(pbar=None):
    """キャッシュをファイルに保存する"""
    global CACHE_DATA
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(CACHE_DATA, f)
        
        message = f"キャッシュを保存しました: {len(CACHE_DATA)} 件"
        if pbar:
            pbar.write(message)
        else:
            print(message)
    except Exception as e:
        error_message = f"キャッシュファイルの保存に失敗: {e}"
        if pbar:
            pbar.write(error_message)
        else:
            print(error_message)


def load_product_data_and_gt_clusters(yaml_path):
    """YAMLファイルから製品データと正解クラスタ情報を読み込む"""
    global BIB_DATA, GROUND_TRUTH_CLUSTERS
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAMLファイルが見つかりません: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    BIB_DATA = {}
    GROUND_TRUTH_CLUSTERS = {}
    
    if 'records' in data:
        orphan_counter = 0
        
        for cluster_id, records in data['records'].items():
            cluster_id_str = str(cluster_id)
            for record in records:
                record_id = record['id']
                product_record = record['data'].copy()
                product_record['record_id'] = record_id
                BIB_DATA[record_id] = product_record
                
                if len(records) == 1:
                    GROUND_TRUTH_CLUSTERS[record_id] = (
                        f"gt_orphan_{orphan_counter}"
                    )
                    orphan_counter += 1
                else:
                    GROUND_TRUTH_CLUSTERS[record_id] = cluster_id_str
    else:
        raise ValueError("YAMLファイルに 'records' キーがありません")
    
    print(f"製品データを読み込みました: {len(BIB_DATA)} レコード")
    print(f"正解クラスタ情報を読み込みました: {len(GROUND_TRUTH_CLUSTERS)} レコード")


def load_evaluation_pairs(csv_path, limit=None):
    """CSVファイルから評価用ペアを読み込む"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
    
    pairs_df = pd.read_csv(csv_path)
    if limit:
        pairs_df = pairs_df.head(limit)
        print(f"評価ペアを {limit} 件に制限しました。")

    pairs = []
    all_record_ids = set()

    for _, row in pairs_df.iterrows():
        record_id_1 = str(row['record_id_1']).strip()
        record_id_2 = str(row['record_id_2']).strip()
        pairs.append((record_id_1, record_id_2))
        all_record_ids.add(record_id_1)
        all_record_ids.add(record_id_2)

    print(
        f"評価ペアを読み込みました: {len(pairs)} ペア "
        f"({len(all_record_ids)} ユニークレコード)"
    )
    return pairs, all_record_ids


def get_record_details_for_prompt(record_id):
    """レコードIDから製品情報を取得してプロンプト形式で返す"""
    global BIB_DATA
    
    if record_id not in BIB_DATA:
        return f"レコードID {record_id} の製品情報が見つかりません"
    
    record = BIB_DATA[record_id]
    details = []
    
    for field, value in record.items():
        if field == 'record_id':
            continue
        
        if isinstance(value, str) and value.strip():
            field_name = field.title()
            details.append(f"{field_name}: {value.strip()}")
        elif value is not None and str(value).strip():
            field_name = field.title()
            details.append(f"{field_name}: {str(value).strip()}")

    if not details:
        return (
            f"レコードID {record_id} には表示可能な製品情報フィールドが"
            "ありません"
        )
    
    return f"Record ID: {record_id}\n" + "\n".join(details)


def get_product_matching_prompts():
    """製品マッチング用のプロンプトを返す"""
    system_prompt = (
        "You are an expert at determining whether two product records "
        "refer to essentially the same product.\n"
        "First, please clearly answer 'Yes' if you believe the two "
        "product records refer to the same product, or 'No' otherwise.\n"
        "Next, provide a confidence score from 0.0 (completely different) "
        "to 1.0 (completely identical) indicating your certainty in this "
        "judgment.\n"
        "Your judgment must strictly follow these rules:\n"
        " - If the confidence score is 0.5 or higher, your answer must be "
        "'Yes'.\n"
        " - If the confidence score is below 0.5, your answer must be "
        "'No'.\n"
        "Format your response as:\n"
        "Answer: [Yes/No]\n"
        "Confidence Score: [0.0-1.0]\n"
    )
    user_prompt_template = (
        "Please determine whether the following two product records refer "
        "to essentially the same product.\n\n"
        "Product 1:\n{info_1}\n\n"
        "Product 2:\n{info_2}\n\n"
        "Do these refer to the same product?\n"
    )

    return system_prompt, user_prompt_template


async def get_llm_evaluation_for_pair_async_ollama(
    session, record_id_1, record_id_2, model_id, ollama_url
):
    """Ollamaを使用した非同期LLM評価関数"""
    global CACHE_DATA
    
    cache_key = f"{record_id_1}_{record_id_2}_{model_id}_wdc_product"

    # キャッシュチェック
    if cache_key in CACHE_DATA:
        cached_item = CACHE_DATA[cache_key]
        if (
            isinstance(cached_item, dict)
            and "is_similar" in cached_item
            and "score" in cached_item
        ):
            return cached_item["is_similar"], cached_item["score"], None

    # 製品情報取得
    info_1 = get_record_details_for_prompt(record_id_1)
    info_2 = get_record_details_for_prompt(record_id_2)
    
    if "見つかりません" in info_1 or "フィールドがありません" in info_1:
        return (
            None,
            None,
            f"レコード {record_id_1} の情報取得に失敗: {info_1}"
        )
    if "見つかりません" in info_2 or "フィールドがありません" in info_2:
        return (
            None,
            None,
            f"レコード {record_id_2} の情報取得に失敗: {info_2}"
        )

    system_prompt, user_prompt_template = get_product_matching_prompts()
    user_prompt = user_prompt_template.format(info_1=info_1, info_2=info_2)

    try:
        # Ollama API呼び出し
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 100
            }
        }
        
        async with session.post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return (
                    None,
                    None,
                    f"Ollama API Error: {response.status} - {error_text}"
                )

            result = await response.json()
            response_text = result["message"]["content"].strip()

        # レスポンスのパース
        lines = response_text.split("\n")
        is_similar_str = ""
        similarity_score_str = ""

        # "Answer:"行を探す
        for line in lines:
            line = line.strip()
            if line.lower().startswith("answer:"):
                is_similar_str = line.split(":", 1)[1].strip().lower()
                break

        # "Confidence Score:"行を探す
        for line in lines:
            line = line.strip()
            if line.lower().startswith("confidence score:"):
                similarity_score_str = line.split(":", 1)[1].strip()
                break

        # フォールバック: 全体から探す
        if not is_similar_str:
            first_line_check = lines[0].strip().lower() if lines else ""
            if "yes" in first_line_check:
                is_similar_str = "yes"
            elif "no" in first_line_check:
                is_similar_str = "no"

        # スコアのフォールバック
        if not similarity_score_str:
            score_pattern = r"(?:confidence score|score):\s*([0-9.]+)"
            match = re.search(score_pattern, response_text, re.IGNORECASE)
            if match:
                similarity_score_str = match.group(1)

        # パース処理
        parsed_is_similar = None
        if is_similar_str:
            if "yes" in is_similar_str:
                parsed_is_similar = True
            elif "no" in is_similar_str:
                parsed_is_similar = False
        
        parsed_similarity_score = None
        if similarity_score_str:
            try:
                parsed_similarity_score = float(similarity_score_str)
                if not (0.0 <= parsed_similarity_score <= 1.0):
                    print(
                        f"警告: ペア ({record_id_1}, {record_id_2}) の類似度"
                        f"スコア '{parsed_similarity_score}' が範囲外です。"
                    )
                    # クリップ
                    parsed_similarity_score = max(
                        0.0, min(1.0, parsed_similarity_score)
                    )
            except ValueError:
                print(
                    f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコア"
                    f"が数値に変換できません: '{similarity_score_str}'"
                )

        # フォールバック処理
        if parsed_is_similar is None and parsed_similarity_score is not None:
            parsed_is_similar = parsed_similarity_score >= 0.5
        elif parsed_is_similar is not None and parsed_similarity_score is None:
            parsed_similarity_score = 1.0 if parsed_is_similar else 0.0

        if parsed_is_similar is None:
            return (
                None,
                None,
                f"LLMの応答から判定を抽出できませんでした。応答: {response_text}"
            )
        if parsed_similarity_score is None:
            return (
                parsed_is_similar,
                None,
                f"LLMの応答から類似度スコアを抽出できませんでした。応答: {response_text}"
            )

        # キャッシュに保存
        CACHE_DATA[cache_key] = {
            "is_similar": parsed_is_similar,
            "score": parsed_similarity_score
        }

        return parsed_is_similar, parsed_similarity_score, None
        
    except Exception as e:
        error_msg = (
            f"Ollama API Error for pair ({record_id_1}, {record_id_2}): "
            f"{type(e).__name__} - {e}\n"
        )
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(error_msg)
        return None, None, error_msg


async def evaluate_model_on_pairs_async_ollama(
    model_id, pairs_to_evaluate, all_record_ids_in_pairs, ollama_url
):
    """Ollamaを使用した非同期モデル評価関数"""
    predictions = []
    ground_truths = []
    predicted_positive_pairs = []
    llm_scores = []
    errors = []
    processed_pairs = []

    print(
        f"\nモデル '{model_id}' で {len(pairs_to_evaluate)} ペア"
        "のOllama評価を開始します..."
    )

    # HTTPセッションを初期化
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        async def evaluate_single_pair(pair_info):
            async with semaphore:
                r_id1, r_id2, index = pair_info
                
                # 正解ラベルの決定
                gt_cluster1 = GROUND_TRUTH_CLUSTERS.get(r_id1)
                gt_cluster2 = GROUND_TRUTH_CLUSTERS.get(r_id2)
                
                is_truly_similar = False
                if (
                    gt_cluster1 is not None
                    and gt_cluster2 is not None
                    and not gt_cluster1.startswith("gt_orphan_")
                    and not gt_cluster2.startswith("gt_orphan_")
                    and gt_cluster1 == gt_cluster2
                ):
                    is_truly_similar = True

                # Ollama評価実行
                (
                    llm_is_similar,
                    llm_score,
                    error_msg
                ) = await get_llm_evaluation_for_pair_async_ollama(
                    session, r_id1, r_id2, model_id, ollama_url
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

        # バッチサイズを設定
        batch_size = MAX_CONCURRENT_REQUESTS * 2

        # 進捗バーを初期化
        with tqdm(
            total=len(pair_info_list),
            desc=f"評価中 ({model_id})",
            unit="ペア",
            leave=True,
            ncols=100
        ) as pbar:
            for i in range(0, len(pair_info_list), batch_size):
                batch = pair_info_list[i:i + batch_size]

                # バッチを並列実行
                tasks = [
                    evaluate_single_pair(pair_info) for pair_info in batch
                ]
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
                        f"評価中 ({model_id}) [タスクエラー: {batch_task_errors}件]"
                    )
                if len(errors) > 0:
                    pbar.set_postfix_str(f"APIエラー: {len(errors)}件")

                # 進捗保存
                if i % (batch_size * 3) == 0:
                    save_cache(pbar)

    print(f"モデル '{model_id}' でのOllama評価完了。エラー: {len(errors)}件。")

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
    print(f"  | {'Ground Truth':^14} | {'Predicted: Pos':^15} | {'Predicted: Neg':^15} |")
    print("  +================+=================+=================+")
    print(f"  | {'Positive':<14} | TP: {tp:<12d} | FN: {fn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  | {'Negative':<14} | FP: {fp:<12d} | TN: {tn:<12d} |")
    print("  +----------------+-----------------+-----------------+")
    print(f"  適合率 (Precision): {precision:.4f} (TP / (TP + FP))")
    print(f"  再現率 (Recall):    {recall:.4f} (TP / (TP + FN))")
    print(f"  F1スコア:           {f1:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


async def main(args):
    # キャッシュと製品データの読み込み
    load_cache()
    load_product_data_and_gt_clusters(args.ground_truth_yaml)
    
    # 評価ペアの読み込み
    pairs_to_evaluate, all_record_ids_in_pairs = load_evaluation_pairs(
        args.pairs_csv, args.limit_pairs
    )

    print("\n===== Llamaによる製品マッチング評価 =====")
    print(f"使用モデル: {args.model}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"最大同時リクエスト数: {MAX_CONCURRENT_REQUESTS}")
    
    # モデル評価
    print("\n===== Llama評価開始 =====")
    start_time = time.time()
    results = await evaluate_model_on_pairs_async_ollama(
        args.model, pairs_to_evaluate, all_record_ids_in_pairs, args.ollama_url
    )
    end_time = time.time()
    
    pairwise_metrics = calculate_pairwise_metrics(
        results["ground_truths"],
        results["predictions"],
        args.model
    )

    # 詳細結果のCSV作成
    results_df_data = []
    for i, (r_id1, r_id2) in enumerate(results["processed_pairs"]):
        results_df_data.append({
            "record_id_1": r_id1,
            "record_id_2": r_id2,
            "ground_truth_similar": results["ground_truths"][i],
            "predicted_similar": results["predictions"][i],
            "score": results["llm_scores"][i],
            "error": next(
                (
                    err[1] for pair_ids, err in results["errors"]
                    if pair_ids == (r_id1, r_id2)
                ),
                None
            ),
        })

    # 結果の保存
    output_dir = (
        "/Users/kasiwamochi/Document/Lab2501/results_wdc/run_1k_wdc/"
        "evaluation_results"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名の生成
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"llama_eval_{args.model}_{timestamp}"

    # 詳細結果の保存
    detailed_csv_filename = os.path.join(
        output_dir, f"{base_filename}_details.csv"
    )
    detailed_results_df = pd.DataFrame(results_df_data)
    detailed_results_df.to_csv(
        detailed_csv_filename, index=False, encoding="utf-8-sig"
    )
    print(f"\n詳細な評価結果を {detailed_csv_filename} に保存しました。")

    # パフォーマンスレポートの生成
    report_content = f"""# Llama製品マッチング評価レポート
日付: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 評価対象
- 製品データ: {args.ground_truth_yaml}
- 候補ペアリスト: {args.pairs_csv} ({len(pairs_to_evaluate)} ペア)
- 使用モデル: {args.model}
- Ollama URL: {args.ollama_url}

## 処理時間
- 評価時間: {end_time - start_time:.2f}秒
- 平均処理時間: {(end_time - start_time) / len(pairs_to_evaluate):.3f}秒/ペア

## ペアワイズ評価結果
- 混合行列:
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics['tp']:<18d} | FN: {pairwise_metrics['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics['fp']:<18d} | TN: {pairwise_metrics['tn']:<18d}
- 適合率: {pairwise_metrics['precision']:.4f}
- 再現率: {pairwise_metrics['recall']:.4f}
- F1スコア: {pairwise_metrics['f1_score']:.4f}
- エラー数: {len(results['errors'])}

## エラー詳細
"""
    
    if results['errors']:
        report_content += "\n### エラーが発生したペア:\n"
        for i, (pair, error) in enumerate(results['errors'][:10]):
            report_content += f"{i+1}. {pair}: {error[:100]}...\n"
        if len(results['errors']) > 10:
            report_content += f"... 他 {len(results['errors']) - 10} 件のエラー\n"
    else:
        report_content += "エラーは発生しませんでした。\n"
    
    # レポートの保存
    report_filename = os.path.join(output_dir, f"{base_filename}_report.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"パフォーマンスレポートを {report_filename} に保存しました。")
    
    # 最終的にキャッシュを保存
    save_cache()
    
    print("\n===== 評価完了 =====")
    print(f"総処理時間: {end_time - start_time:.2f}秒")
    print(f"F1スコア: {pairwise_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Llamaを使用した製品マッチング評価スクリプト"
    )
    parser.add_argument(
        "--pairs_csv", required=True, help="評価ペアのCSVファイルパス"
    )
    parser.add_argument(
        "--ground_truth_yaml", required=True, help="正解データのYAMLファイルパス"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"使用するLlamaモデル名。デフォルト: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--ollama_url", type=str, default=DEFAULT_OLLAMA_URL,
        help=f"Ollama サーバーのURL。デフォルト: {DEFAULT_OLLAMA_URL}"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=5, help="最大同時リクエスト数"
    )
    parser.add_argument(
        "--limit_pairs", type=int, default=None,
        help="評価ペアの数を制限します。デバッグ用。"
    )

    args = parser.parse_args()
    
    # グローバル設定を更新
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    
    # 非同期実行
    asyncio.run(main(args))
