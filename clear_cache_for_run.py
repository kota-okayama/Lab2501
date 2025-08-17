#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指定した実行ディレクトリ配下のCSVに含まれる全てのペア( record_id_1, record_id_2 )に対応する
LLM評価キャッシュ(JSON/Pickle)キーを部分削除するスクリプト。

- キャッシュファイル:
  - Pickle: llm_evaluation_cache.pkl
  - JSON  : openai_embedding_experiment/evaluation_results/llm_api_cache.json

- キー仕様（evaluate_finetuning_performance_async.py より）
  - data_type == 'bib':    f"{id1}_{id2}_{model_id}"
  - data_type in others:   f"{id1}_{id2}_{model_id}_{data_type}"

本スクリプトでは、必要に応じて model_id を指定して削除可能。
  - すべてのモデルを対象: --model を指定しない
  - 特定モデルのみ:      --model を複数回指定（例: --model gpt-4o-mini-2024-07-18）

判定ロジック（順不同の両向き対応）
  - data_type == 'bib':
      接頭: key.startswith(f"{id1}_{id2}_") or key.startswith(f"{id2}_{id1}_")
      かつ（モデル指定あり時のみ）接尾: key.endswith(f"_{model}")
  - others:
      上記 接頭 判定に加え、key.endswith(f"_{data_type}")
      さらに（モデル指定あり時のみ）key.endswith(f"_{model}_{data_type}")

使い方:
  python3 clear_cache_for_run.py \
    --run-dir "/path/to/results_music/run_1k_music" \
    --data-type music \
    [--model gpt-4o-mini-2024-07-18] [--model ft:gpt-4o-mini-...:suffix] \
    [--dry-run]
"""

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PICKLE_CACHE_PATH = os.path.join(PROJECT_ROOT, "llm_evaluation_cache.pkl")
JSON_CACHE_PATH = os.path.join(
    PROJECT_ROOT,
    "openai_embedding_experiment",
    "evaluation_results",
    "llm_api_cache.json",
)


def load_json_cache(path: str) -> Dict[str, dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_json_cache(path: str, data: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_pickle_cache(path: str) -> Dict[str, dict]:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_pickle_cache(path: str, data: Dict[str, dict]) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def find_csv_files(root_dir: str) -> List[str]:
    csv_files: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".csv"):
                csv_files.append(os.path.join(base, fn))
    return csv_files


def iter_pairs_from_csv(csv_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            if {"record_id_1", "record_id_2"}.issubset(headers):
                for row in reader:
                    id1 = str(row.get("record_id_1", "")).strip()
                    id2 = str(row.get("record_id_2", "")).strip()
                    if id1 and id2:
                        pairs.append((id1, id2))
    except Exception:
        # 壊れたCSVなどは無視して続行
        pass
    return pairs


def collect_pairs(run_dir: str) -> Set[Tuple[str, str]]:
    all_pairs: Set[Tuple[str, str]] = set()
    for csv_path in find_csv_files(run_dir):
        pairs = iter_pairs_from_csv(csv_path)
        all_pairs.update(pairs)
    return all_pairs


def should_delete_key_for_pair(
    key: str,
    id1: str,
    id2: str,
    data_type: str,
    models: Optional[Set[str]] = None,
) -> bool:
    # どのモデルでもマッチするよう接頭/接尾で判定
    prefix_a = f"{id1}_{id2}_"
    prefix_b = f"{id2}_{id1}_"
    if not (key.startswith(prefix_a) or key.startswith(prefix_b)):
        return False

    # data_type による末尾判定
    if data_type == "bib":
        if not models:
            return True
        # モデル指定あり: 末尾が _{model}
        return any(key.endswith(f"_{m}") for m in models)

    # others (music/person): 末尾は常に _{data_type}
    if not key.endswith(f"_{data_type}"):
        return False

    if not models:
        return True
    # モデル指定あり: 末尾が _{model}_{data_type}
    return any(key.endswith(f"_{m}_{data_type}") for m in models)


def process_keys(
    keys_to_check: Set[str],
    pairs: Set[Tuple[str, str]],
    data_type: str,
    model_set: Optional[Set[str]],
) -> Set[str]:
    """
    指定されたキーセットから、ペアに一致するものを効率的に見つけ出す
    """
    to_delete = set()
    # パフォーマンス向上のため、ペアをIDでインデックス化（双方向）
    # {id1: {id2, id3}, id2: {id1}, ...}
    pair_map = defaultdict(set)
    for id1, id2 in pairs:
        pair_map[id1].add(id2)
        pair_map[id2].add(id1)

    # O(キー数 * IDあたりの平均ペア数) になるようにループを構成
    for key in keys_to_check:
        # キーの構造に関する仮定: 最初のアンダースコアまでが片方のIDである
        # この仮定が崩れると正しく動作しないが、パフォーマンスのために必要
        try:
            id1_candidate = key.split("_", 1)[0]
        except IndexError:
            continue

        # 抽出したIDがペアリストに存在する場合のみ、詳細なチェックを行う
        if id1_candidate in pair_map:
            # このIDが持つすべてのペア候補をチェック
            for id2_candidate in pair_map[id1_candidate]:
                if should_delete_key_for_pair(
                    key, id1_candidate, id2_candidate, data_type, model_set
                ):
                    to_delete.add(key)
                    # このキーのペアが見つかったので、次のキーのチェックに移る
                    break
    return to_delete


def main():
    parser = argparse.ArgumentParser(
        description="ディレクトリ配下のペアに該当するLLMキャッシュを部分削除"
    )
    parser.add_argument("--run-dir", required=True, help="ペアCSVを含むルートディレクトリ")
    parser.add_argument(
        "--data-type",
        required=True,
        choices=["bib", "music", "person"],
        help="対象データタイプ（キー末尾の判定に使用）",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="対象モデルID（複数指定可・未指定なら全モデル対象）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="削除せずに対象キー数のみ表示"
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"エラー: ディレクトリが存在しません: {run_dir}")
        return

    model_set: Optional[Set[str]] = set(args.model) if args.model else None
    if model_set:
        print(f"モデル指定: {sorted(model_set)}")
    else:
        print("モデル指定: なし（全モデル対象）")

    print(f"走査対象ディレクトリ: {run_dir}")

    # 1) すべてのペアを収集
    pairs = collect_pairs(run_dir)
    print(f"検出ペア数: {len(pairs)} 件")

    if not pairs:
        print("対象ペアが見つかりませんでした。処理を終了します。")
        return

    # 2) キャッシュのロード
    json_cache = load_json_cache(JSON_CACHE_PATH)
    pkl_cache = load_pickle_cache(PICKLE_CACHE_PATH)

    json_keys = set(json_cache.keys())
    pkl_keys = set(pkl_cache.keys())

    # 3) 削除対象キーを抽出 (最適化されたロジック)
    print("削除対象キーの抽出を開始します（最適化ロジック使用）...")
    json_to_delete = process_keys(json_keys, pairs, args.data_type, model_set)
    pkl_to_delete = process_keys(pkl_keys, pairs, args.data_type, model_set)

    to_delete = set()
    for key in json_to_delete:
        to_delete.add(("json", key))
    for key in pkl_to_delete:
        to_delete.add(("pkl", key))

    if not to_delete:
        print("削除対象キーが見つかりませんでした。")
        return

    # 4) レポート
    print(f"削除対象キー総数: {len(to_delete)} 件")
    preview = list(sorted(to_delete))[:20]
    if preview:
        print("サンプル(最大20件):")
        for kind, key in preview:
            print(f"  [{kind}] {key}")

    if args.dry_run:
        print("dry-runのため削除は行いません。")
        return

    # 5) 実削除
    removed_json = 0
    removed_pkl = 0
    for kind, key in to_delete:
        if kind == "json" and key in json_cache:
            del json_cache[key]
            removed_json += 1
        elif kind == "pkl" and key in pkl_cache:
            del pkl_cache[key]
            removed_pkl += 1

    save_json_cache(JSON_CACHE_PATH, json_cache)
    save_pickle_cache(PICKLE_CACHE_PATH, pkl_cache)

    print(f"削除完了: JSON={removed_json} 件, Pickle={removed_pkl} 件")


if __name__ == "__main__":
    main()