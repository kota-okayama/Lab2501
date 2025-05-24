import csv
import json
import os
import time  # API呼び出しのレート制限対策用
import yaml
import pandas as pd
import itertools
from sklearn.metrics import (
    precision_recall_fscore_support,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
    confusion_matrix,
)
import networkx as nx
from openai import OpenAI  # もしV1未満の古いライブラリをお使いの場合は要調整
import argparse

# --- グローバル設定 (デフォルト値として使用) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))

DEFAULT_RECORD_YAML_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024"
DEFAULT_RECORD_YAML_FILENAME = "extract_subset_10k.yml"
DEFAULT_EVAL_PAIRS_CSV_DIR_RELATIVE_TO_BASE_DIR = "."  # スクリプトと同じ場所
DEFAULT_EVAL_PAIRS_CSV_FILENAME = "evaluation_candidate_pairs.csv"
DEFAULT_OUTPUT_DIR_RELATIVE_TO_BASE_DIR = "evaluation_results"
DEFAULT_MODEL_ID_BEFORE_FINETUNING = "gpt-4o-mini-2024-07-18"
# MODEL_ID_AFTER_FINETUNING は引数で必須とする

# グローバル変数は main 関数内でパスが確定してから設定
RECORD_YAML_PATH = None
EVALUATION_PAIRS_CSV_PATH = None
OUTPUT_DIR = None  # グローバルスコープでは None で初期化
PERFORMANCE_REPORT_PATH = None
DETAILED_RESULTS_CSV_PATH = None
CACHE_FILE_PATH = None
# グローバルスコープでの os.makedirs 呼び出しは削除 (もし存在する場合)

# キャッシュデータ (グローバル)
CACHE_DATA = {}

# OpenAI APIクライアント (グローバル)
client = None  # main の最初で初期化する

# モデルID
MODEL_ID_BEFORE_FINETUNING = "gpt-4o-mini-2024-07-18"
MODEL_ID_AFTER_FINETUNING = None  # スクリプト実行時にユーザーに入力させるか、引数で渡す

# グローバル変数として書誌データを保持
BIB_DATA = {}
# グローバル変数として正解クラスタIDを保持 (record_id -> cluster_id)
GROUND_TRUTH_CLUSTERS = {}


# --- キャッシュ管理関数 ---
def load_cache():
    """キャッシュファイルをロードする"""
    global CACHE_DATA, CACHE_FILE_PATH
    if not CACHE_FILE_PATH:
        print("エラー: キャッシュファイルパスが設定されていません。ロードをスキップします。")
        return
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                CACHE_DATA = json.load(f)
            print(f"{len(CACHE_DATA)} 件のキャッシュを {CACHE_FILE_PATH} からロードしました。")
        except json.JSONDecodeError as e:
            print(
                f"エラー: キャッシュファイル {CACHE_FILE_PATH} のJSON形式が正しくありません: {e}。新規キャッシュで開始します。"
            )
            CACHE_DATA = {}
        except Exception as e:
            print(
                f"エラー: キャッシュファイル {CACHE_FILE_PATH} の読み込み中に予期せぬエラー: {e}。新規キャッシュで開始します。"
            )
            CACHE_DATA = {}
    else:
        print(f"キャッシュファイル {CACHE_FILE_PATH} は見つかりませんでした。新規に作成されます。")


def save_cache():
    """メモリ上のキャッシュをファイルに保存する"""
    global CACHE_DATA, CACHE_FILE_PATH
    if not CACHE_FILE_PATH:
        print("エラー: キャッシュファイルパスが設定されていません。保存をスキップします。")
        return
    try:
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(CACHE_DATA, f, ensure_ascii=False, indent=4)
        print(f"{len(CACHE_DATA)} 件のキャッシュを {CACHE_FILE_PATH} に保存しました。")
    except Exception as e:
        print(f"エラー: キャッシュファイル {CACHE_FILE_PATH} への書き込み中にエラー: {e}")


# --- データ読み込み関数 ---


def load_bib_data_and_gt_clusters(yaml_path):
    """
    record.ymlから書誌データと正解クラスタIDをロードする。
    BIB_DATA と GROUND_TRUTH_CLUSTERS をグローバルに設定する。
    """
    global BIB_DATA, GROUND_TRUTH_CLUSTERS
    BIB_DATA = {}
    GROUND_TRUTH_CLUSTERS = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        return False
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        # prepare_finetuning_data.py の load_bib_data_for_finetuning を参考に構造を解析
        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            if isinstance(possible_records_dict, dict):
                for key, value_list in possible_records_dict.items():
                    if key in ["version", "type", "id", "summary", "inf_attr"]:
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            record_id_str = None
                            if isinstance(record, dict) and "id" in record:
                                record_id_str = str(record["id"])
                                if "data" in record:
                                    BIB_DATA[record_id_str] = record["data"]
                                else:  # 'data'キーがない場合、idとcluster_id以外の全てをデータとする
                                    record_data_candidate = {
                                        k_rec: v_rec
                                        for k_rec, v_rec in record.items()
                                        if k_rec not in ["id", "cluster_id"]
                                    }
                                    if record_data_candidate:
                                        BIB_DATA[record_id_str] = record_data_candidate

                                if "cluster_id" in record:
                                    GROUND_TRUTH_CLUSTERS[record_id_str] = str(record["cluster_id"])
                                else:
                                    # cluster_id がないレコードは、それ自身がクラスタを形成するとみなすか、
                                    # あるいは特定のデフォルトIDを振るか。ここではユニークなIDを振っておく。
                                    GROUND_TRUTH_CLUSTERS[record_id_str] = f"gt_orphan_{record_id_str}"

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。")
            return False
        if not GROUND_TRUTH_CLUSTERS:
            print(f"エラー: {yaml_path} から正解クラスタIDロード不可、または空。")
            # cluster_idが必須でなければ、このチェックは緩和しても良い
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


def load_evaluation_pairs(csv_path):
    """
    評価対象のペアリストをCSVからロードする。
    CSVには 'record_id_1' と 'record_id_2' の列が必要。
    Returns:
        list of tuples: [(record_id_1, record_id_2), ...]
    """
    if not os.path.exists(csv_path):
        print(f"エラー: 評価ペアリストファイルが見つかりません: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        if "record_id_1" not in df.columns or "record_id_2" not in df.columns:
            print(f"エラー: {csv_path} に 'record_id_1' または 'record_id_2' の列がありません。")
            return None

        pairs = []
        for _, row in df.iterrows():
            pairs.append((str(row["record_id_1"]), str(row["record_id_2"])))

        print(f"{len(pairs)} 組の評価対象ペアを {csv_path} からロードしました。")
        return pairs
    except Exception as e:
        print(f"エラー: 評価ペアリストファイル ({csv_path}) の読み込み中にエラー: {e}")
        return None


# --- 書誌情報取得関数 (prepare_finetuning_data.py から拝借・調整) ---
def get_record_details_for_prompt(record_id):
    if not BIB_DATA:
        # この関数が呼ばれる時点ではBIB_DATAはロードされているはず
        return "情報取得エラー: BIB_DATA未ロード"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    # record.yml の構造に合わせてキーを調整
    # siamese_model_pytorch/prepare_finetuning_data.py のものとは異なり、
    # 'bib1_title' のようなプレフィックスがないキーを想定 (record.yml の構造による)
    # 以下は一般的なキーの例。実際の record.yml の構造に合わせて調整が必要。
    title = bib_details.get("title", bib_details.get("bib1_title", "タイトル不明"))
    authors_list = bib_details.get("author", bib_details.get("bib1_author", []))
    if isinstance(authors_list, list):
        authors_str = ", ".join(authors_list) if authors_list else "著者不明"
    elif isinstance(authors_list, str):  # authorが文字列の場合も考慮
        authors_str = authors_list if authors_list else "著者不明"
    else:
        authors_str = "著者不明"

    publisher = bib_details.get("publisher", bib_details.get("bib1_publisher", "出版社不明"))
    pubdate = bib_details.get("pubdate", bib_details.get("bib1_pubdate", "出版日不明"))

    # 他にも重要なフィールドがあれば追加
    # entrytype = bib_details.get("entrytype", "タイプ不明")
    # journal = bib_details.get("journal", bib_details.get("bib1_journal", ""))

    details = f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"
    # if journal:
    #     details += f"\n雑誌名: {journal}"
    return details


# --- LLM呼び出し関連関数 ---


def get_llm_evaluation_for_pair(record_id_1, record_id_2, model_id, openai_client):
    """
    指定されたモデルIDを使用し、2つの書誌レコードのペアが同一かLLMに判定させる。
    結果はキャッシュされる。
    Returns:
        tuple: (is_similar (bool|None), similarity_score (float|None), error_message (str|None))
               エラー時は is_similar と similarity_score が None になる。
    """
    global CACHE_DATA
    cache_key = f"{record_id_1}_{record_id_2}_{model_id}"
    error_msg = None

    if cache_key in CACHE_DATA:
        cached_item = CACHE_DATA[cache_key]
        # キャッシュされたデータ構造を想定してアクセス
        # ここでは {'is_similar': bool, 'score': float} を想定
        if isinstance(cached_item, dict) and "is_similar" in cached_item and "score" in cached_item:
            print(f"  キャッシュヒット: ペア ({record_id_1}, {record_id_2}), モデル {model_id}")
            return cached_item["is_similar"], cached_item["score"], None
        else:
            # 予期しないキャッシュ形式の場合は、キャッシュを無視してAPIを呼び出す
            print(f"警告: キャッシュキー {cache_key} のデータ形式が不正です。APIを呼び出します。")

    if not openai_client:
        return None, None, "OpenAI APIクライアントが初期化されていません。"

    bib_info_1 = get_record_details_for_prompt(record_id_1)
    bib_info_2 = get_record_details_for_prompt(record_id_2)

    if "情報取得エラー" in bib_info_1 or "書誌情報なし" in bib_info_1:
        return None, None, f"レコード {record_id_1} の情報取得に失敗: {bib_info_1}"
    if "情報取得エラー" in bib_info_2 or "書誌情報なし" in bib_info_2:
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
        # レート制限を考慮して少し待機 (必要に応じて調整)
        time.sleep(0.1)  # 0.1秒待機

        completion = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,  # 再現性のため温度は低めに設定
            max_tokens=50,  # 回答は短いはず
        )
        response_text = completion.choices[0].message.content.strip()

        # レスポンスのパース (evaluate_pairs_with_openai_async.py の parse_llm_response を参考にする)
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

        # パースロジックの改善 (より頑健に)
        parsed_is_similar = None
        if "はい" in is_similar_str:
            parsed_is_similar = True
        elif "いいえ" in is_similar_str:
            parsed_is_similar = False
        else:  # 予期しない回答形式の場合
            # 応答全体からキーワードで探す試み
            if "はい" in response_text and not "いいえ" in response_text:  # "はい"だけある
                parsed_is_similar = True
            elif "いいえ" in response_text and not "はい" in response_text:  # "いいえ"だけある
                parsed_is_similar = False
            # else: "はい"と"いいえ"両方ある場合や、どちらもない場合はNoneのまま

        parsed_similarity_score = None
        if similarity_score_str:
            try:
                parsed_similarity_score = float(similarity_score_str)
                if not (0.0 <= parsed_similarity_score <= 1.0):
                    # スコアが範囲外の場合はNone扱いにするか、警告を出す
                    print(
                        f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコア '{parsed_similarity_score}' が範囲外です。応答: {response_text}"
                    )
                    # parsed_similarity_score = None # または補正する
            except ValueError:
                print(
                    f"警告: ペア ({record_id_1}, {record_id_2}) の類似度スコアが数値に変換できません: '{similarity_score_str}'。応答: {response_text}"
                )

        # 判定とスコアのいずれかが取得できなかった場合のフォールバック
        if parsed_is_similar is None and parsed_similarity_score is not None:
            # スコアがあるが判定がない場合、スコアから判定を推測する (例: 0.5以上ならTrue)
            if parsed_similarity_score >= 0.5:  # しきい値は要検討
                parsed_is_similar = True
            else:
                parsed_is_similar = False
            print(
                f"情報: ペア ({record_id_1}, {record_id_2}) の判定をスコアから推測しました: {parsed_is_similar} (スコア: {parsed_similarity_score})"
            )

        elif parsed_is_similar is not None and parsed_similarity_score is None:
            # 判定があるがスコアがない場合、判定からスコアを推測する
            if parsed_is_similar:
                parsed_similarity_score = 1.0
            else:
                parsed_similarity_score = 0.0
            print(
                f"情報: ペア ({record_id_1}, {record_id_2}) のスコアを判定から推測しました: {parsed_similarity_score} (判定: {parsed_is_similar})"
            )

        if parsed_is_similar is None:  # どうしても判定が不明な場合
            return None, None, f"LLMの応答から判定（はい/いいえ）を抽出できませんでした。応答: {response_text}"
        if parsed_similarity_score is None:  # スコアが不明な場合
            return parsed_is_similar, None, f"LLMの応答から類似度スコアを抽出できませんでした。応答: {response_text}"

        if (
            parsed_is_similar is not None and parsed_similarity_score is not None and error_msg is None
        ):  # エラーがない正常な結果のみキャッシュ
            CACHE_DATA[cache_key] = {"is_similar": parsed_is_similar, "score": parsed_similarity_score}
            # print(f"  結果をキャッシュに保存: {cache_key}") # 詳細ログが必要な場合

        return parsed_is_similar, parsed_similarity_score, None

    except Exception as e:
        error_msg = f"OpenAI API呼び出し中にエラーが発生 (ペア: {record_id_1}, {record_id_2}, モデル: {model_id}): {e}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        return None, None, error_msg


# --- 評価指標計算関連関数 ---


def format_clusters_with_details(predicted_clusters, bib_data_dict):
    """
    予測されたクラスタ情報を、レコード詳細を含めて整形する。
    Args:
        predicted_clusters (dict): {record_id: cluster_label}
        bib_data_dict (dict): {record_id: bib_details_dict} グローバルなBIB_DATAを期待
    Returns:
        dict: {cluster_label: [{"record_id": record_id, "details": formatted_bib_string}, ...]}
    """
    grouped_clusters = {}
    for record_id, cluster_label in predicted_clusters.items():
        if cluster_label not in grouped_clusters:
            grouped_clusters[cluster_label] = []

        details_str = get_record_details_for_prompt(record_id)  # 既存関数で詳細取得
        grouped_clusters[cluster_label].append({"record_id": record_id, "details": details_str})
    # クラスタラベルでソートして、IDリストもレコードIDでソートしておくと見やすいかも
    sorted_grouped_clusters = {}
    for label in sorted(grouped_clusters.keys()):
        sorted_records = sorted(grouped_clusters[label], key=lambda x: x["record_id"])
        sorted_grouped_clusters[str(label)] = sorted_records  # JSONのキーは文字列にする
    return sorted_grouped_clusters


def evaluate_model_on_pairs(model_id, pairs_to_evaluate, all_record_ids_in_pairs, openai_client):
    """
    指定されたモデルで全評価ペアを判定し、結果を収集する。
    Args:
        model_id (str): 評価に使用するLLMのモデルID。
        pairs_to_evaluate (list of tuples): [(record_id_1, record_id_2), ...] 評価対象のペア。
        all_record_ids_in_pairs (set): 評価ペアに含まれる全てのユニークなレコードIDのセット。
                                      クラスタリング評価で全レコードをカバーするために使用。
    Returns:
        dict: 以下のキーを含む辞書
            'predictions': list of bool (各ペアのLLMによる予測ラベル True=similar)
            'ground_truths': list of bool (各ペアの正解ラベル True=similar)
            'predicted_positive_pairs': list of tuples (LLMがsimilarと判定したペア)
            'llm_scores': list of float (各ペアのLLMによる類似度スコア)
            'errors': list of tuples ((id1, id2, error_msg)) 判定エラーが発生したペア
            'processed_pairs': list of tuples ((id1, id2)) 処理された全ペア
    """
    predictions = []
    ground_truths = []
    predicted_positive_pairs = []
    llm_scores = []
    errors = []
    processed_pairs = []

    print(f"\nモデル '{model_id}' で {len(pairs_to_evaluate)} ペアの評価を開始します...")
    for i, (r_id1, r_id2) in enumerate(pairs_to_evaluate):
        processed_pairs.append((r_id1, r_id2))
        # 正解ラベルの決定: 同じcluster_idを持つか否か
        gt_cluster1 = GROUND_TRUTH_CLUSTERS.get(r_id1)
        gt_cluster2 = GROUND_TRUTH_CLUSTERS.get(r_id2)

        # どちらかのレコードのクラスタ情報がない場合は、正解判定不能ペアとして扱うか検討
        # ここでは、両方に有効なクラスタIDがあり、かつそれらが一致する場合のみTrueとする
        is_truly_similar = False
        if (
            gt_cluster1 is not None
            and gt_cluster2 is not None
            and not gt_cluster1.startswith("gt_orphan_")
            and not gt_cluster2.startswith("gt_orphan_")
            and gt_cluster1 == gt_cluster2
        ):
            is_truly_similar = True

        ground_truths.append(is_truly_similar)

        llm_is_similar, llm_score, error_msg = get_llm_evaluation_for_pair(r_id1, r_id2, model_id, openai_client)

        if error_msg:
            errors.append(((r_id1, r_id2), error_msg))
            # エラーの場合、予測はFalse扱いとし、スコアは0.0とする (あるいはNoneのまま別途処理)
            predictions.append(False)  # またはNoneを追加して後でフィルタリング
            llm_scores.append(0.0)  # またはNone
            print(f"  ペア ({r_id1}, {r_id2}) 評価エラー: {error_msg}")
        else:
            predictions.append(llm_is_similar)
            llm_scores.append(
                llm_score if llm_score is not None else (1.0 if llm_is_similar else 0.0)
            )  # スコアNoneなら判定から補完
            if llm_is_similar:
                predicted_positive_pairs.append((r_id1, r_id2))

        if (i + 1) % 50 == 0:  # 50ペアごとに進捗表示
            print(f"  {i+1}/{len(pairs_to_evaluate)} ペア処理完了...")
            save_cache()  # 定期的なキャッシュ保存

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
    """ペアごとの評価指標と混合行列を計算する"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average="binary", zero_division=0
    )
    # average='binary' は Positiveクラスに対する指標。TrueがPositive。
    # もしmulti-classの場合は average='weighted' や 'macro' も検討。

    # 混合行列の計算: labels=[False, True] を指定して順序を明確にする
    # True: 一致ペア (Positive), False: 不一致ペア (Negative)
    # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
    cm = confusion_matrix(ground_truths, predictions, labels=[False, True])
    # labels=[False, True] を指定した場合、cm は通常2x2。
    # ground_truths や predictions が空、または単一クラスのみの場合の挙動も考慮。
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1 and len(set(ground_truths)) == 1:  # 全て同じ正解ラベルで、予測も全てそのラベルなど
        label_val = list(set(ground_truths))[0]
        if label_val == False:  # All TN
            tn, fp, fn, tp = cm.item(), 0, 0, 0
        else:  # All TP
            tn, fp, fn, tp = 0, 0, 0, cm.item()
    else:  # 予期しないケースや、より複雑な片側だけの予測など。
        print(
            f"警告: {model_name} の混合行列が予期せぬ形状 {cm.shape} / サイズ {cm.size} です。 TN/FP/FN/TP は0として扱われる可能性があります。"
        )
        # 安全のため、明示的に初期化し、可能な範囲でcmから取得試行
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):  # 再度確認
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        # ここはさらに詳細な分岐が必要になる場合もあるが、多くは labels 指定でカバーされるはず

    print(f"\n--- {model_name} ペアワイズ評価指標 ---")
    print(f"  混合行列 (Positive: 一致ペア, Negative: 不一致ペア):")
    print(
        f"  +----------------+-----------------+-----------------+\n  | {'Ground Truth':^14} | {'Predicted: Pos':^15} | {'Predicted: Neg':^15} |"
    )
    print(f"  +================+=================+=================+")
    print(f"  | {'Positive':<14} | TP: {tp:<12d} | FN: {fn:<12d} |")
    print(
        f"  +----------------+-----------------+-----------------+\n  | {'Negative':<14} | FP: {fp:<12d} | TN: {tn:<12d} |"
    )
    print(
        f"  +----------------+-----------------+-----------------+\n  適合率 (Precision): {precision:.4f} (TP / (TP + FP))"
    )
    print(f"  再現率 (Recall):    {recall:.4f} (TP / (TP + FN))")
    print(f"  F1スコア:           {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1_score": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def form_predicted_clusters(positive_pairs, all_record_ids):
    """
    LLMが「はい」と判定したペアから予測クラスタを形成する。
    Args:
        positive_pairs (list of tuples): LLMがsimilarと判定したペア [(id1, id2), ...]
        all_record_ids (set or list): 評価対象となった全てのユニークなレコードID。
                                    これにより、どのクラスタにも属さない孤立ノードも考慮できる。
    Returns:
        dict: record_id -> predicted_cluster_label (int)
    """
    graph = nx.Graph()
    graph.add_nodes_from(all_record_ids)  # まず全てのレコードIDをノードとして追加
    graph.add_edges_from(positive_pairs)

    predicted_cluster_map = {}
    cluster_label_counter = 0
    for component_nodes in nx.connected_components(graph):
        for node in component_nodes:
            predicted_cluster_map[node] = cluster_label_counter
        cluster_label_counter += 1

    # どの連結成分にも属さなかった（つまりpositive_pairsに含まれなかった）ノードにも
    # ユニークなクラスタラベルを割り当てる (既にadd_nodes_fromで追加されているので、
    # connected_components が各孤立ノードを単一要素の成分として返すはずだが念のため)
    for record_id in all_record_ids:
        if record_id not in predicted_cluster_map:
            predicted_cluster_map[record_id] = cluster_label_counter
            cluster_label_counter += 1

    return predicted_cluster_map


def calculate_clustering_metrics(true_cluster_map, pred_cluster_map, all_record_ids, model_name=""):
    """
    クラスタリング評価指標 (ARI, NMI, V-measure) を計算する。
    Args:
        true_cluster_map (dict): record_id -> true_cluster_id
        pred_cluster_map (dict): record_id -> predicted_cluster_label
        all_record_ids (list or set): 評価対象の全レコードID。ラベルリストの順序を揃えるため。
        model_name (str): ログ表示用のモデル名
    Returns:
        dict: ari, nmi, v_measure をキーとする辞書
    """
    # sklearnの関数はラベルのリストを期待するので、all_record_idsの順序でラベルリストを作成
    # true_cluster_map には gt_orphan_X のような仮IDが含まれる場合があるが、
    # 評価指標の計算上は異なるIDとして扱われれば問題ない。
    true_labels = [true_cluster_map.get(rid, f"missing_gt_{rid}") for rid in all_record_ids]
    pred_labels = [pred_cluster_map.get(rid, -1) for rid in all_record_ids]  # 予測がない場合は-1 (孤立ノード)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    # v_measure_score は labels_true, labels_pred の順
    # homogeneity, completeness, v_measure = v_measure_score(true_labels, pred_labels, beta=1.0) # 元の行
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)  # beta=1.0 は v_measure_score のデフォルト
    # beta=1.0 で F-measure と同様に Homogeneity と Completeness の調和平均 (v_measure_scoreのデフォルト動作)

    print(f"\n--- {model_name} クラスタリング評価指標 ---")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Completeness: {completeness:.4f}")
    print(f"  V-measure: {v_measure:.4f}")

    return {"ari": ari, "nmi": nmi, "homogeneity": homogeneity, "completeness": completeness, "v_measure": v_measure}


# --- ヘルパー関数 (ファイル名サニタイズ用) ---
def sanitize_model_name_for_filename(model_name):
    """モデル名をファイル名として安全な形式に変換する。"""
    if not model_name:
        return "unknown_model"
    # ファイル名に使えない文字をアンダースコアに置換
    # Windowsでは \\ / : * ? \" < > | が使えない。Unix系では / と NUL。
    # OpenAIのモデルIDは通常 : を含むので、これを置換対象の主とする。
    # 他にも必要に応じて追加。
    sanitized_name = model_name.replace(":", "_").replace("/", "-")
    # 長すぎる場合は短縮することも検討できるが、ここでは行わない
    return sanitized_name


# --- メイン処理 ---


def main(args):
    """
    ファインチューニング前後のモデル性能評価を実行するメイン関数。
    """
    global RECORD_YAML_PATH, EVALUATION_PAIRS_CSV_PATH, OUTPUT_DIR
    global PERFORMANCE_REPORT_PATH, DETAILED_RESULTS_CSV_PATH, CACHE_FILE_PATH
    global client

    # パス設定
    if args.record_yaml_dir:
        record_yaml_base_dir = args.record_yaml_dir
    else:
        record_yaml_base_dir = os.path.join(PROJECT_ROOT_ASSUMED, DEFAULT_RECORD_YAML_DIR_RELATIVE_TO_PROJECT_ROOT)
    RECORD_YAML_PATH = os.path.join(record_yaml_base_dir, args.record_yaml_filename)

    if args.eval_pairs_csv_dir:
        eval_pairs_base_dir = args.eval_pairs_csv_dir
    else:
        eval_pairs_base_dir = os.path.join(BASE_DIR, DEFAULT_EVAL_PAIRS_CSV_DIR_RELATIVE_TO_BASE_DIR)
    EVALUATION_PAIRS_CSV_PATH = os.path.join(eval_pairs_base_dir, args.eval_pairs_csv_filename)

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(BASE_DIR, DEFAULT_OUTPUT_DIR_RELATIVE_TO_BASE_DIR)

    # OUTPUT_DIR が決定した後にディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # OUTPUT_DIR を使って他のパスを設定
    model_id_before_ft = args.model_before_ft
    model_id_after_ft = args.model_after_ft

    # モデル名に基づいてファイル名を生成
    model_before_ft_sanitized = sanitize_model_name_for_filename(model_id_before_ft)
    model_after_ft_sanitized = sanitize_model_name_for_filename(model_id_after_ft)

    # 既存のファイル名定義を上書き
    report_filename = (
        f"finetuning_performance_report_before-{model_before_ft_sanitized}_after-{model_after_ft_sanitized}.txt"
    )
    detailed_csv_filename = (
        f"detailed_evaluation_results_before-{model_before_ft_sanitized}_after-{model_after_ft_sanitized}.csv"
    )
    # cache_filename = f"llm_api_cache_before-{model_before_ft_sanitized}_after-{model_after_ft_sanitized}.json"  # キャッシュファイルもモデル別に

    PERFORMANCE_REPORT_PATH = os.path.join(OUTPUT_DIR, report_filename)
    DETAILED_RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, detailed_csv_filename)
    CACHE_FILE_PATH = os.path.join(OUTPUT_DIR, "llm_api_cache.json")  # グローバル変数を更新 (固定名に戻す)

    try:
        client = OpenAI()
    except Exception as e:
        print(f"OpenAI APIクライアントの初期化に失敗しました。環境変数 OPENAI_API_KEY を確認してください。エラー: {e}")
        client = None

    load_cache()

    try:
        print("評価処理を開始します...")
        print(
            f"使用するパス:\n  書誌YAML: {RECORD_YAML_PATH}\n  評価ペアCSV: {EVALUATION_PAIRS_CSV_PATH}\n  出力ディレクトリ: {OUTPUT_DIR}"
        )
        print(f"ファインチューニング前のモデルID: {model_id_before_ft}")
        print(f"ファインチューニング後のモデルID: {model_id_after_ft}")

        if not client:
            print("OpenAI APIクライアントが利用できません。処理を中断します。")
            return

        if not load_bib_data_and_gt_clusters(RECORD_YAML_PATH):
            return
        evaluation_pairs = load_evaluation_pairs(EVALUATION_PAIRS_CSV_PATH)
        if not evaluation_pairs:
            return

        all_record_ids_in_pairs = set(r_id for pair in evaluation_pairs for r_id in pair)
        all_record_ids_list = sorted(list(all_record_ids_in_pairs))
        results_df_data = []

        print("\n===== ファインチューニング「前」のモデル性能評価 =====")
        results_before = evaluate_model_on_pairs(model_id_before_ft, evaluation_pairs, all_record_ids_in_pairs, client)
        pairwise_metrics_before = calculate_pairwise_metrics(
            results_before["ground_truths"], results_before["predictions"], model_id_before_ft
        )
        pred_clusters_before = form_predicted_clusters(results_before["predicted_positive_pairs"], all_record_ids_list)
        clustering_metrics_before = calculate_clustering_metrics(
            GROUND_TRUTH_CLUSTERS, pred_clusters_before, all_record_ids_list, model_id_before_ft
        )

        print("\n===== ファインチューニング「後」のモデル性能評価 =====")
        results_after = evaluate_model_on_pairs(model_id_after_ft, evaluation_pairs, all_record_ids_in_pairs, client)
        pairwise_metrics_after = calculate_pairwise_metrics(
            results_after["ground_truths"], results_after["predictions"], model_id_after_ft
        )
        pred_clusters_after = form_predicted_clusters(results_after["predicted_positive_pairs"], all_record_ids_list)
        clustering_metrics_after = calculate_clustering_metrics(
            GROUND_TRUTH_CLUSTERS, pred_clusters_after, all_record_ids_list, model_id_after_ft
        )

        for i, (r_id1, r_id2) in enumerate(results_before["processed_pairs"]):
            results_df_data.append(
                {
                    "record_id_1": r_id1,
                    "record_id_2": r_id2,
                    "ground_truth_similar": results_before["ground_truths"][i],
                    "predicted_similar_before": results_before["predictions"][i],
                    "score_before": results_before["llm_scores"][i],
                    "error_before": next(
                        (err[1] for pair_ids, err in results_before["errors"] if pair_ids == (r_id1, r_id2)), None
                    ),
                    "predicted_similar_after": (
                        results_after["predictions"][i] if i < len(results_after["predictions"]) else None
                    ),
                    "score_after": results_after["llm_scores"][i] if i < len(results_after["llm_scores"]) else None,
                    "error_after": next(
                        (err[1] for pair_ids, err in results_after["errors"] if pair_ids == (r_id1, r_id2)), None
                    ),
                }
            )
        detailed_results_df = pd.DataFrame(results_df_data)
        try:
            detailed_results_df.to_csv(DETAILED_RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"\n詳細な評価結果を {DETAILED_RESULTS_CSV_PATH} に保存しました。")
        except Exception as e:
            print(f"エラー: 詳細な評価結果のCSV保存に失敗: {e}")

        # === 全レコードペアを対象とした推論的ペアワイズ評価 ===
        print("\n\n===== 全レコードペアを対象とした推論的ペアワイズ評価 =====")
        all_record_ids_global = sorted(list(BIB_DATA.keys()))  # YAML由来の全てのレコードID

        pairwise_metrics_all_before = None
        pairwise_metrics_all_after = None

        if len(all_record_ids_global) < 2:
            print("評価対象のレコードが2件未満のため、全ペア評価はスキップします。")
            # レポート用にダミーの指標を初期化 (あるいはレポート側で存在チェック)
            empty_metrics = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "precision": 0, "recall": 0, "f1_score": 0}
            pairwise_metrics_all_before = empty_metrics
            pairwise_metrics_all_after = empty_metrics
        else:
            # 予測クラスタを全レコードスコープで構築
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

            for r_id1, r_id2 in itertools.combinations(all_record_ids_global, 2):
                # 正解ラベル
                gt_c1 = GROUND_TRUTH_CLUSTERS.get(str(r_id1))
                gt_c2 = GROUND_TRUTH_CLUSTERS.get(str(r_id2))
                is_truly_similar = (
                    gt_c1 is not None
                    and gt_c2 is not None
                    and not str(gt_c1).startswith("gt_orphan_")
                    and not str(gt_c2).startswith("gt_orphan_")
                    and str(gt_c1) == str(gt_c2)
                )
                all_pairs_true_labels.append(is_truly_similar)

                # ファインチューニング前モデルの予測ラベル (クラスタベース)
                pred_c1_b = pred_clusters_all_scope_before.get(str(r_id1))
                pred_c2_b = pred_clusters_all_scope_before.get(str(r_id2))
                all_pairs_pred_labels_before.append(pred_c1_b is not None and pred_c1_b == pred_c2_b)

                # ファインチューニング後モデルの予測ラベル (クラスタベース)
                pred_c1_a = pred_clusters_all_scope_after.get(str(r_id1))
                pred_c2_a = pred_clusters_all_scope_after.get(str(r_id2))
                all_pairs_pred_labels_after.append(pred_c1_a is not None and pred_c1_a == pred_c2_a)

            print("全ペアのラベル生成完了。評価指標を計算します...")

            if not all_pairs_true_labels:
                print("全ペア評価のためのラベルが生成できませんでした。")
                empty_metrics = {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "precision": 0, "recall": 0, "f1_score": 0}
                pairwise_metrics_all_before = empty_metrics
                pairwise_metrics_all_after = empty_metrics
            else:
                pairwise_metrics_all_before = calculate_pairwise_metrics(
                    all_pairs_true_labels, all_pairs_pred_labels_before, f"{model_id_before_ft} (全ペア推論)"
                )
                pairwise_metrics_all_after = calculate_pairwise_metrics(
                    all_pairs_true_labels, all_pairs_pred_labels_after, f"{model_id_after_ft} (全ペア推論)"
                )

        # レポート内容の更新 (K近傍 + 全ペア)
        report_content = f"""# ファインチューニング性能評価レポート
日付: {time.strftime("%Y-%m-%d %H:%M:%S")}
## 評価対象
- 書誌データ: {RECORD_YAML_PATH}
- K近傍ペアリスト (LLM直接評価対象): {EVALUATION_PAIRS_CSV_PATH} ({len(evaluation_pairs)} ペア)
- 全レコード数 (全ペア推論評価の母数): {len(all_record_ids_global)}

## K近傍ペア評価 (LLMが直接判定したペアに基づく)
### ファインチューニング前モデル ({model_id_before_ft})
- 混合行列 (Positive: 一致ペア, Negative: 不一致ペア):
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_before['tp']:<18d} | FN: {pairwise_metrics_before['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_before['fp']:<18d} | TN: {pairwise_metrics_before['tn']:<18d}
- 適合率: {pairwise_metrics_before['precision']:.4f}, 再現率: {pairwise_metrics_before['recall']:.4f}, F1: {pairwise_metrics_before['f1_score']:.4f}
- エラー数: {len(results_before['errors'])}

### クラスタリング評価 (K近傍ペアのLLM判定結果から形成されたクラスタ - 評価対象: K近傍ペアに含まれるID群)
- Adjusted Rand Index (ARI): {clustering_metrics_before['ari']:.4f}
- Normalized Mutual Information (NMI): {clustering_metrics_before['nmi']:.4f}
- Homogeneity: {clustering_metrics_before['homogeneity']:.4f}, Completeness: {clustering_metrics_before['completeness']:.4f}, V-measure: {clustering_metrics_before['v_measure']:.4f}

### ファインチューニング後モデル ({model_id_after_ft})
- 混合行列 (Positive: 一致ペア, Negative: 不一致ペア):
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_after['tp']:<18d} | FN: {pairwise_metrics_after['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_after['fp']:<18d} | TN: {pairwise_metrics_after['tn']:<18d}
- 適合率: {pairwise_metrics_after['precision']:.4f}, 再現率: {pairwise_metrics_after['recall']:.4f}, F1: {pairwise_metrics_after['f1_score']:.4f}
- エラー数: {len(results_after['errors'])}

### クラスタリング評価 (K近傍ペアのLLM判定結果から形成されたクラスタ - 評価対象: K近傍ペアに含まれるID群)
- Adjusted Rand Index (ARI): {clustering_metrics_after['ari']:.4f}
- Normalized Mutual Information (NMI): {clustering_metrics_after['nmi']:.4f}
- Homogeneity: {clustering_metrics_after['homogeneity']:.4f}, Completeness: {clustering_metrics_after['completeness']:.4f}, V-measure: {clustering_metrics_after['v_measure']:.4f}
"""
        # pairwise_metrics_all_before と pairwise_metrics_all_after が None でないことを確認してレポートに追加
        if pairwise_metrics_all_before and pairwise_metrics_all_after:
            report_content += f""" 

## 全ペア推論評価 (LLM判定から形成されたクラスタに基づき、全レコード間のペアを推論)
### ファインチューニング前モデル ({model_id_before_ft} - 全ペア推論)
- 混合行列 (Positive: 一致ペア, Negative: 不一致ペア):
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_all_before['tp']:<18d} | FN: {pairwise_metrics_all_before['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_all_before['fp']:<18d} | TN: {pairwise_metrics_all_before['tn']:<18d}
- 適合率: {pairwise_metrics_all_before['precision']:.4f}, 再現率: {pairwise_metrics_all_before['recall']:.4f}, F1: {pairwise_metrics_all_before['f1_score']:.4f}

### ファインチューニング後モデル ({model_id_after_ft} - 全ペア推論)
- 混合行列 (Positive: 一致ペア, Negative: 不一致ペア):
    予測ラベル     |  Predicted: Positive | Predicted: Negative
  ----------------|----------------------|----------------------
  Actual: Positive  | TP: {pairwise_metrics_all_after['tp']:<18d} | FN: {pairwise_metrics_all_after['fn']:<18d}
  Actual: Negative  | FP: {pairwise_metrics_all_after['fp']:<18d} | TN: {pairwise_metrics_all_after['tn']:<18d}
- 適合率: {pairwise_metrics_all_after['precision']:.4f}, 再現率: {pairwise_metrics_all_after['recall']:.4f}, F1: {pairwise_metrics_all_after['f1_score']:.4f}
"""
        try:
            with open(PERFORMANCE_REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"パフォーマンスレポートを {PERFORMANCE_REPORT_PATH} に保存しました。")
        except Exception as e:
            print(f"エラー: パフォーマンスレポートの保存に失敗: {e}")

        try:
            if "pred_clusters_before" in locals() and pred_clusters_before:
                formatted_clusters_before = format_clusters_with_details(pred_clusters_before, BIB_DATA)
                # モデル名を含めたファイル名に変更
                filename_before_detailed = f"grouped_clusters_with_details_before-{model_before_ft_sanitized}.json"
                path_before_detailed = os.path.join(OUTPUT_DIR, filename_before_detailed)
                with open(path_before_detailed, "w", encoding="utf-8") as f:
                    json.dump(formatted_clusters_before, f, ensure_ascii=False, indent=4)
                print(f"詳細なファインチューニング前予測クラスタ情報を {path_before_detailed} に保存しました。")
            if "pred_clusters_after" in locals() and pred_clusters_after:
                formatted_clusters_after = format_clusters_with_details(pred_clusters_after, BIB_DATA)
                # モデル名を含めたファイル名に変更
                filename_after_detailed = f"grouped_clusters_with_details_after-{model_after_ft_sanitized}.json"
                path_after_detailed = os.path.join(OUTPUT_DIR, filename_after_detailed)
                with open(path_after_detailed, "w", encoding="utf-8") as f:
                    json.dump(formatted_clusters_after, f, ensure_ascii=False, indent=4)
                print(f"詳細なファインチューニング後予測クラスタ情報を {path_after_detailed} に保存しました。")
        except Exception as e:
            print(f"エラー: 詳細な予測クラスタ情報のJSON保存に失敗: {e}")
        print("\n評価処理が完了しました。")
    finally:
        save_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ファインチューニング前後のLLM性能を評価するスクリプト")
    parser.add_argument(
        "--record_yaml_dir",
        type=str,
        default=None,
        help=f"書誌データYAMLファイルが格納されているディレクトリのパス。デフォルト: PROJECT_ROOT/{DEFAULT_RECORD_YAML_DIR_RELATIVE_TO_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--record_yaml_filename",
        type=str,
        default=DEFAULT_RECORD_YAML_FILENAME,
        help=f"書誌データYAMLファイル名。デフォルト: {DEFAULT_RECORD_YAML_FILENAME}",
    )
    parser.add_argument(
        "--eval_pairs_csv_dir",
        type=str,
        default=None,
        help=f"評価ペアCSVファイルが格納されているディレクトリのパス。デフォルト: スクリプトの場所/{DEFAULT_EVAL_PAIRS_CSV_DIR_RELATIVE_TO_BASE_DIR}",
    )
    parser.add_argument(
        "--eval_pairs_csv_filename",
        type=str,
        default=DEFAULT_EVAL_PAIRS_CSV_FILENAME,
        help=f"評価ペアCSVファイル名。デフォルト: {DEFAULT_EVAL_PAIRS_CSV_FILENAME}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"評価結果を保存するディレクトリのパス。デフォルト: スクリプトの場所/{DEFAULT_OUTPUT_DIR_RELATIVE_TO_BASE_DIR}",
    )
    parser.add_argument(
        "--model_before_ft",
        type=str,
        default=DEFAULT_MODEL_ID_BEFORE_FINETUNING,
        help=f"ファインチューニング前のモデルID。デフォルト: {DEFAULT_MODEL_ID_BEFORE_FINETUNING}",
    )
    parser.add_argument("--model_after_ft", type=str, required=True, help="ファインチューニング後のモデルID (必須)")

    parsed_args = parser.parse_args()
    main(parsed_args)
