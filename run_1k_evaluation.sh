#!/bin/bash

# このスクリプトは、各データセットの1k評価を順番に実行します。
# 途中でエラーが発生した場合は、その時点でスクリプトは停止します。

echo "===== Starting All 1k Evaluations ====="
echo ""

# --- 1. Music 1k Evaluation ---
echo "===== (1/4) Starting Music 1k Evaluation ====="
python3 siamese_model_pytorch/evaluate_zeroshot_fewshot_performance_async.py \
--pairs_csv results_music/run_1k_music/llm_evaluation_pairs/candidate_pairs_from_record_k10.csv \
--ground_truth_yaml benchmark/music_lepizig_20241024/1k/record.yml \
--data_type music \
--few_shot_data results_music/run_2k_music/fewshot_examples_hybrid_k10.json && \

echo "===== Music 1k Evaluation COMPLETE ====="
echo "" && \

# --- 2. Person 1k Evaluation ---
echo "===== (2/4) Starting Person 1k Evaluation ====="
python3 siamese_model_pytorch/evaluate_zeroshot_fewshot_performance_async.py \
--pairs_csv results_person/run_1k_person/llm_evaluation_pairs/candidate_pairs_from_record_k10.csv \
--ground_truth_yaml benchmark/persons_lepizig_20241024/1k/record.yml \
--data_type person \
--few_shot_data results_person/run_2k_person/fewshot_examples_hybrid.json && \

echo "===== Person 1k Evaluation COMPLETE ====="
echo "" && \

# --- 3. Walmart-Amazon 1k Evaluation ---
echo "===== (3/4) Starting Walmart-Amazon 1k Evaluation ====="
python3 siamese_model_pytorch/evaluate_zeroshot_fewshot_performance_async.py \
--pairs_csv results_walmart-amazon/run_1k_walmart-amazon/llm_evaluation_pairs/candidate_pairs_from_test_subset_1_k10.csv \
--ground_truth_yaml benchmark/product_walmart_amazon/refined/topN_large_clusters/1k/test_subset_1.yml \
--data_type product \
--few_shot_data results_walmart-amazon/run_2k_walmart-amazon/fewshot_examples_hybrid.json && \

echo "===== Walmart-Amazon 1k Evaluation COMPLETE ====="
echo "" && \

# --- 4. WDC 1k Evaluation ---
echo "===== (4/4) Starting WDC 1k Evaluation ====="
python3 siamese_model_pytorch/evaluate_zeroshot_fewshot_performance_async.py \
--pairs_csv results_wdc/run_1k_wdc/llm_evaluation_pairs/candidate_pairs_from_test_subset_1_k10.csv \
--ground_truth_yaml benchmark/product_wdc/wdc_large/large_clusters/1k/test_subset_1.yml \
--data_type product \
--few_shot_data results_wdc/run_2k_wdc/fewshot_examples_hybrid.json && \

echo "===== WDC 1k Evaluation COMPLETE ====="
echo "" && \

echo "===== All 1k Evaluations Have Finished Successfully! ====="
