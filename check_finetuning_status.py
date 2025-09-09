#!/usr/bin/env python3
"""
Fine-tuning Job Creator, Status Checker, and Recorder
Usage: python3 check_finetuning_status.py <path_to_data_directory>
Example: python3 check_finetuning_status.py results_wdc/run_2k_ite1_wdc/evaluation_results/
"""

import openai
from datetime import datetime
import glob
import os
import re
import time
import argparse

def create_finetuning_jobs_from_files(client, data_directory, record_file_path):
    """
    指定されたディレクトリ内のデータファイルからFine-tuningジョブを作成し、
    job_configsリストと、記録ファイル用の初期コンテンツを返す
    """
    search_path = os.path.join(data_directory, 'ft_data_*ite*.jsonl')
    data_files = glob.glob(search_path)
    
    if not data_files:
        print(f"No data files found in {data_directory}")
        return [], ""

    # データタイプをディレクトリ名から推測 (例: "results_wdc" -> "wdc")
    datatype_match = re.search(r'results_(\w+)', data_directory)
    datatype = datatype_match.group(1) if datatype_match else "unknown"

    job_configs = []
    record_file_content = f"Fine-tuning Jobs Record ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"

    for filepath in data_files:
        filename = os.path.basename(filepath)
        match = re.search(r'ft_data_(.+)\.jsonl', filename)
        if not match:
            continue
            
        strategy = match.group(1)
        
        try:
            with open(filepath, 'r') as f:
                samples = len(f.readlines())
        except Exception as e:
            print(f"Warning: Could not read {filepath} to get sample count. Using 'unknown'. Error: {e}")
            samples = "unknown"
            
        suffix = f"{datatype}-product-{strategy}-{samples}"
        
        print(f"\n=== Processing: {filename} ===")
        
        try:
            # 1. ファイルアップロード
            print(f"Uploading {strategy} training file...")
            with open(filepath, 'rb') as f:
                training_file = client.files.create(file=f, purpose='fine-tune')
            print(f"  > File uploaded: {training_file.id}")

            # 2. Fine-tuningジョブ作成
            print("Creating fine-tuning job...")
            job = client.fine_tuning.jobs.create(
                training_file=training_file.id,
                model='gpt-4o-mini-2024-07-18',
                suffix=suffix,
                hyperparameters={'n_epochs': 3, 'learning_rate_multiplier': 1.8, 'batch_size': 4}
            )
            print(f"  > Job created: {job.id} (Status: {job.status})")
            
            job_configs.append({
                'strategy': strategy,
                'job_id': job.id,
                'training_file': training_file.id,
                'suffix': suffix
            })

            # 記録ファイル用のコンテンツを準備
            record_file_content += f"{strategy.title()} Strategy:\n"
            record_file_content += f"  - Job ID: {job.id}\n"
            record_file_content += f"  - Training File: {training_file.id}\n"
            record_file_content += f"  - Suffix: {suffix}\n"
            record_file_content += f"  - Model ID: [Pending...]\n\n"

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # 初期記録ファイルを書き込む
    try:
        os.makedirs(os.path.dirname(record_file_path), exist_ok=True)
        with open(record_file_path, 'w') as f:
            f.write(record_file_content)
        print(f"\n✅ Initial record file created: {record_file_path}")
    except Exception as e:
        print(f"❌ Failed to create initial record file: {e}")

    return job_configs

def check_and_update_status(client, job_configs, record_file_path):
    """
    Fine-tuningジョブのステータスを監視し、完了したら記録ファイルを更新する
    """
    print(f"\n\n=== Monitoring Fine-tuning Status ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    
    while True:
        all_done = True
        completed_models = []
        in_progress_jobs = 0

        for config in job_configs:
            strategy = config['strategy']
            job_id = config['job_id']
            
            try:
                job = client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                if status == 'succeeded':
                    if not any(m['job_id'] == job_id for m in completed_models):
                        model_id = job.fine_tuned_model
                        print(f"✅ SUCCESS: {strategy.upper():<15} | Model: {model_id}")
                        completed_models.append({
                            'strategy': strategy,
                            'model_id': model_id,
                            'job_id': job_id
                        })
                elif status in ['running', 'validating_files', 'queued']:
                    print(f"🔄 IN PROGRESS: {strategy.upper():<12} | Status: {status}")
                    all_done = False
                    in_progress_jobs += 1
                else: # failed, cancelled
                    print(f"❌ FAILED/CANCELLED: {strategy.upper():<12} | Status: {status}")
            
            except Exception as e:
                print(f"❌ ERROR ({strategy.upper()}): {str(e)}")
                # エラーが発生した場合、そのジョブは一旦監視対象から外す
                all_done = False


        if completed_models:
            update_record_file(record_file_path, completed_models)
            # 更新済みのモデルをjob_configsから削除して、二重更新を防ぐ
            job_configs = [j for j in job_configs if j['job_id'] not in [m['job_id'] for m in completed_models]]

        if all_done:
            print("\n🎉 All fine-tuning jobs are completed or have failed.")
            break
        
        print(f"\n---\n📊 In Progress: {in_progress_jobs} jobs. Waiting for 5 minutes before next check...\n---")
        time.sleep(300)

def update_record_file(record_file, completed_models):
    """完了したモデルIDを記録ファイルに更新"""
    try:
        with open(record_file, 'r') as f:
            content = f.read()
        
        updated = False
        for model in completed_models:
            strategy = model['strategy']
            model_id = model['model_id']
            
            # [Pending...] を実際のモデルIDに置換
            pattern = re.compile(f"({strategy.title()} Strategy:.*?Model ID: )\[Pending...\]", re.DOTALL)
            new_content, count = pattern.subn(rf"\1{model_id}", content)
            
            if count > 0:
                content = new_content
                updated = True
                print(f"   > Updated model ID for {strategy}")

        if updated:
            with open(record_file, 'w') as f:
                f.write(content)
            print(f"✅ Record file updated: {record_file}")
        
    except Exception as e:
        print(f"❌ Failed to update record file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Create, monitor, and record OpenAI fine-tuning jobs from data files in a specified directory.",
        usage="python3 %(prog)s <data_directory>"
    )
    parser.add_argument('data_directory', type=str, help='The directory containing ft_data_*.jsonl files.')
    args = parser.parse_args()

    client = openai.OpenAI()
    
    data_directory = args.data_directory
    if not os.path.isdir(data_directory):
        print(f"Error: Directory not found at {data_directory}")
        return
        
    record_file = os.path.join(data_directory, "finetuning_jobs_record.txt")

    # 1. データファイルからFine-tuningジョブを作成
    job_configs = create_finetuning_jobs_from_files(client, data_directory, record_file)
    
    if not job_configs:
        print("No jobs to monitor. Exiting.")
        return

    # 2. ジョブのステータスを監視し、完了したら記録を更新
    check_and_update_status(client, job_configs, record_file)

if __name__ == "__main__":
    main()
