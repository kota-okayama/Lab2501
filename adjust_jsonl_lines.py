#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path

def extract_target_lines_from_folder_name(folder_path):
    """
    フォルダ名から目標行数を抽出する
    例: FT100 -> 100, FT200 -> 200
    """
    folder_name = Path(folder_path).name
    match = re.search(r'FT(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

def adjust_jsonl_file_lines(file_path, target_lines):
    """
    jsonlファイルの行数を目標行数に調整する（下から削除）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_lines = len(lines)
    print(f"  現在の行数: {current_lines}")
    
    if current_lines > target_lines:
        # 下から削除して目標行数にする
        lines = lines[:target_lines]
        print(f"  {current_lines - target_lines}行削除して{target_lines}行に調整")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
    elif current_lines < target_lines:
        print(f"  警告: 現在の行数({current_lines})が目標行数({target_lines})より少ないです")
        return False
    else:
        print(f"  すでに目標行数({target_lines})です")
        return False

def find_ft_folders(base_path):
    """
    指定されたディレクトリ内のFTフォルダを検索する
    """
    base_path = Path(base_path)
    ft_folders = []
    
    for item in base_path.iterdir():
        if item.is_dir() and re.match(r'^FT\d+$', item.name):
            ft_folders.append(item)
    
    # フォルダ名の数字順にソート
    ft_folders.sort(key=lambda x: int(re.search(r'FT(\d+)', x.name).group(1)))
    return ft_folders

def process_single_folder(folder_path, dry_run=False):
    """
    単一のFTフォルダを処理する
    """
    # フォルダ名から目標行数を抽出
    target_lines = extract_target_lines_from_folder_name(folder_path)
    if target_lines is None:
        print(f"エラー: フォルダ名から数字を抽出できません: {folder_path.name}")
        return 0, 0
    
    print(f"\n{'='*60}")
    print(f"フォルダ: {folder_path}")
    print(f"目標行数: {target_lines}")
    print('='*60)
    
    # jsonlファイルを検索
    jsonl_files = list(folder_path.glob('*.jsonl'))
    
    if not jsonl_files:
        print("jsonlファイルが見つかりません")
        return 0, 0
    
    print(f"{len(jsonl_files)}個のjsonlファイルを処理します:")
    
    modified_count = 0
    processed_count = len(jsonl_files)
    
    for jsonl_file in jsonl_files:
        print(f"\n処理中: {jsonl_file.name}")
        
        if dry_run:
            # ドライランモード：現在の行数のみ表示
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                current_lines = len(f.readlines())
            print(f"  現在の行数: {current_lines}")
            if current_lines > target_lines:
                print(f"  {current_lines - target_lines}行削除して{target_lines}行に調整予定")
                modified_count += 1
            elif current_lines < target_lines:
                print(f"  警告: 現在の行数({current_lines})が目標行数({target_lines})より少ないです")
            else:
                print(f"  すでに目標行数({target_lines})です")
        else:
            # 実際に調整
            if adjust_jsonl_file_lines(jsonl_file, target_lines):
                modified_count += 1
    
    print(f"\nフォルダ完了: {modified_count}個のファイルを{'調整予定' if dry_run else '調整しました'}")
    return processed_count, modified_count

def main():
    parser = argparse.ArgumentParser(description='jsonlファイルの行数をフォルダ名の数字に合わせて調整')
    parser.add_argument('path', help='調整対象のフォルダパス、または親ディレクトリパス（FTフォルダを自動検索）')
    parser.add_argument('--dry-run', action='store_true', help='実際の変更は行わず、処理内容のみ表示')
    parser.add_argument('--single', action='store_true', help='単一フォルダのみ処理（FTフォルダの自動検索を無効化）')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"エラー: パスが存在しません: {path}")
        return 1
    
    if not path.is_dir():
        print(f"エラー: 指定されたパスはディレクトリではありません: {path}")
        return 1
    
    # 単一フォルダモードまたは既にFTフォルダの場合
    if args.single or extract_target_lines_from_folder_name(path) is not None:
        if extract_target_lines_from_folder_name(path) is None:
            print(f"エラー: フォルダ名から数字を抽出できません: {path.name}")
            print("フォルダ名は 'FT数字' の形式である必要があります（例: FT100）")
            return 1
        
        processed, modified = process_single_folder(path, args.dry_run)
        
        if args.dry_run:
            print(f"\n実際に変更を行う場合は --dry-run オプションを外して再実行してください")
        
        return 0
    
    # 複数のFTフォルダを自動検索して処理
    ft_folders = find_ft_folders(path)
    
    if not ft_folders:
        print(f"エラー: 指定されたディレクトリ内にFTフォルダが見つかりません: {path}")
        print("FTフォルダは 'FT数字' の形式である必要があります（例: FT100, FT200）")
        return 1
    
    print(f"見つかったFTフォルダ: {len(ft_folders)}個")
    for folder in ft_folders:
        print(f"  - {folder.name}")
    
    total_processed = 0
    total_modified = 0
    
    for ft_folder in ft_folders:
        processed, modified = process_single_folder(ft_folder, args.dry_run)
        total_processed += processed
        total_modified += modified
    
    print(f"\n{'='*60}")
    print(f"全体完了: {len(ft_folders)}個のフォルダ、{total_processed}個のファイルを処理")
    print(f"          {total_modified}個のファイルを{'調整予定' if args.dry_run else '調整しました'}")
    
    if args.dry_run:
        print(f"\n実際に変更を行う場合は --dry-run オプションを外して再実行してください")
    
    return 0

if __name__ == "__main__":
    exit(main())
