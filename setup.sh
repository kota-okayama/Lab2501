#!/bin/bash

echo "=== Siamese Network for Bibliographic Record Matching - Setup Script ==="

# Python3の存在確認
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Python バージョン確認
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# 仮想環境の作成
echo "Creating virtual environment..."
python3 -m venv venv

# 仮想環境のアクティベート
echo "Activating virtual environment..."
source venv/bin/activate

# pipのアップグレード
echo "Upgrading pip..."
pip install --upgrade pip

# 依存ライブラリのインストール
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 必要なディレクトリの作成
echo "Creating necessary directories..."
mkdir -p fasttext_models
mkdir -p benchmark
mkdir -p pipeline-output

# 実行権限の付与
echo "Setting execute permissions..."
chmod +x run_pipeline.sh

echo ""
echo "=== Setup completed successfully! ==="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download FastText models to fasttext_models/ directory"
echo "   Example for Japanese:"
echo "   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.bin.gz"
echo "   gunzip cc.ja.300.bin.gz"
echo "   mv cc.ja.300.bin fasttext_models/"
echo ""
echo "3. Place your benchmark YAML files in the benchmark/ directory"
echo ""
echo "4. Run the pipeline:"
echo "   ./run_pipeline.sh <train_yaml> <eval_yaml> <fasttext_model> <output_dir>"
echo ""
echo "For more information, see README.md" 