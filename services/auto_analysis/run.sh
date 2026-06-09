#!/bin/bash

# Auto-Analysis Service Runner
# This script runs the annotation validation service

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables if .env exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
    echo "✓ Loaded environment from .env"
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY is not set"
    echo "Please set GEMINI_API_KEY environment variable or add it to .env file"
    exit 1
fi

echo ""
echo "=================================================="
echo "AUTO-ANALYSIS SERVICE"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  API Key: ${GEMINI_API_KEY:0:20}..."
echo "  Model: gemini-3.1-flash-lite-preview"
echo "  Batch Size: 100 items"
echo "  Concurrent Requests: 10"
echo ""

# Run the main service
cd "$SCRIPT_DIR"
python3 main.py
