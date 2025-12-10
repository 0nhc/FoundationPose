#!/bin/bash

# Script to download model weights from Google Drive
# Usage: ./download_weights.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if gdown is installed
if command -v gdown &> /dev/null; then
    DOWNLOAD_CMD="gdown"
    echo "Using gdown for downloads"
else
    echo "Error: gdown is not installed. Please install it:"
    echo "  pip install gdown"
    echo ""
    echo "Note: gdown is required to handle Google Drive virus scan warnings for large files."
    exit 1
fi

# Function to download from Google Drive using sharing link
download_file() {
    local sharing_link=$1
    local output_path=$2
    local file_name=$(basename "$output_path")
    local dir_name=$(dirname "$output_path")
    
    # Create directory if it doesn't exist
    mkdir -p "$dir_name"
    
    echo "Downloading $file_name..."
    echo "  From: $sharing_link"
    
    # Extract file ID from sharing link
    file_id=$(echo "$sharing_link" | sed -n 's/.*\/d\/\([a-zA-Z0-9_-]*\).*/\1/p')
    
    if [ -z "$file_id" ]; then
        echo "✗ Could not extract file ID from sharing link"
        return 1
    fi
    
    # Try method 1: Use gdown with --fuzzy flag (handles sharing links and virus scan warnings)
    echo "  Attempting download with gdown --fuzzy..."
    if gdown --fuzzy "$sharing_link" -O "$output_path" 2>/dev/null; then
        # Check if download was successful and not HTML
        if [ -f "$output_path" ] && [ -s "$output_path" ]; then
            if ! head -n 1 "$output_path" 2>/dev/null | grep -q "<!DOCTYPE html"; then
                file_size=$(du -h "$output_path" | cut -f1)
                echo "✓ Successfully downloaded $file_name"
                echo "  File size: $file_size"
                return 0
            fi
        fi
    fi
    
    # Try method 2: Use direct download URL with confirm parameter
    echo "  Attempting direct download with confirm parameter..."
    if gdown "https://drive.google.com/uc?id=$file_id&confirm=t" -O "$output_path" 2>/dev/null; then
        # Check if download was successful and not HTML
        if [ -f "$output_path" ] && [ -s "$output_path" ]; then
            if ! head -n 1 "$output_path" 2>/dev/null | grep -q "<!DOCTYPE html"; then
                file_size=$(du -h "$output_path" | cut -f1)
                echo "✓ Successfully downloaded $file_name"
                echo "  File size: $file_size"
                return 0
            fi
        fi
    fi
    
    # If both methods failed, show error
    echo "✗ Failed to download $file_name"
    echo "  Google Drive requires manual confirmation for large files."
    echo "  Please download manually:"
    echo "  1. Visit: $sharing_link"
    echo "  2. Click 'Download anyway' when prompted"
    echo "  3. Save the file to: $output_path"
    echo ""
    echo "  Or try updating gdown: pip install --upgrade gdown"
    rm -f "$output_path"
    return 1
}

# Download model_best.pth for 2023-10-28-18-33-37
SHARING_LINK_2023="https://drive.google.com/file/d/1tIKRNayiqFbzAgUrhM3pGxRDkbI88aUE/view?usp=sharing"
OUTPUT_PATH_2023="2023-10-28-18-33-37/model_best.pth"
download_file "$SHARING_LINK_2023" "$OUTPUT_PATH_2023"

echo ""

# Download model_best.pth for 2024-01-11-20-02-45
SHARING_LINK_2024="https://drive.google.com/file/d/1zW595DbWOkxG6sx8Vi32drS9DK7P2toK/view?usp=sharing"
OUTPUT_PATH_2024="2024-01-11-20-02-45/model_best.pth"
download_file "$SHARING_LINK_2024" "$OUTPUT_PATH_2024"

echo ""
echo "=========================================="
echo "All downloads completed!"
echo "=========================================="