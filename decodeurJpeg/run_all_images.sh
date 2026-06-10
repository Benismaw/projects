#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Clean and build the project
echo "=== Cleaning previous build ==="
make -s clean

echo "=== Building jpeg2ppm ==="
make -s

# Check if build was successful
if [ ! -x "./jpeg2ppm" ]; then
    echo "Error: Build failed or executable not found!"
    exit 1
fi

# Process all files in the images folder
echo "=== Processing images ==="
IMAGES_DIR="./images"

# Make sure the images directory exists
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

# Count number of images to process
image_count=$(find "$IMAGES_DIR" -type f | wc -l)
echo "Found $image_count images to process"
echo ""

# Process each image
find "$IMAGES_DIR" -type f | while read img; do
    filename=$(basename "$img")
    echo "Processing: $filename"
    ./jpeg2ppm "$img"
done

echo ""
echo "=== Processing complete ==="