#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Check if valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "Error: Valgrind is not installed. Please install it first."
    exit 1
fi

# Clean and build the project (suppressed output)
echo "=== Cleaning previous build ==="
make -s clean



echo "=== Building jpeg2ppm ==="
make -s


# Check if build exists
if [ ! -x "./jpeg2ppm" ]; then
    echo "Error: Build failed!"
    exit 1
fi


# Process all files in the images folder
echo "=== Checking memory leaks for each image ==="
IMAGES_DIR="./images"

# Make sure the images directory exists
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

# Create results directory
RESULTS_DIR="./valgrind_results"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Count number of images to process
image_count=$(find "$IMAGES_DIR" -type f | wc -l)
echo "Found $image_count images to process"
echo ""


# Process each image
find "$IMAGES_DIR" -type f | while read img; do
    filename=$(basename "$img")
    log_file="$RESULTS_DIR/${filename%.jpg}.log"
    log_file="${log_file%.jpeg}.log"
    
    echo -n "Testing: $filename ... "
    
    # Run valgrind and capture output
    valgrind --leak-check=full --error-exitcode=1 ./jpeg2ppm "$img" > /dev/null 2> "$log_file"
    val_result=$?
    
    # Check if "All heap blocks were freed" is in the output
    if grep -q "All heap blocks were freed -- no leaks are possible" "$log_file"; then
        echo -e "\e[32mNo leaks detected\e[0m"
    else
        echo -e "\e[31mFAILED - Memory issues detected\e[0m"
        
        # Extract the leak summary
        echo "  Summary from valgrind:"
        grep -A 5 "LEAK SUMMARY" "$log_file" | head -n 6 | sed 's/^/  /'
        echo ""
    fi
done

# Delete results directory
rm -rf "$RESULTS_DIR"

echo ""
echo "=== Memory leak check complete ==="