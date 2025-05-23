#!/bin/bash

# Exit on error
set -e

# Define directories
BASE_DIR=$(dirname "$0") # Should be src/communication
PROTO_DIR="$BASE_DIR/proto"
GENERATED_DIR="$BASE_DIR/generated"
PYTHON_OUT_DIR="$GENERATED_DIR"
GRPC_PYTHON_OUT_DIR="$GENERATED_DIR"

echo "--- gRPC Generation Script Start ---"
echo "Script working directory: $(pwd)"
echo "Script location (dirname \$0): $BASE_DIR"
echo "Proto directory (absolute): $(realpath "$PROTO_DIR")"
echo "Generated directory (absolute): $(realpath "$GENERATED_DIR")"

# Create the generated directory if it doesn't exist
mkdir -p "$GENERATED_DIR"
echo "Ensured $GENERATED_DIR exists."

# Create __init__.py in generated directory to make it a package
touch "$GENERATED_DIR/__init__.py"
echo "Ensured $GENERATED_DIR/__init__.py exists."

# Find all .proto files in the proto directory
echo "Looking for .proto files in $PROTO_DIR..."
PROTO_FILES=$(find "$PROTO_DIR" -name "*.proto")

# Check if any .proto files were found
if [ -z "$PROTO_FILES" ]; then
    echo "Error: No .proto files found in $PROTO_DIR"
    ls -la "$PROTO_DIR" # List contents of proto dir for debugging
    exit 1
fi

echo "Found .proto files:"
for f in $PROTO_FILES; do
  echo "  - $(realpath "$f")"
done
echo "---"

# Check if grpc_tools.protoc is available
if ! python -m grpc_tools.protoc --version > /dev/null 2>&1; then
    echo "Error: python -m grpc_tools.protoc not found or not working. Make sure grpcio-tools is installed."
    exit 1
fi
echo "python -m grpc_tools.protoc found, version: $(python -m grpc_tools.protoc --version)"

# Generate Python gRPC code
echo "Generating Python gRPC code using python -m grpc_tools.protoc..."
echo "Command: python -m grpc_tools.protoc --proto_path=\"$PROTO_DIR\" --python_out=\"$PYTHON_OUT_DIR\" --grpc_python_out=\"$GRPC_PYTHON_OUT_DIR\" $PROTO_FILES"

python -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$PYTHON_OUT_DIR" \
    --grpc_python_out="$GRPC_PYTHON_OUT_DIR" \
    $PROTO_FILES

echo "Finished attempting gRPC code generation."

# Check if the expected files were created
echo "Checking for generated files in $GENERATED_DIR..."
ls -la "$GENERATED_DIR"

# Specifically check for aggregator_service_pb2.py
EXPECTED_PB2_FILE="$GENERATED_DIR/aggregator_service_pb2.py"
EXPECTED_GRPC_FILE="$GENERATED_DIR/aggregator_service_pb2_grpc.py"

if [ -f "$EXPECTED_PB2_FILE" ] && [ -s "$EXPECTED_PB2_FILE" ]; then
    echo "SUCCESS: $EXPECTED_PB2_FILE was created and is not empty."
else
    echo "ERROR: $EXPECTED_PB2_FILE was NOT created or is empty."
fi

if [ -f "$EXPECTED_GRPC_FILE" ] && [ -s "$EXPECTED_GRPC_FILE" ]; then
    echo "SUCCESS: $EXPECTED_GRPC_FILE was created and is not empty."
    
    # Attempt to fix imports to be relative
    # Example: import aggregator_service_pb2 -> from . import aggregator_service_pb2
    echo "Attempting to make imports relative in $EXPECTED_GRPC_FILE for aggregator_service_pb2..."
    # Use a temporary file for sed on systems that require an extension for -i (like macOS, though in Linux container it's not strictly needed)
    # However, direct -i without extension is fine for GNU sed in the container.
    sed -i -E "s/^import (aggregator_service_pb2\\b.*)/from . import \\1/g" "$EXPECTED_GRPC_FILE"
    
    if grep -q "from . import aggregator_service_pb2" "$EXPECTED_GRPC_FILE"; then
        echo "Successfully made imports relative in $EXPECTED_GRPC_FILE."
    else
        echo "WARNING: Failed or did not need to make imports relative in $EXPECTED_GRPC_FILE for aggregator_service_pb2. Checking original import..."
        if grep -q "^import aggregator_service_pb2" "$EXPECTED_GRPC_FILE"; then
            echo "Original absolute import still present. sed command might have failed."
        else
            echo "Neither relative nor original absolute import found as expected. Manual check might be needed."
        fi
        # For debugging, print relevant import lines
        echo "Relevant import lines from $EXPECTED_GRPC_FILE:"
        grep -E "^(import |from \\.)(${base_name}_pb2|.*${base_name}_pb2)" "$EXPECTED_GRPC_FILE" || echo "No matching import lines found."
    fi
else
    echo "ERROR: $EXPECTED_GRPC_FILE was NOT created or is empty."
fi

echo "--- gRPC Generation Script End ---"
