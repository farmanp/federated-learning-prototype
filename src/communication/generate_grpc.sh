#!/bin/bash
# Script to generate gRPC code from protobuf definitions

# Make sure we're in the project root
cd "$(dirname "$0")/../.."

# Ensure the output directory exists
mkdir -p src/communication/generated

# Generate gRPC code
python -m grpc_tools.protoc \
    --proto_path=src/communication/proto \
    --python_out=src/communication/generated \
    --grpc_python_out=src/communication/generated \
    src/communication/proto/aggregator_service.proto

# Create an __init__.py to make the directory a proper package
touch src/communication/generated/__init__.py

echo "gRPC code generation completed successfully!"
