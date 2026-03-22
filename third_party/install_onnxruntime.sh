#!/bin/bash

# Install ONNX Runtime C++ library for x86_64.

set -euo pipefail

ONNX_VERSION="1.23.2"
PACKAGE="onnxruntime-linux-x64-${ONNX_VERSION}"
ARCHIVE="${PACKAGE}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ARCHIVE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

if [ ! -f "$ARCHIVE" ]; then
	echo "Archive not found locally: $ARCHIVE"
	echo "Downloading $URL"
	if command -v wget >/dev/null 2>&1; then
		wget -O "$ARCHIVE" "$URL"
	elif command -v curl >/dev/null 2>&1; then
		curl -L "$URL" -o "$ARCHIVE"
	else
		echo "Error: Neither wget nor curl is available for download."
		exit 1
	fi
fi

echo "Extracting..."
rm -rf "$PACKAGE"
tar -xzf "$ARCHIVE"

echo "Installing to /usr/local/..."
cd "$PACKAGE"
sudo cp -r include/* /usr/local/include/
sudo cp -r lib/* /usr/local/lib/
sudo ldconfig

echo
echo "Installation complete"
echo
echo "Verification:"
ls -lh /usr/local/include/onnxruntime_cxx_api.h
ls -lh /usr/local/lib/libonnxruntime.so*
