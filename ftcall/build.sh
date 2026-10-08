#!/bin/sh
# Build the ftcall helper (docs/DESIGN_FACETIME_CALL.md section 3).
set -e
cd "$(dirname "$0")"
swiftc -O main.swift -o ftcall
echo "built $(pwd)/ftcall"
