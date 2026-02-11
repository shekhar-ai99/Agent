#!/bin/bash
# TradingPlatform UI - Quick Start Script
# Opens the UI in your default browser and optionally starts a local server

set -e

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000

echo ""
echo "========================================================================"
echo "TradingPlatform UI - Quick Start"
echo "========================================================================"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✓ Python found: $(python3 --version)"
else
    echo "✗ Python 3 not found. Install Python 3.8+ and try again."
    exit 1
fi

# Option 1: Start local server
echo ""
echo "Options:"
echo "  1) Start local server (recommended)"
echo "  2) Open HTML directly (no server needed)"
echo ""
read -p "Choose option (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Starting local HTTP server on port $PORT..."
    echo "Open http://localhost:$PORT in your browser"
    echo ""
    echo "To stop the server, press Ctrl+C"
    echo ""
    
    cd "$UI_DIR"
    python3 -m http.server $PORT --directory .
    
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Opening UI directly..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$UI_DIR/index.html"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "$UI_DIR/index.html"
    elif [[ "$OSTYPE" == "msys" ]]; then
        # Windows
        start "$UI_DIR/index.html"
    fi
    
    echo "✓ UI opened in browser"
    echo ""
    echo "Note: Some features (like charts) work better with a local server."
    echo "Run 'python3 -m http.server 8000' in this directory to start one."
else
    echo "Invalid choice"
    exit 1
fi

echo ""
