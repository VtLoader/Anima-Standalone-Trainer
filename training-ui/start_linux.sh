#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "Starting Anima Training UI..."

# Check if node_modules exists, if not prompt to install
if [ ! -d "node_modules" ]; then
    echo "node_modules not found. Installing dependencies..."
    npm install
fi

# Start the application
npm start -- "$@"

echo ""
echo "Application exited (check for errors above)."
