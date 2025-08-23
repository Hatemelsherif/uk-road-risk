#!/bin/bash

# Training Monitor Script
echo "🔍 UK Road Risk - Training Monitor"
echo "=================================="

while true; do
    clear
    echo "🔍 UK Road Risk - Training Monitor"
    echo "=================================="
    echo "📅 $(date)"
    echo ""
    
    # Check active training processes
    echo "🚀 Active Training Processes:"
    echo "----------------------------"
    TRAINING_PROCS=$(ps aux | grep "train_models_enhanced.py" | grep -v grep)
    
    if [ -z "$TRAINING_PROCS" ]; then
        echo "❌ No training processes running"
    else
        echo "$TRAINING_PROCS" | while read line; do
            PID=$(echo $line | awk '{print $2}')
            CPU=$(echo $line | awk '{print $3}')
            MEM=$(echo $line | awk '{print $4}')
            TIME=$(echo $line | awk '{print $10}')
            echo "  PID: $PID | CPU: $CPU% | Memory: $MEM% | Time: $TIME"
        done
    fi
    
    echo ""
    
    # Check training status
    echo "📊 Training Status:"
    echo "-------------------"
    if [ -f "data/models/training_status.json" ]; then
        cat data/models/training_status.json | jq -r '
            "Status: " + .status + 
            " | Progress: " + (.progress | tostring) + "%" +
            " | Message: " + .message +
            " | Updated: " + .timestamp'
    else
        echo "❌ No training status file found"
    fi
    
    echo ""
    
    # Check for new model directories
    echo "📁 Latest Model Artifacts:"
    echo "-------------------------"
    ls -la data/models/ | tail -3
    
    echo ""
    echo "🔄 Refreshing in 10 seconds... (Ctrl+C to exit)"
    
    sleep 10
done