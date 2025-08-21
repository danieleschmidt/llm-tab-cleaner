#!/bin/bash
# Rollback script for LLM Tab Cleaner

set -e

NAMESPACE="llm-tab-cleaner"
REVISION=${1:-""}

echo "🔄 Rolling back LLM Tab Cleaner deployment..."

if [ -z "$REVISION" ]; then
    echo "Rolling back to previous revision..."
    helm rollback llm-tab-cleaner -n $NAMESPACE
else
    echo "Rolling back to revision $REVISION..."
    helm rollback llm-tab-cleaner $REVISION -n $NAMESPACE
fi

echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/llm-tab-cleaner -n $NAMESPACE --timeout=300s

echo "✅ Rollback completed successfully!"
