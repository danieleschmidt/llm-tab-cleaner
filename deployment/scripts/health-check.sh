#!/bin/bash
# Health check script for LLM Tab Cleaner

NAMESPACE="llm-tab-cleaner"
SERVICE_NAME="llm-tab-cleaner-service"

echo "🔍 Checking LLM Tab Cleaner health..."

# Check pods
echo "📦 Pod status:"
kubectl get pods -n $NAMESPACE

# Check services
echo "🌐 Service status:"
kubectl get services -n $NAMESPACE

# Check ingress
echo "🚪 Ingress status:"
kubectl get ingress -n $NAMESPACE

# Test health endpoint
echo "❤️  Testing health endpoint..."
kubectl port-forward -n $NAMESPACE service/$SERVICE_NAME 8080:80 &
PF_PID=$!
sleep 5

if curl -f http://localhost:8080/health >/dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
fi

kill $PF_PID 2>/dev/null || true
