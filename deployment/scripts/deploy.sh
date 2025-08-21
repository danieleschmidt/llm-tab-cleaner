#!/bin/bash
# Production deployment script for LLM Tab Cleaner

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="llm-tab-cleaner"
CHART_PATH="./helm/llm-tab-cleaner"
VALUES_FILE="values.production.yaml"

echo -e "${GREEN}🚀 Starting LLM Tab Cleaner Deployment${NC}"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl is required but not installed.${NC}" >&2; exit 1; }
command -v helm >/dev/null 2>&1 || { echo -e "${RED}helm is required but not installed.${NC}" >&2; exit 1; }

# Check cluster connectivity
echo -e "${YELLOW}🔗 Checking cluster connectivity...${NC}"
kubectl cluster-info >/dev/null 2>&1 || { echo -e "${RED}Cannot connect to Kubernetes cluster.${NC}" >&2; exit 1; }

# Create namespace if it doesn't exist
echo -e "${YELLOW}📦 Creating namespace...${NC}"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets
echo -e "${YELLOW}🔐 Applying secrets...${NC}"
if [ -f "secrets.yaml" ]; then
    kubectl apply -f secrets.yaml -n $NAMESPACE
else
    echo -e "${YELLOW}⚠️  No secrets.yaml found. Please ensure secrets are configured.${NC}"
fi

# Deploy with Helm
echo -e "${YELLOW}⛵ Deploying with Helm...${NC}"
helm upgrade --install llm-tab-cleaner $CHART_PATH \
    --namespace $NAMESPACE \
    --values $VALUES_FILE \
    --timeout 10m \
    --wait

# Wait for rollout
echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
kubectl rollout status deployment/llm-tab-cleaner -n $NAMESPACE --timeout=300s

# Run health checks
echo -e "${YELLOW}🔍 Running health checks...${NC}"
sleep 10
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Test endpoint
EXTERNAL_IP=$(kubectl get service llm-tab-cleaner-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ ! -z "$EXTERNAL_IP" ]; then
    echo -e "${GREEN}✅ Deployment successful! External IP: $EXTERNAL_IP${NC}"
    echo -e "${GREEN}🌐 Health check: curl http://$EXTERNAL_IP/health${NC}"
else
    echo -e "${YELLOW}⚠️  External IP not yet assigned. Check service status.${NC}"
fi

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
