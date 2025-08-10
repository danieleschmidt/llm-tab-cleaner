#!/bin/bash
set -euo pipefail

# LLM Tab Cleaner Deployment Script
# Usage: ./deploy.sh [environment] [action]
# Examples: 
#   ./deploy.sh production deploy
#   ./deploy.sh staging destroy
#   ./deploy.sh development plan

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
ENVIRONMENT="${1:-production}"
ACTION="${2:-deploy}"
REGION="${AWS_REGION:-us-west-2}"
CLUSTER_NAME="llm-tab-cleaner-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if required tools are installed
    local tools=("docker" "kubectl" "terraform" "aws" "helm")
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is not installed. Please install it first."
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Please run 'aws configure' first."
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi
    
    success "All prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log "Building and pushing Docker image..."
    
    local ecr_uri=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${REGION}.amazonaws.com
    local image_name="${ecr_uri}/${CLUSTER_NAME}:$(git rev-parse --short HEAD)"
    local latest_name="${ecr_uri}/${CLUSTER_NAME}:latest"
    
    # Login to ECR
    aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ecr_uri}"
    
    # Build image
    cd "${PROJECT_ROOT}"
    docker build -t "${image_name}" -t "${latest_name}" .
    
    # Push images
    docker push "${image_name}"
    docker push "${latest_name}"
    
    success "Docker image built and pushed: ${image_name}"
    echo "IMAGE_URI=${image_name}" > "${SCRIPT_DIR}/.env.image"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd "${SCRIPT_DIR}/../terraform"
    
    # Initialize Terraform
    terraform init -upgrade
    
    # Create workspace if it doesn't exist
    terraform workspace select "${ENVIRONMENT}" || terraform workspace new "${ENVIRONMENT}"
    
    case "${ACTION}" in
        "plan")
            terraform plan \
                -var="environment=${ENVIRONMENT}" \
                -var="region=${REGION}" \
                -var="cluster_name=${CLUSTER_NAME}" \
                -out="terraform.tfplan"
            ;;
        "deploy"|"apply")
            terraform plan \
                -var="environment=${ENVIRONMENT}" \
                -var="region=${REGION}" \
                -var="cluster_name=${CLUSTER_NAME}" \
                -out="terraform.tfplan"
            
            echo
            read -p "Do you want to apply these changes? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                terraform apply "terraform.tfplan"
                
                # Save outputs
                terraform output -json > "${SCRIPT_DIR}/.terraform-outputs.json"
                success "Infrastructure deployed successfully"
            else
                warning "Deployment cancelled by user"
                exit 0
            fi
            ;;
        "destroy")
            warning "This will destroy ALL infrastructure for environment: ${ENVIRONMENT}"
            echo "Cluster: ${CLUSTER_NAME}"
            echo "Region: ${REGION}"
            echo
            read -p "Are you sure you want to destroy everything? Type 'yes' to confirm: " -r
            echo
            
            if [[ $REPLY == "yes" ]]; then
                terraform destroy \
                    -var="environment=${ENVIRONMENT}" \
                    -var="region=${REGION}" \
                    -var="cluster_name=${CLUSTER_NAME}" \
                    -auto-approve
                success "Infrastructure destroyed"
            else
                warning "Destruction cancelled"
                exit 0
            fi
            ;;
        *)
            error "Unknown action: ${ACTION}. Use: plan, deploy, apply, or destroy"
            ;;
    esac
}

# Configure kubectl
configure_kubectl() {
    log "Configuring kubectl..."
    
    aws eks update-kubeconfig \
        --region "${REGION}" \
        --name "${CLUSTER_NAME}" \
        --alias "${CLUSTER_NAME}"
    
    # Test connection
    if kubectl cluster-info &> /dev/null; then
        success "kubectl configured and connected to cluster"
    else
        error "Failed to connect to Kubernetes cluster"
    fi
}

# Deploy Kubernetes applications
deploy_kubernetes_apps() {
    log "Deploying Kubernetes applications..."
    
    cd "${SCRIPT_DIR}/../k8s"
    
    # Apply namespace and basic resources first
    kubectl apply -f deployment.yaml
    
    # Wait for deployments to be ready
    log "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/llm-tab-cleaner-api -n llm-tab-cleaner
    kubectl wait --for=condition=available --timeout=600s deployment/llm-tab-cleaner-worker -n llm-tab-cleaner
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n llm-tab-cleaner
    
    success "Kubernetes applications deployed successfully"
}

# Install monitoring stack
install_monitoring() {
    log "Installing monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values "${SCRIPT_DIR}/../monitoring/prometheus-values.yaml" \
        --wait
    
    # Install Grafana dashboards
    kubectl apply -f "${SCRIPT_DIR}/../monitoring/grafana-dashboards/"
    
    success "Monitoring stack installed"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check if all pods are running
    if ! kubectl get pods -n llm-tab-cleaner --field-selector=status.phase!=Running --no-headers | grep -q .; then
        success "All pods are running"
    else
        warning "Some pods are not running:"
        kubectl get pods -n llm-tab-cleaner --field-selector=status.phase!=Running
    fi
    
    # Check API health endpoint
    local api_url=$(kubectl get svc llm-tab-cleaner-api-service -n llm-tab-cleaner -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [[ -n "${api_url}" ]]; then
        if curl -f "http://${api_url}/health" &> /dev/null; then
            success "API health check passed"
        else
            warning "API health check failed"
        fi
    else
        warning "LoadBalancer URL not available yet"
    fi
    
    # Check HPA status
    kubectl get hpa -n llm-tab-cleaner
}

# Display deployment information
show_deployment_info() {
    log "Deployment Information"
    echo "=================================="
    echo "Environment: ${ENVIRONMENT}"
    echo "Region: ${REGION}"
    echo "Cluster: ${CLUSTER_NAME}"
    echo
    
    # Get LoadBalancer URLs
    echo "Service Endpoints:"
    kubectl get svc -n llm-tab-cleaner -o wide
    echo
    
    # Get Ingress information
    if kubectl get ingress -n llm-tab-cleaner &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n llm-tab-cleaner
        echo
    fi
    
    # Show Grafana access information
    if kubectl get svc -n monitoring prometheus-grafana &> /dev/null; then
        local grafana_password=$(kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode)
        echo "Grafana Dashboard:"
        echo "URL: http://$(kubectl get svc -n monitoring prometheus-grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
        echo "Username: admin"
        echo "Password: ${grafana_password}"
        echo
    fi
    
    echo "To access your cluster:"
    echo "kubectl config use-context ${CLUSTER_NAME}"
    echo "kubectl get pods -n llm-tab-cleaner"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f "${SCRIPT_DIR}/.env.image"
    rm -f "${SCRIPT_DIR}/../terraform/terraform.tfplan"
}

# Main execution
main() {
    trap cleanup EXIT
    
    log "Starting LLM Tab Cleaner deployment"
    log "Environment: ${ENVIRONMENT}"
    log "Action: ${ACTION}"
    log "Region: ${REGION}"
    log "Cluster: ${CLUSTER_NAME}"
    
    check_prerequisites
    
    case "${ACTION}" in
        "plan")
            deploy_infrastructure
            ;;
        "deploy"|"apply")
            build_and_push_image
            deploy_infrastructure
            configure_kubectl
            deploy_kubernetes_apps
            install_monitoring
            run_health_checks
            show_deployment_info
            ;;
        "destroy")
            deploy_infrastructure
            ;;
        "health")
            configure_kubectl
            run_health_checks
            ;;
        "info")
            configure_kubectl
            show_deployment_info
            ;;
        *)
            error "Unknown action: ${ACTION}"
            echo "Available actions: plan, deploy, apply, destroy, health, info"
            ;;
    esac
    
    success "Deployment script completed successfully!"
}

# Script help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "LLM Tab Cleaner Deployment Script"
    echo
    echo "Usage: $0 [environment] [action]"
    echo
    echo "Environments: production, staging, development"
    echo "Actions:"
    echo "  plan     - Plan infrastructure changes"
    echo "  deploy   - Deploy everything (default)"
    echo "  apply    - Same as deploy"
    echo "  destroy  - Destroy all infrastructure"
    echo "  health   - Run health checks"
    echo "  info     - Show deployment information"
    echo
    echo "Examples:"
    echo "  $0 production deploy"
    echo "  $0 staging plan" 
    echo "  $0 development destroy"
    echo "  $0 production health"
    exit 0
fi

# Run main function
main "$@"