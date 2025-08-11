#!/bin/bash
set -euo pipefail

# Production Deployment Script for LLM Tab Cleaner
# This script handles zero-downtime deployment with comprehensive checks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAMESPACE="llm-tab-cleaner-prod"
IMAGE_NAME="llm-tab-cleaner"
DEPLOYMENT_NAME="llm-tab-cleaner-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -v, --version VERSION    Container image version to deploy (required)
    -e, --environment ENV    Environment (prod, staging) [default: prod]
    -n, --namespace NS       Kubernetes namespace [default: llm-tab-cleaner-prod]
    -r, --registry REGISTRY  Container registry URL
    --dry-run               Show what would be deployed without making changes
    --skip-tests            Skip pre-deployment tests
    --rollback              Rollback to previous deployment
    -h, --help              Show this help message

Examples:
    $0 -v 0.3.0                           # Deploy version 0.3.0 to prod
    $0 -v 0.3.1 -e staging               # Deploy to staging
    $0 --rollback                         # Rollback current deployment
    $0 -v 0.3.0 --dry-run                # Show deployment plan

EOF
}

# Parse command line arguments
parse_args() {
    VERSION=""
    ENVIRONMENT="prod"
    REGISTRY="docker.io/terragonlabs"
    DRY_RUN=false
    SKIP_TESTS=false
    ROLLBACK=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    if [[ "$ROLLBACK" == false && -z "$VERSION" ]]; then
        log_error "Version is required unless using --rollback"
        show_usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("kubectl" "docker" "helm")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "$cmd is required but not installed"
            exit 1
        fi
    done
    
    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context 2>/dev/null || echo "")
    if [[ -z "$current_context" ]]; then
        log_error "No kubectl context set"
        exit 1
    fi
    
    log_info "Using kubectl context: $current_context"
    
    # Verify cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push container image
build_and_push_image() {
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would build and push image: $REGISTRY/$IMAGE_NAME:$VERSION"
        return 0
    fi
    
    log_info "Building container image..."
    
    cd "$PROJECT_ROOT"
    
    # Build multi-stage image
    docker build \
        -f Dockerfile.production \
        -t "$REGISTRY/$IMAGE_NAME:$VERSION" \
        -t "$REGISTRY/$IMAGE_NAME:latest" \
        --target production \
        .
    
    # Run security scan
    log_info "Running security scan..."
    if command -v trivy &> /dev/null; then
        trivy image --exit-code 0 --severity HIGH,CRITICAL "$REGISTRY/$IMAGE_NAME:$VERSION"
    else
        log_warning "Trivy not installed, skipping security scan"
    fi
    
    # Push image
    log_info "Pushing image to registry..."
    docker push "$REGISTRY/$IMAGE_NAME:$VERSION"
    docker push "$REGISTRY/$IMAGE_NAME:latest"
    
    log_success "Image built and pushed: $REGISTRY/$IMAGE_NAME:$VERSION"
}

# Run pre-deployment tests
run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping pre-deployment tests"
        return 0
    fi
    
    log_info "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Create test container
    local test_container="llm-tab-cleaner-test-$$"
    
    docker run --name "$test_container" \
        -v "$PROJECT_ROOT:/app" \
        --rm \
        "$REGISTRY/$IMAGE_NAME:$VERSION" \
        sh -c "cd /app && python -m pytest tests/ -v --tb=short" || {
        log_error "Pre-deployment tests failed"
        return 1
    }
    
    log_success "Pre-deployment tests passed"
}

# Backup current deployment
backup_current_deployment() {
    log_info "Backing up current deployment..."
    
    local backup_file="$PROJECT_ROOT/deployment/backups/backup-$(date +%Y%m%d-%H%M%S).yaml"
    mkdir -p "$(dirname "$backup_file")"
    
    kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o yaml > "$backup_file" 2>/dev/null || {
        log_warning "No existing deployment to backup"
        return 0
    }
    
    log_success "Deployment backed up to: $backup_file"
}

# Deploy application
deploy_application() {
    log_info "Deploying application..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would apply manifests with image: $REGISTRY/$IMAGE_NAME:$VERSION"
        kubectl apply -f "$PROJECT_ROOT/deployment/production-ready.yml" --dry-run=client
        return 0
    fi
    
    # Update image in deployment manifest
    local temp_manifest="/tmp/production-ready-$$.yml"
    sed "s|image: llm-tab-cleaner:latest|image: $REGISTRY/$IMAGE_NAME:$VERSION|g" \
        "$PROJECT_ROOT/deployment/production-ready.yml" > "$temp_manifest"
    
    # Apply manifests
    kubectl apply -f "$temp_manifest"
    
    # Clean up
    rm -f "$temp_manifest"
    
    log_success "Application deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would wait for deployment to be ready"
        return 0
    fi
    
    log_info "Waiting for deployment to be ready..."
    
    # Wait for rollout to complete
    if kubectl rollout status deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=600s; then
        log_success "Deployment is ready"
    else
        log_error "Deployment failed to become ready"
        return 1
    fi
    
    # Verify pods are running
    local ready_pods
    ready_pods=$(kubectl get pods -n "$NAMESPACE" -l "app=llm-tab-cleaner" --field-selector=status.phase=Running --no-headers | wc -l)
    log_info "Ready pods: $ready_pods"
    
    if [[ "$ready_pods" -eq 0 ]]; then
        log_error "No pods are running"
        return 1
    fi
}

# Run health checks
run_health_checks() {
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run health checks"
        return 0
    fi
    
    log_info "Running health checks..."
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service "$DEPLOYMENT_NAME-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -z "$service_ip" ]]; then
        log_error "Could not get service IP"
        return 1
    fi
    
    # Run health check from within cluster
    kubectl run health-check-$$ \
        --rm -i --restart=Never \
        --image=curlimages/curl:latest \
        --command -- curl -f "http://$service_ip:8000/health" || {
        log_error "Health check failed"
        return 1
    }
    
    log_success "Health checks passed"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would rollback deployment"
        return 0
    fi
    
    kubectl rollout undo deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=300s
    
    log_success "Deployment rolled back"
}

# Send deployment notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Placeholder for notification system (Slack, email, etc.)
    log_info "Notification: $status - $message"
    
    # Example: Send to Slack webhook
    # curl -X POST -H 'Content-type: application/json' \
    #     --data "{\"text\":\"Deployment $status: $message\"}" \
    #     "$SLACK_WEBHOOK_URL"
}

# Main deployment workflow
main() {
    parse_args "$@"
    
    log_info "Starting deployment process..."
    log_info "Version: ${VERSION:-rollback}"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Registry: $REGISTRY"
    
    # Handle rollback
    if [[ "$ROLLBACK" == true ]]; then
        check_prerequisites
        rollback_deployment
        send_notification "SUCCESS" "Rollback completed"
        exit 0
    fi
    
    # Deployment workflow
    check_prerequisites || {
        send_notification "FAILED" "Prerequisites check failed"
        exit 1
    }
    
    build_and_push_image || {
        send_notification "FAILED" "Image build/push failed"
        exit 1
    }
    
    run_pre_deployment_tests || {
        send_notification "FAILED" "Pre-deployment tests failed"
        exit 1
    }
    
    backup_current_deployment
    
    deploy_application || {
        send_notification "FAILED" "Application deployment failed"
        exit 1
    }
    
    wait_for_deployment || {
        log_error "Deployment failed, attempting rollback..."
        rollback_deployment
        send_notification "FAILED" "Deployment failed, rolled back"
        exit 1
    }
    
    run_health_checks || {
        log_error "Health checks failed, attempting rollback..."
        rollback_deployment
        send_notification "FAILED" "Health checks failed, rolled back"
        exit 1
    }
    
    log_success "🎉 Deployment completed successfully!"
    send_notification "SUCCESS" "Version $VERSION deployed successfully"
}

# Run main function
main "$@"