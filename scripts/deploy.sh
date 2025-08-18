#!/bin/bash
set -euo pipefail

# Deployment script for LLM Tab Cleaner
# Supports multiple deployment targets and environments

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REGISTRY="${REGISTRY:-ghcr.io/danieleschmidt}"
IMAGE_NAME="${IMAGE_NAME:-llm-tab-cleaner}"
NAMESPACE="${NAMESPACE:-llm-tab-cleaner}"
ENVIRONMENT="${ENVIRONMENT:-staging}"

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

# Help function
show_help() {
    cat << EOF
LLM Tab Cleaner Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  docker-compose    Deploy using Docker Compose
  kubernetes        Deploy to Kubernetes cluster
  aws-ecs          Deploy to AWS ECS
  azure-aci        Deploy to Azure Container Instances
  helm             Deploy using Helm chart
  local            Deploy locally for development

Options:
  --environment, -e    Deployment environment (dev|staging|prod) [default: staging]
  --registry, -r       Container registry [default: ghcr.io/danieleschmidt]
  --tag, -t           Image tag [default: latest]
  --namespace, -n      Kubernetes namespace [default: llm-tab-cleaner]
  --config, -c         Configuration file path
  --dry-run           Show what would be deployed without actually deploying
  --force             Force deployment even if validation fails
  --help, -h          Show this help message

Environment Variables:
  REGISTRY            Container registry URL
  IMAGE_NAME          Docker image name
  NAMESPACE           Kubernetes namespace
  ENVIRONMENT         Deployment environment
  OPENAI_API_KEY      OpenAI API key
  ANTHROPIC_API_KEY   Anthropic API key

Examples:
  $0 docker-compose --environment prod
  $0 kubernetes --tag v1.2.3 --namespace prod
  $0 helm --environment staging --dry-run
  $0 local --config ./configs/dev.env
EOF
}

# Validation functions
check_prerequisites() {
    local deployment_type="$1"
    
    log_info "Checking prerequisites for $deployment_type deployment..."
    
    case "$deployment_type" in
        docker-compose)
            if ! command -v docker-compose >/dev/null 2>&1; then
                log_error "docker-compose is required but not installed"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl >/dev/null 2>&1; then
                log_error "kubectl is required but not installed"
                exit 1
            fi
            ;;
        helm)
            if ! command -v helm >/dev/null 2>&1; then
                log_error "helm is required but not installed"
                exit 1
            fi
            if ! command -v kubectl >/dev/null 2>&1; then
                log_error "kubectl is required but not installed"
                exit 1
            fi
            ;;
        aws-ecs)
            if ! command -v aws >/dev/null 2>&1; then
                log_error "aws cli is required but not installed"
                exit 1
            fi
            ;;
        azure-aci)
            if ! command -v az >/dev/null 2>&1; then
                log_error "azure cli is required but not installed"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

# Validate environment variables
validate_environment() {
    local required_vars=()
    
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        required_vars+=("OPENAI_API_KEY" "ANTHROPIC_API_KEY")
    fi
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Environment validation passed"
}

# Generate configuration from template
generate_config() {
    local template_file="$1"
    local output_file="$2"
    
    log_info "Generating configuration from template: $template_file"
    
    # Use envsubst to substitute environment variables
    envsubst < "$template_file" > "$output_file"
    
    log_success "Configuration generated: $output_file"
}

# Docker Compose deployment
deploy_docker_compose() {
    local compose_file="docker-compose.yml"
    local env_file=".env.${ENVIRONMENT}"
    
    log_info "Deploying with Docker Compose..."
    
    # Check if environment file exists
    if [[ ! -f "$env_file" ]]; then
        log_warning "Environment file $env_file not found, using defaults"
        env_file=".env.example"
    fi
    
    # Build and deploy
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would execute: docker-compose --env-file $env_file up -d"
    else
        docker-compose --env-file "$env_file" up -d --build
        log_success "Docker Compose deployment completed"
        
        # Show running services
        docker-compose ps
    fi
}

# Kubernetes deployment
deploy_kubernetes() {
    local k8s_dir="deployment/k8s"
    
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "Dry run - would create namespace $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    # Apply Kubernetes manifests
    if [[ -d "$k8s_dir" ]]; then
        log_info "Applying Kubernetes manifests from $k8s_dir"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            kubectl apply --dry-run=client -f "$k8s_dir" --namespace "$NAMESPACE"
        else
            kubectl apply -f "$k8s_dir" --namespace "$NAMESPACE"
            
            # Wait for deployment to be ready
            log_info "Waiting for deployment to be ready..."
            kubectl wait --for=condition=available --timeout=300s deployment/llm-tab-cleaner -n "$NAMESPACE"
            
            log_success "Kubernetes deployment completed"
            
            # Show deployment status
            kubectl get pods -n "$NAMESPACE"
        fi
    else
        log_error "Kubernetes manifests directory not found: $k8s_dir"
        exit 1
    fi
}

# Helm deployment
deploy_helm() {
    local chart_dir="deployment/helm/llm-tab-cleaner"
    local values_file="deployment/helm/values-${ENVIRONMENT}.yaml"
    
    log_info "Deploying with Helm..."
    
    if [[ ! -d "$chart_dir" ]]; then
        log_error "Helm chart directory not found: $chart_dir"
        exit 1
    fi
    
    # Use default values if environment-specific values don't exist
    if [[ ! -f "$values_file" ]]; then
        log_warning "Environment-specific values file not found: $values_file"
        values_file="deployment/helm/values.yaml"
    fi
    
    # Install or upgrade Helm release
    local release_name="llm-tab-cleaner-${ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm upgrade --install "$release_name" "$chart_dir" \
            --namespace "$NAMESPACE" --create-namespace \
            --values "$values_file" \
            --set image.tag="$TAG" \
            --dry-run --debug
    else
        helm upgrade --install "$release_name" "$chart_dir" \
            --namespace "$NAMESPACE" --create-namespace \
            --values "$values_file" \
            --set image.tag="$TAG" \
            --wait --timeout=5m
        
        log_success "Helm deployment completed"
        
        # Show release status
        helm status "$release_name" -n "$NAMESPACE"
    fi
}

# AWS ECS deployment
deploy_aws_ecs() {
    local cluster_name="llm-tab-cleaner-${ENVIRONMENT}"
    local service_name="llm-tab-cleaner"
    local task_definition_file="deployment/aws/task-definition.json"
    
    log_info "Deploying to AWS ECS..."
    
    if [[ ! -f "$task_definition_file" ]]; then
        log_error "ECS task definition not found: $task_definition_file"
        exit 1
    fi
    
    # Register new task definition
    local task_definition_arn
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would register ECS task definition"
        task_definition_arn="arn:aws:ecs:region:account:task-definition/family:revision"
    else
        task_definition_arn=$(aws ecs register-task-definition \
            --cli-input-json file://"$task_definition_file" \
            --query 'taskDefinition.taskDefinitionArn' --output text)
        log_success "Registered task definition: $task_definition_arn"
    fi
    
    # Update service
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would update ECS service"
    else
        aws ecs update-service \
            --cluster "$cluster_name" \
            --service "$service_name" \
            --task-definition "$task_definition_arn"
        
        # Wait for deployment
        log_info "Waiting for service to stabilize..."
        aws ecs wait services-stable \
            --cluster "$cluster_name" \
            --services "$service_name"
        
        log_success "ECS deployment completed"
    fi
}

# Azure Container Instances deployment
deploy_azure_aci() {
    local resource_group="llm-tab-cleaner-${ENVIRONMENT}"
    local container_group="llm-tab-cleaner"
    
    log_info "Deploying to Azure Container Instances..."
    
    # Create resource group if it doesn't exist
    if ! az group show --name "$resource_group" >/dev/null 2>&1; then
        log_info "Creating resource group: $resource_group"
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "Dry run - would create resource group $resource_group"
        else
            az group create --name "$resource_group" --location eastus
        fi
    fi
    
    # Deploy container group
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would deploy ACI container group"
    else
        az container create \
            --resource-group "$resource_group" \
            --name "$container_group" \
            --image "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
            --cpu 2 --memory 4 \
            --environment-variables \
                ENVIRONMENT="$ENVIRONMENT" \
                LOG_LEVEL=INFO \
            --secure-environment-variables \
                OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
                ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
        
        log_success "Azure Container Instances deployment completed"
    fi
}

# Local development deployment
deploy_local() {
    log_info "Starting local development deployment..."
    
    # Start local services
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would start local services"
    else
        # Start with development profile
        docker-compose --profile dev up -d --build
        
        log_success "Local deployment completed"
        
        # Show running containers
        docker-compose ps
        
        log_info "Development environment available at:"
        log_info "  Main service: http://localhost:5000"
        log_info "  Documentation: http://localhost:3000"
    fi
}

# Health check after deployment
health_check() {
    local endpoint="$1"
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check on $endpoint..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f "$endpoint/health" >/dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Main function
main() {
    local command=""
    local tag="latest"
    local config_file=""
    local dry_run="false"
    local force="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            docker-compose|kubernetes|aws-ecs|azure-aci|helm|local)
                command="$1"
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            --force)
                force="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$command" ]]; then
        log_error "No deployment command specified"
        show_help
        exit 1
    fi
    
    # Set global variables
    export REGISTRY NAMESPACE ENVIRONMENT
    export TAG="$tag"
    export DRY_RUN="$dry_run"
    export FORCE="$force"
    
    # Load configuration file if specified
    if [[ -n "$config_file" ]]; then
        if [[ -f "$config_file" ]]; then
            log_info "Loading configuration from $config_file"
            # shellcheck source=/dev/null
            source "$config_file"
        else
            log_error "Configuration file not found: $config_file"
            exit 1
        fi
    fi
    
    log_info "Starting $command deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Registry: $REGISTRY"
    log_info "Image: ${IMAGE_NAME}:${tag}"
    log_info "Namespace: $NAMESPACE"
    
    if [[ "$dry_run" == "true" ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run deployment
    case "$command" in
        docker-compose)
            check_prerequisites "$command"
            validate_environment
            deploy_docker_compose
            ;;
        kubernetes)
            check_prerequisites "$command"
            validate_environment
            deploy_kubernetes
            ;;
        helm)
            check_prerequisites "$command"
            validate_environment
            deploy_helm
            ;;
        aws-ecs)
            check_prerequisites "$command"
            validate_environment
            deploy_aws_ecs
            ;;
        azure-aci)
            check_prerequisites "$command"
            validate_environment
            deploy_azure_aci
            ;;
        local)
            check_prerequisites "docker-compose"
            deploy_local
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"