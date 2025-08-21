"""Production Deployment Manifest - Final Deployment Automation.

This module creates the final production deployment package with all necessary
components for enterprise production deployment.

Author: Terry (Terragon Labs)
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class ProductionDeploymentManifest:
    """Creates comprehensive production deployment manifest."""
    
    def __init__(self, project_root: str = "/root/repo"):
        """Initialize deployment manifest generator."""
        self.project_root = Path(project_root)
        self.deployment_dir = self.project_root / "deployment"
        self.deployment_dir.mkdir(exist_ok=True)
        
        # Create deployment structure
        self._create_deployment_structure()
        
    def _create_deployment_structure(self):
        """Create deployment directory structure."""
        dirs = [
            "docker",
            "k8s",
            "helm",
            "terraform",
            "scripts",
            "configs",
            "monitoring"
        ]
        
        for dir_name in dirs:
            (self.deployment_dir / dir_name).mkdir(exist_ok=True)
    
    def generate_production_package(self) -> Dict[str, Any]:
        """Generate complete production deployment package."""
        print("🚀 Generating Production Deployment Package...")
        
        package_info = {
            "package_id": f"prod-deploy-{int(time.time())}",
            "version": "v2.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": []
        }
        
        # 1. Docker Configuration
        docker_config = self._generate_docker_config()
        package_info["components"].append(docker_config)
        
        # 2. Kubernetes Manifests
        k8s_config = self._generate_k8s_config()
        package_info["components"].append(k8s_config)
        
        # 3. Helm Charts
        helm_config = self._generate_helm_config()
        package_info["components"].append(helm_config)
        
        # 4. Terraform Infrastructure
        terraform_config = self._generate_terraform_config()
        package_info["components"].append(terraform_config)
        
        # 5. Monitoring Configuration
        monitoring_config = self._generate_monitoring_config()
        package_info["components"].append(monitoring_config)
        
        # 6. Deployment Scripts
        scripts_config = self._generate_deployment_scripts()
        package_info["components"].append(scripts_config)
        
        # 7. Configuration Files
        configs = self._generate_configuration_files()
        package_info["components"].append(configs)
        
        # Save package manifest
        self._save_package_manifest(package_info)
        
        return package_info
    
    def _generate_docker_config(self) -> Dict[str, Any]:
        """Generate Docker configuration files."""
        print("  📦 Generating Docker configuration...")
        
        # Production Dockerfile
        dockerfile_content = '''# Production Dockerfile for LLM Tab Cleaner
FROM python:3.9-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev postgresql-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-alpine

# Create non-root user
RUN addgroup -g 1001 appgroup && \\
    adduser -D -u 1001 -G appgroup appuser

# Install runtime dependencies
RUN apk add --no-cache postgresql-libs libffi

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup pyproject.toml /app/

WORKDIR /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "llm_tab_cleaner.api"]
'''
        
        # Docker Compose for development
        docker_compose_content = '''version: '3.8'

services:
  llm-tab-cleaner:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=llm_tab_cleaner
      - POSTGRES_USER=app
      - POSTGRES_PASSWORD=secret123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  grafana_data:
'''
        
        # Production Docker Compose
        production_compose_content = '''version: '3.8'

services:
  llm-tab-cleaner:
    image: llm-tab-cleaner:v2.0.0
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=WARNING
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@postgres:5432/llm_tab_cleaner
    secrets:
      - openai_api_key
      - anthropic_api_key
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

secrets:
  openai_api_key:
    external: true
  anthropic_api_key:
    external: true
'''
        
        # Save Docker files
        docker_dir = self.deployment_dir / "docker"
        
        with open(docker_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        with open(docker_dir / "docker-compose.yml", "w") as f:
            f.write(docker_compose_content)
        
        with open(docker_dir / "docker-compose.production.yml", "w") as f:
            f.write(production_compose_content)
        
        return {
            "name": "Docker Configuration",
            "files": ["Dockerfile", "docker-compose.yml", "docker-compose.production.yml"],
            "path": "deployment/docker/",
            "status": "generated"
        }
    
    def _generate_k8s_config(self) -> Dict[str, Any]:
        """Generate Kubernetes manifests."""
        print("  ☸️  Generating Kubernetes configuration...")
        
        k8s_dir = self.deployment_dir / "k8s"
        
        # Namespace
        namespace_yaml = '''apiVersion: v1
kind: Namespace
metadata:
  name: llm-tab-cleaner
  labels:
    app: llm-tab-cleaner
    version: v2.0.0
'''
        
        # Deployment
        deployment_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-tab-cleaner
  namespace: llm-tab-cleaner
  labels:
    app: llm-tab-cleaner
    version: v2.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-tab-cleaner
  template:
    metadata:
      labels:
        app: llm-tab-cleaner
        version: v2.0.0
    spec:
      containers:
      - name: llm-tab-cleaner
        image: llm-tab-cleaner:v2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        envFrom:
        - secretRef:
            name: llm-tab-cleaner-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
'''
        
        # Service
        service_yaml = '''apiVersion: v1
kind: Service
metadata:
  name: llm-tab-cleaner-service
  namespace: llm-tab-cleaner
  labels:
    app: llm-tab-cleaner
spec:
  selector:
    app: llm-tab-cleaner
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
'''
        
        # Ingress
        ingress_yaml = '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-tab-cleaner-ingress
  namespace: llm-tab-cleaner
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.llm-tab-cleaner.com
    secretName: llm-tab-cleaner-tls
  rules:
  - host: api.llm-tab-cleaner.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-tab-cleaner-service
            port:
              number: 80
'''
        
        # ConfigMap
        configmap_yaml = '''apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-tab-cleaner-config
  namespace: llm-tab-cleaner
data:
  config.yaml: |
    app:
      name: "LLM Tab Cleaner"
      version: "v2.0.0"
      environment: "production"
    
    logging:
      level: "INFO"
      format: "json"
    
    cache:
      redis_url: "redis://redis-service:6379"
      ttl: 3600
    
    performance:
      max_workers: 10
      timeout: 30
      batch_size: 100
'''
        
        # HPA (Horizontal Pod Autoscaler)
        hpa_yaml = '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-tab-cleaner-hpa
  namespace: llm-tab-cleaner
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-tab-cleaner
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
        
        # Save K8s files
        files = {
            "namespace.yaml": namespace_yaml,
            "deployment.yaml": deployment_yaml,
            "service.yaml": service_yaml,
            "ingress.yaml": ingress_yaml,
            "configmap.yaml": configmap_yaml,
            "hpa.yaml": hpa_yaml
        }
        
        for filename, content in files.items():
            with open(k8s_dir / filename, "w") as f:
                f.write(content)
        
        return {
            "name": "Kubernetes Configuration",
            "files": list(files.keys()),
            "path": "deployment/k8s/",
            "status": "generated"
        }
    
    def _generate_helm_config(self) -> Dict[str, Any]:
        """Generate Helm chart."""
        print("  ⛵ Generating Helm chart...")
        
        helm_dir = self.deployment_dir / "helm" / "llm-tab-cleaner"
        helm_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = '''apiVersion: v2
name: llm-tab-cleaner
description: A Helm chart for LLM Tab Cleaner autonomous production system
type: application
version: 0.1.0
appVersion: "v2.0.0"
keywords:
  - llm
  - data-cleaning
  - ai
  - etl
maintainers:
  - name: Terry
    email: terry@terragonlabs.com
sources:
  - https://github.com/terragonlabs/llm-tab-cleaner
'''
        
        # values.yaml
        values_yaml = '''# Default values for llm-tab-cleaner
replicaCount: 3

image:
  repository: llm-tab-cleaner
  pullPolicy: IfNotPresent
  tag: "v2.0.0"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 1001
  runAsNonRoot: true
  runAsUser: 1001

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1001

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.llm-tab-cleaner.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: llm-tab-cleaner-tls
      hosts:
        - api.llm-tab-cleaner.com

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

# Application configuration
config:
  environment: production
  logLevel: INFO
  
# External services
redis:
  enabled: true
  host: redis-service
  port: 6379

postgresql:
  enabled: false
  host: ""
  port: 5432
  database: llm_tab_cleaner

# Monitoring
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true
'''
        
        # Save Helm files
        with open(helm_dir / "Chart.yaml", "w") as f:
            f.write(chart_yaml)
        
        with open(helm_dir / "values.yaml", "w") as f:
            f.write(values_yaml)
        
        return {
            "name": "Helm Chart",
            "files": ["Chart.yaml", "values.yaml"],
            "path": "deployment/helm/llm-tab-cleaner/",
            "status": "generated"
        }
    
    def _generate_terraform_config(self) -> Dict[str, Any]:
        """Generate Terraform infrastructure."""
        print("  🏗️  Generating Terraform configuration...")
        
        terraform_dir = self.deployment_dir / "terraform"
        
        # Main configuration
        main_tf = '''# Main Terraform configuration for LLM Tab Cleaner infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Data sources
data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

data "aws_availability_zones" "available" {}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  single_nat_gateway = false
  
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "${var.project_name}-cluster"
  cluster_version = var.kubernetes_version
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups
  eks_managed_node_groups = {
    main = {
      min_size     = 3
      max_size     = 20
      desired_size = 6
      
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
  }
  
  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-db"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.db_instance_class
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  skip_final_snapshot = true
  deletion_protection = var.environment == "production"
  
  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = "${var.project_name}-redis"
  description                  = "Redis cluster for ${var.project_name}"
  
  node_type                    = var.redis_node_type
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  num_cache_clusters           = 2
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  
  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
'''
        
        # Variables
        variables_tf = '''# Variables for LLM Tab Cleaner infrastructure

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "llm-tab-cleaner"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.27"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "llm_tab_cleaner"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "app"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}
'''
        
        # Outputs
        outputs_tf = '''# Outputs for LLM Tab Cleaner infrastructure

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}
'''
        
        # Save Terraform files
        files = {
            "main.tf": main_tf,
            "variables.tf": variables_tf,
            "outputs.tf": outputs_tf
        }
        
        for filename, content in files.items():
            with open(terraform_dir / filename, "w") as f:
                f.write(content)
        
        return {
            "name": "Terraform Infrastructure",
            "files": list(files.keys()),
            "path": "deployment/terraform/",
            "status": "generated"
        }
    
    def _generate_monitoring_config(self) -> Dict[str, Any]:
        """Generate monitoring configuration."""
        print("  📊 Generating monitoring configuration...")
        
        monitoring_dir = self.deployment_dir / "monitoring"
        
        # Prometheus configuration
        prometheus_yml = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'llm-tab-cleaner'
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: llm-tab-cleaner-service
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        action: keep
        regex: metrics

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
'''
        
        # Grafana dashboard
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "title": "LLM Tab Cleaner - Production Metrics",
                "tags": ["llm", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                                "legendFormat": "5xx errors/sec"
                            }
                        ]
                    }
                ]
            }
        }
        
        # Alert rules
        alerts_yml = '''groups:
  - name: llm-tab-cleaner.rules
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} is restarting frequently"
'''
        
        # Save monitoring files
        with open(monitoring_dir / "prometheus.yml", "w") as f:
            f.write(prometheus_yml)
        
        with open(monitoring_dir / "grafana-dashboard.json", "w") as f:
            json.dump(grafana_dashboard, f, indent=2)
        
        with open(monitoring_dir / "alerts.yml", "w") as f:
            f.write(alerts_yml)
        
        return {
            "name": "Monitoring Configuration",
            "files": ["prometheus.yml", "grafana-dashboard.json", "alerts.yml"],
            "path": "deployment/monitoring/",
            "status": "generated"
        }
    
    def _generate_deployment_scripts(self) -> Dict[str, Any]:
        """Generate deployment scripts."""
        print("  📜 Generating deployment scripts...")
        
        scripts_dir = self.deployment_dir / "scripts"
        
        # Main deployment script
        deploy_sh = '''#!/bin/bash
# Production deployment script for LLM Tab Cleaner

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

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
helm upgrade --install llm-tab-cleaner $CHART_PATH \\
    --namespace $NAMESPACE \\
    --values $VALUES_FILE \\
    --timeout 10m \\
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
'''
        
        # Rollback script
        rollback_sh = '''#!/bin/bash
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
'''
        
        # Health check script
        health_check_sh = '''#!/bin/bash
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
'''
        
        # Save scripts
        scripts = {
            "deploy.sh": deploy_sh,
            "rollback.sh": rollback_sh,
            "health-check.sh": health_check_sh
        }
        
        for filename, content in scripts.items():
            script_path = scripts_dir / filename
            with open(script_path, "w") as f:
                f.write(content)
            
            # Make scripts executable
            os.chmod(script_path, 0o755)
        
        return {
            "name": "Deployment Scripts",
            "files": list(scripts.keys()),
            "path": "deployment/scripts/",
            "status": "generated"
        }
    
    def _generate_configuration_files(self) -> Dict[str, Any]:
        """Generate configuration files."""
        print("  ⚙️  Generating configuration files...")
        
        configs_dir = self.deployment_dir / "configs"
        
        # Production values for Helm
        production_values = '''# Production values for LLM Tab Cleaner
replicaCount: 5

image:
  repository: llm-tab-cleaner
  tag: "v2.0.0"
  pullPolicy: Always

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 50
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api.llm-tab-cleaner.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: llm-tab-cleaner-tls
      hosts:
        - api.llm-tab-cleaner.com

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
  prometheusRule:
    enabled: true

config:
  environment: production
  logLevel: WARNING
  maxWorkers: 20
  cacheEnabled: true
  cacheSize: 10000
'''
        
        # Environment configuration
        env_config = '''# Environment configuration for LLM Tab Cleaner production
ENV=production
LOG_LEVEL=WARNING
DEBUG=false

# Database
DATABASE_URL=postgresql://app:${DB_PASSWORD}@postgres-service:5432/llm_tab_cleaner
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://redis-service:6379
REDIS_POOL_SIZE=10

# LLM Providers
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3

# Performance
MAX_WORKERS=20
BATCH_SIZE=100
CACHE_SIZE=10000
CACHE_TTL=3600

# Security
SECRET_KEY=${SECRET_KEY}
JWT_EXPIRY=3600
CORS_ORIGINS=https://app.llm-tab-cleaner.com

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8001

# Global deployment
DEPLOYMENT_REGION=${DEPLOYMENT_REGION}
COMPLIANCE_MODE=${COMPLIANCE_MODE}
DATA_RESIDENCY=${DATA_RESIDENCY}
'''
        
        # Secrets template
        secrets_template = '''apiVersion: v1
kind: Secret
metadata:
  name: llm-tab-cleaner-secrets
  namespace: llm-tab-cleaner
type: Opaque
data:
  # Base64 encoded values - replace with actual encoded secrets
  db-password: <BASE64_ENCODED_DB_PASSWORD>
  openai-api-key: <BASE64_ENCODED_OPENAI_KEY>
  anthropic-api-key: <BASE64_ENCODED_ANTHROPIC_KEY>
  secret-key: <BASE64_ENCODED_SECRET_KEY>
'''
        
        # Save configuration files
        configs = {
            "values.production.yaml": production_values,
            "production.env": env_config,
            "secrets.yaml.template": secrets_template
        }
        
        for filename, content in configs.items():
            with open(configs_dir / filename, "w") as f:
                f.write(content)
        
        return {
            "name": "Configuration Files",
            "files": list(configs.keys()),
            "path": "deployment/configs/",
            "status": "generated"
        }
    
    def _save_package_manifest(self, package_info: Dict[str, Any]):
        """Save the complete package manifest."""
        manifest_file = self.deployment_dir / "deployment-manifest.json"
        
        with open(manifest_file, "w") as f:
            json.dump(package_info, f, indent=2, default=str)
        
        # Also save a readable summary
        summary_file = self.deployment_dir / "deployment-summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Deployment Package Summary\n")
            f.write(f"Package ID: {package_info['package_id']}\n")
            f.write(f"Version: {package_info['version']}\n")
            f.write(f"Generated: {package_info['timestamp']}\n")
            f.write(f"Components: {len(package_info['components'])}\n\n")
            for component in package_info['components']:
                f.write(f"- {component['name']}: {len(component['files'])} files\n")
        
        print(f"📋 Package manifest saved: {manifest_file}")


def main():
    """Generate production deployment package."""
    print("🚀 LLM Tab Cleaner - Production Deployment Package Generator")
    print("=" * 70)
    
    # Initialize manifest generator
    manifest_generator = ProductionDeploymentManifest()
    
    # Generate complete package
    package_info = manifest_generator.generate_production_package()
    
    print("\n" + "=" * 70)
    print("📦 PRODUCTION PACKAGE GENERATED SUCCESSFULLY")
    print("=" * 70)
    
    print(f"Package ID: {package_info['package_id']}")
    print(f"Version: {package_info['version']}")
    print(f"Components Generated: {len(package_info['components'])}")
    
    print("\n📋 Component Summary:")
    for component in package_info['components']:
        print(f"  ✅ {component['name']}")
        print(f"     Path: {component['path']}")
        print(f"     Files: {len(component['files'])} files")
    
    print("\n🚀 Ready for Production Deployment!")
    print("   Use the deployment scripts in deployment/scripts/ to deploy")
    print("   Review configuration files in deployment/configs/ before deployment")
    print("\n" + "=" * 70)
    
    return package_info


if __name__ == "__main__":
    main()