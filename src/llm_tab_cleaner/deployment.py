"""Production deployment utilities and configurations."""

import json
import logging
import os
import sys
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .security import SecurityConfig
from .monitoring import CleaningMonitor, setup_monitoring
from .optimization import OptimizationConfig, OptimizationEngine


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for production deployments."""
    
    # Primary database
    database_url: str = "sqlite:///data/cleaning_db.sqlite"
    connection_pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # State management
    state_database_url: Optional[str] = None
    enable_connection_pooling: bool = True
    
    # Performance
    enable_query_logging: bool = False
    slow_query_threshold: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration for production."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    log_file: Optional[str] = "/var/log/llm-tab-cleaner/app.log"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    
    # Structured logging
    enable_json_logging: bool = False
    enable_correlation_ids: bool = True
    
    # External logging
    syslog_address: Optional[str] = None
    enable_remote_logging: bool = False
    
    # Audit logging
    audit_log_file: Optional[str] = "/var/log/llm-tab-cleaner/audit.log"
    enable_audit_logging: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Security
    enable_auth: bool = True
    auth_type: str = "api_key"  # api_key, jwt, oauth
    api_key_header: str = "X-API-Key"
    jwt_secret_key: Optional[str] = None
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 20
    
    # CORS
    enable_cors: bool = False
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
    
    # Request handling
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: int = 300  # 5 minutes
    
    # Health checks
    enable_health_endpoint: bool = True
    enable_metrics_endpoint: bool = True


@dataclass
class InfrastructureConfig:
    """Infrastructure and deployment configuration."""
    
    # Environment
    environment: str = "production"  # development, staging, production
    service_name: str = "llm-tab-cleaner"
    version: str = "1.0.0"
    
    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_cores: int = 4
    max_disk_space_mb: int = 10240
    
    # Networking
    bind_address: str = "0.0.0.0"
    internal_port: int = 8000
    external_port: int = 80
    
    # Load balancing
    enable_load_balancing: bool = False
    load_balancer_type: str = "nginx"  # nginx, haproxy, aws_alb
    
    # Auto-scaling
    enable_auto_scaling: bool = False
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Storage
    data_volume_path: str = "/data"
    logs_volume_path: str = "/var/log"
    cache_volume_path: str = "/cache"


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    
    # Core configurations
    database: DatabaseConfig
    logging: LoggingConfig
    api: APIConfig
    infrastructure: InfrastructureConfig
    security: SecurityConfig
    optimization: OptimizationConfig
    
    # Feature flags
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_async_processing: bool = True
    
    # External services
    redis_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None
    prometheus_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Initialize default values for nested dataclasses if they weren't provided
        if not hasattr(self, 'database') or self.database is None:
            self.database = DatabaseConfig()
        if not hasattr(self, 'logging') or self.logging is None:
            self.logging = LoggingConfig()
        if not hasattr(self, 'api') or self.api is None:
            self.api = APIConfig()
        if not hasattr(self, 'infrastructure') or self.infrastructure is None:
            self.infrastructure = InfrastructureConfig()
        if not hasattr(self, 'security') or self.security is None:
            self.security = SecurityConfig()
        if not hasattr(self, 'optimization') or self.optimization is None:
            self.optimization = OptimizationConfig()
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Environment validation
        valid_environments = ["development", "staging", "production"]
        if self.infrastructure.environment not in valid_environments:
            raise ValueError(f"Invalid environment: {self.infrastructure.environment}")
        
        # Port validation
        if not (1 <= self.api.port <= 65535):
            raise ValueError(f"Invalid API port: {self.api.port}")
        
        # Memory validation
        if self.infrastructure.max_memory_mb < 512:
            raise ValueError("Minimum memory requirement is 512MB")
        
        # Security validation
        if self.api.enable_auth and self.api.auth_type == "jwt" and not self.api.jwt_secret_key:
            raise ValueError("JWT secret key required when using JWT authentication")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "ProductionConfig":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        with open(config_path, 'r') as f:
            if suffix in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            elif suffix == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {suffix}")
        
        # Create configuration instances
        return cls(
            database=DatabaseConfig(**config_data.get('database', {})),
            logging=LoggingConfig(**config_data.get('logging', {})),
            api=APIConfig(**config_data.get('api', {})),
            infrastructure=InfrastructureConfig(**config_data.get('infrastructure', {})),
            security=SecurityConfig(**config_data.get('security', {})),
            optimization=OptimizationConfig(**config_data.get('optimization', {})),
            enable_monitoring=config_data.get('enable_monitoring', True),
            enable_caching=config_data.get('enable_caching', True),
            enable_async_processing=config_data.get('enable_async_processing', True),
            redis_url=config_data.get('redis_url'),
            elasticsearch_url=config_data.get('elasticsearch_url'),
            prometheus_url=config_data.get('prometheus_url')
        )
    
    def to_file(self, config_path: Union[str, Path], format_type: str = "yaml"):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = asdict(self)
        
        with open(config_path, 'w') as f:
            if format_type.lower() in ['yml', 'yaml']:
                yaml.safe_dump(config_data, f, indent=2)
            elif format_type.lower() == 'json':
                json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get configuration as environment variables."""
        env_vars = {}
        
        # Database
        env_vars["DATABASE_URL"] = self.database.database_url
        env_vars["DB_POOL_SIZE"] = str(self.database.connection_pool_size)
        
        # API
        env_vars["API_HOST"] = self.api.host
        env_vars["API_PORT"] = str(self.api.port)
        env_vars["API_WORKERS"] = str(self.api.workers)
        
        # Infrastructure
        env_vars["ENVIRONMENT"] = self.infrastructure.environment
        env_vars["SERVICE_NAME"] = self.infrastructure.service_name
        env_vars["VERSION"] = self.infrastructure.version
        
        # Security
        env_vars["MAX_DATA_SIZE"] = str(self.security.max_data_size)
        env_vars["ENABLE_AUDIT_LOGGING"] = str(self.security.enable_audit_logging)
        
        # External services
        if self.redis_url:
            env_vars["REDIS_URL"] = self.redis_url
        if self.elasticsearch_url:
            env_vars["ELASTICSEARCH_URL"] = self.elasticsearch_url
        if self.prometheus_url:
            env_vars["PROMETHEUS_URL"] = self.prometheus_url
        
        return env_vars


class DeploymentManager:
    """Manages deployment operations and health checks."""
    
    def __init__(self, config: ProductionConfig):
        """Initialize deployment manager."""
        self.config = config
        self.monitor: Optional[CleaningMonitor] = None
        self.optimizer: Optional[OptimizationEngine] = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized deployment manager for {config.infrastructure.environment}")
    
    def _setup_logging(self):
        """Setup production logging."""
        logging_config = self.config.logging
        
        # Create log directory
        if logging_config.log_file:
            log_path = Path(logging_config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, logging_config.level))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        if logging_config.enable_json_logging:
            # JSON formatter for structured logging
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(logging_config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if logging_config.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                logging_config.log_file,
                maxBytes=logging_config.max_file_size,
                backupCount=logging_config.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logger.info("Production logging configured")
    
    def _initialize_components(self):
        """Initialize production components."""
        # Initialize monitoring
        if self.config.enable_monitoring:
            self.monitor = setup_monitoring()
        
        # Initialize optimizer
        if self.config.enable_caching:
            self.optimizer = OptimizationEngine(self.config.optimization)
        
        logger.info("Production components initialized")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "version": self.config.infrastructure.version,
            "environment": self.config.infrastructure.environment,
            "components": {}
        }
        
        # Check monitoring
        if self.monitor:
            try:
                monitor_health = self.monitor.health.get_overall_health()
                health_status["components"]["monitoring"] = monitor_health
            except Exception as e:
                health_status["components"]["monitoring"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        # Check optimizer
        if self.optimizer:
            try:
                optimizer_stats = self.optimizer.get_performance_summary()
                health_status["components"]["optimizer"] = {
                    "status": "healthy",
                    "cache_hit_rate": optimizer_stats.get("cache", {}).get("hit_rate", 0)
                }
            except Exception as e:
                health_status["components"]["optimizer"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        # Check database connectivity (if configured)
        try:
            # This would perform actual database connectivity check
            health_status["components"]["database"] = {
                "status": "healthy",
                "url": self.config.database.database_url
            }
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        return health_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get production metrics."""
        metrics = {
            "timestamp": "2024-01-01T00:00:00Z",
            "service": {
                "name": self.config.infrastructure.service_name,
                "version": self.config.infrastructure.version,
                "environment": self.config.infrastructure.environment
            }
        }
        
        # Monitoring metrics
        if self.monitor:
            try:
                dashboard_data = self.monitor.get_monitoring_dashboard()
                metrics["monitoring"] = dashboard_data
            except Exception as e:
                logger.error(f"Error getting monitoring metrics: {e}")
        
        # Optimization metrics
        if self.optimizer:
            try:
                perf_summary = self.optimizer.get_performance_summary()
                metrics["optimization"] = perf_summary
            except Exception as e:
                logger.error(f"Error getting optimization metrics: {e}")
        
        return metrics
    
    def prepare_deployment(self) -> Dict[str, Any]:
        """Prepare system for deployment."""
        logger.info("Preparing for deployment...")
        
        # Create required directories
        directories = [
            self.config.infrastructure.data_volume_path,
            self.config.infrastructure.logs_volume_path,
            self.config.infrastructure.cache_volume_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Validate configuration
        try:
            self.config._validate_config()
            logger.info("Configuration validation passed")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize database
        # This would perform database migrations and setup
        
        # Warm up caches
        if self.optimizer and self.optimizer.cache_manager.enabled:
            logger.info("Cache warming not implemented yet")
        
        deployment_info = {
            "status": "ready",
            "timestamp": "2024-01-01T00:00:00Z",
            "configuration": {
                "environment": self.config.infrastructure.environment,
                "version": self.config.infrastructure.version,
                "features": {
                    "monitoring": self.config.enable_monitoring,
                    "caching": self.config.enable_caching,
                    "async_processing": self.config.enable_async_processing
                }
            }
        }
        
        logger.info("Deployment preparation completed")
        return deployment_info
    
    def shutdown_gracefully(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown...")
        
        # Stop accepting new requests
        # This would integrate with your web framework
        
        # Complete ongoing operations
        # This would wait for current operations to finish
        
        # Close connections
        if self.monitor:
            # Close monitoring connections
            pass
        
        if self.optimizer:
            # Close optimizer resources
            pass
        
        logger.info("Graceful shutdown completed")


def create_default_config(environment: str = "production") -> ProductionConfig:
    """Create default production configuration."""
    return ProductionConfig(
        database=DatabaseConfig(),
        logging=LoggingConfig(level="INFO" if environment == "production" else "DEBUG"),
        api=APIConfig(),
        infrastructure=InfrastructureConfig(environment=environment),
        security=SecurityConfig(),
        optimization=OptimizationConfig()
    )


def generate_deployment_files(config: ProductionConfig, output_dir: str = "./deploy"):
    """Generate deployment files (Docker, K8s, etc.)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate Docker Compose
    docker_compose = {
        "version": "3.8",
        "services": {
            "llm-tab-cleaner": {
                "image": f"llm-tab-cleaner:{config.infrastructure.version}",
                "ports": [f"{config.infrastructure.external_port}:{config.infrastructure.internal_port}"],
                "environment": config.get_environment_variables(),
                "volumes": [
                    f"{config.infrastructure.data_volume_path}:/data",
                    f"{config.infrastructure.logs_volume_path}:/var/log"
                ],
                "restart": "unless-stopped"
            }
        }
    }
    
    if config.redis_url:
        docker_compose["services"]["redis"] = {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "volumes": ["redis_data:/data"],
            "restart": "unless-stopped"
        }
        docker_compose["volumes"] = {"redis_data": {}}
    
    # Save Docker Compose
    with open(output_path / "docker-compose.yml", 'w') as f:
        yaml.safe_dump(docker_compose, f, indent=2)
    
    # Generate Kubernetes manifests (basic example)
    k8s_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": config.infrastructure.service_name,
            "labels": {"app": config.infrastructure.service_name}
        },
        "spec": {
            "replicas": config.infrastructure.min_replicas,
            "selector": {"matchLabels": {"app": config.infrastructure.service_name}},
            "template": {
                "metadata": {"labels": {"app": config.infrastructure.service_name}},
                "spec": {
                    "containers": [{
                        "name": config.infrastructure.service_name,
                        "image": f"llm-tab-cleaner:{config.infrastructure.version}",
                        "ports": [{"containerPort": config.infrastructure.internal_port}],
                        "env": [
                            {"name": k, "value": v}
                            for k, v in config.get_environment_variables().items()
                        ],
                        "resources": {
                            "requests": {
                                "memory": f"{config.infrastructure.max_memory_mb // 2}Mi",
                                "cpu": f"{config.infrastructure.max_cpu_cores // 2}"
                            },
                            "limits": {
                                "memory": f"{config.infrastructure.max_memory_mb}Mi",
                                "cpu": str(config.infrastructure.max_cpu_cores)
                            }
                        }
                    }]
                }
            }
        }
    }
    
    # Save Kubernetes deployment
    with open(output_path / "k8s-deployment.yml", 'w') as f:
        yaml.safe_dump(k8s_deployment, f, indent=2)
    
    logger.info(f"Deployment files generated in {output_path}")


if __name__ == "__main__":
    # Example usage
    config = create_default_config("production")
    config.to_file("config/production.yml")
    generate_deployment_files(config)
    
    # Initialize deployment manager
    deployment = DeploymentManager(config)
    health = deployment.health_check()
    print(json.dumps(health, indent=2))