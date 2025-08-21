"""Advanced Security Framework - Generation 2 Security Enhancement.

This module implements comprehensive security features for the autonomous
production system, including threat detection, access control, and audit trails.

Features:
- Real-time threat detection and response
- Advanced access control and authorization
- Comprehensive audit logging and compliance
- Automated security scanning and validation
- Encrypted data handling and transmission
- Security incident response automation

Author: Terry (Terragon Labs)
"""

import logging
import hashlib
import hmac
import secrets
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict, deque
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipaddress
import numpy as np

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "auth_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    UNUSUAL_ACCESS_PATTERN = "unusual_access_pattern"
    MALICIOUS_PAYLOAD = "malicious_payload"


class AccessLevel(Enum):
    """Access control levels."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    resource: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


@dataclass
class AccessToken:
    """Secure access token."""
    token_id: str
    user_id: str
    access_level: AccessLevel
    resource_patterns: Set[str]
    expires_at: float
    created_at: float
    last_used: float = 0.0
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityProfile:
    """User security profile."""
    user_id: str
    access_level: AccessLevel
    allowed_resources: Set[str]
    rate_limits: Dict[str, int]
    last_login: float = 0.0
    failed_attempts: int = 0
    locked_until: float = 0.0
    activity_pattern: Dict[str, Any] = field(default_factory=dict)


class AdvancedEncryption:
    """Advanced encryption utilities for data protection."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize encryption with master key."""
        if master_key:
            self.key = master_key
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        self.salt = secrets.token_bytes(16)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt encrypted data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary as JSON."""
        json_data = json.dumps(data, sort_keys=True)
        return self.encrypt_data(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt and parse JSON dictionary."""
        json_data = self.decrypt_data(encrypted_data)
        return json.loads(json_data)
    
    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash password with salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return base64.b64encode(hashed).decode(), salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash."""
        test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return base64.b64encode(test_hash).decode() == hashed
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)


class ThreatDetectionEngine:
    """Real-time threat detection and analysis."""
    
    def __init__(self):
        self.threat_patterns = {
            'sql_injection': [
                r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)",
                r"(?i)('\s*or\s*'1'\s*=\s*'1|admin'\s*--|\d+\s*or\s*\d+)"
            ],
            'xss_attempt': [
                r"(?i)(<script|javascript:|onload=|onerror=)",
                r"(?i)(alert\(|document\.cookie|window\.location)"
            ],
            'command_injection': [
                r"(?i)(;\s*rm\s|;\s*cat\s|;\s*ls\s|&&\s*rm)",
                r"(?i)(\|\s*nc\s|\|\s*curl\s|\|\s*wget)"
            ],
            'path_traversal': [
                r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                r"(?i)(\/etc\/passwd|\/etc\/shadow|\.\.\/.*\.conf)"
            ]
        }
        
        self.behavioral_baselines = {}
        self.anomaly_scores = deque(maxlen=1000)
        self.threat_history = deque(maxlen=10000)
        
        # Rate limiting patterns
        self.rate_patterns = defaultdict(lambda: defaultdict(list))
        
        # IP reputation tracking
        self.ip_reputation = defaultdict(lambda: {'score': 0, 'events': []})
    
    def analyze_request(
        self,
        request_data: Dict[str, Any],
        user_profile: Optional[SecurityProfile] = None
    ) -> Tuple[ThreatLevel, List[str]]:
        """Analyze request for potential threats."""
        threats = []
        max_threat_level = ThreatLevel.LOW
        
        # Content analysis
        content_threats = self._analyze_content(request_data)
        threats.extend(content_threats)
        
        # Behavioral analysis
        if user_profile:
            behavioral_threats = self._analyze_behavior(request_data, user_profile)
            threats.extend(behavioral_threats)
        
        # Rate limiting analysis
        rate_threats = self._analyze_rate_patterns(request_data)
        threats.extend(rate_threats)
        
        # IP reputation analysis
        ip_threats = self._analyze_ip_reputation(request_data.get('source_ip'))
        threats.extend(ip_threats)
        
        # Determine overall threat level
        if any('critical' in threat.lower() for threat in threats):
            max_threat_level = ThreatLevel.CRITICAL
        elif any('high' in threat.lower() for threat in threats):
            max_threat_level = ThreatLevel.HIGH
        elif any('medium' in threat.lower() for threat in threats):
            max_threat_level = ThreatLevel.MEDIUM
        
        return max_threat_level, threats
    
    def _analyze_content(self, request_data: Dict[str, Any]) -> List[str]:
        """Analyze request content for malicious patterns."""
        threats = []
        
        # Get all string values from request
        content_strings = []
        self._extract_strings(request_data, content_strings)
        
        # Check against threat patterns
        for content in content_strings:
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content):
                        threats.append(f"HIGH: {threat_type.upper()} pattern detected")
                        break
        
        return threats
    
    def _extract_strings(self, obj: Any, strings: List[str]):
        """Recursively extract strings from nested objects."""
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_strings(value, strings)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_strings(item, strings)
    
    def _analyze_behavior(
        self,
        request_data: Dict[str, Any],
        user_profile: SecurityProfile
    ) -> List[str]:
        """Analyze behavioral patterns for anomalies."""
        threats = []
        
        user_id = user_profile.user_id
        current_time = time.time()
        
        # Update behavioral baseline
        if user_id not in self.behavioral_baselines:
            self.behavioral_baselines[user_id] = {
                'request_times': deque(maxlen=100),
                'resources_accessed': defaultdict(int),
                'request_sizes': deque(maxlen=50),
                'typical_hours': set()
            }
        
        baseline = self.behavioral_baselines[user_id]
        
        # Time-based analysis
        hour = datetime.fromtimestamp(current_time).hour
        baseline['typical_hours'].add(hour)
        
        # Unusual time access
        if len(baseline['typical_hours']) > 10:
            if hour not in baseline['typical_hours']:
                threats.append("MEDIUM: Unusual time access pattern")
        
        # Request frequency analysis
        baseline['request_times'].append(current_time)
        if len(baseline['request_times']) >= 10:
            recent_requests = [t for t in baseline['request_times'] if current_time - t < 60]
            if len(recent_requests) > 30:  # More than 30 requests per minute
                threats.append("HIGH: Suspicious request frequency")
        
        # Resource access pattern analysis
        resource = request_data.get('resource', 'unknown')
        baseline['resources_accessed'][resource] += 1
        
        # Check for unusual resource access
        total_requests = sum(baseline['resources_accessed'].values())
        if total_requests > 50:
            resource_frequency = baseline['resources_accessed'][resource] / total_requests
            if resource_frequency > 0.8:  # Accessing single resource >80% of time
                threats.append("MEDIUM: Unusual resource access concentration")
        
        return threats
    
    def _analyze_rate_patterns(self, request_data: Dict[str, Any]) -> List[str]:
        """Analyze rate limiting patterns."""
        threats = []
        
        source_ip = request_data.get('source_ip', 'unknown')
        user_id = request_data.get('user_id', 'anonymous')
        current_time = time.time()
        
        # Track IP-based rate limiting
        self.rate_patterns['ip'][source_ip].append(current_time)
        ip_requests = [t for t in self.rate_patterns['ip'][source_ip] if current_time - t < 3600]
        self.rate_patterns['ip'][source_ip] = ip_requests
        
        if len(ip_requests) > 1000:  # More than 1000 requests per hour
            threats.append("CRITICAL: IP rate limit violation")
        elif len(ip_requests) > 500:
            threats.append("HIGH: High IP request rate")
        
        # Track user-based rate limiting
        if user_id != 'anonymous':
            self.rate_patterns['user'][user_id].append(current_time)
            user_requests = [t for t in self.rate_patterns['user'][user_id] if current_time - t < 3600]
            self.rate_patterns['user'][user_id] = user_requests
            
            if len(user_requests) > 2000:  # More than 2000 requests per hour
                threats.append("HIGH: User rate limit violation")
        
        return threats
    
    def _analyze_ip_reputation(self, source_ip: Optional[str]) -> List[str]:
        """Analyze IP reputation and geolocation."""
        threats = []
        
        if not source_ip:
            return threats
        
        try:
            ip = ipaddress.ip_address(source_ip)
            
            # Check for private/local IPs (less suspicious)
            if ip.is_private or ip.is_loopback:
                return threats
            
            # Simple reputation scoring
            reputation = self.ip_reputation[source_ip]
            
            # Update reputation based on recent activity
            current_time = time.time()
            recent_events = [e for e in reputation['events'] if current_time - e['timestamp'] < 86400]
            
            # Calculate reputation score
            threat_score = sum(e.get('threat_weight', 1) for e in recent_events)
            reputation['score'] = max(0, 100 - threat_score)
            
            if reputation['score'] < 20:
                threats.append("CRITICAL: Low IP reputation score")
            elif reputation['score'] < 50:
                threats.append("HIGH: Suspicious IP reputation")
            
        except ValueError:
            threats.append("MEDIUM: Invalid IP address format")
        
        return threats
    
    def update_ip_reputation(self, source_ip: str, event_type: SecurityEventType, threat_weight: int = 1):
        """Update IP reputation based on security event."""
        if source_ip:
            self.ip_reputation[source_ip]['events'].append({
                'timestamp': time.time(),
                'event_type': event_type.value,
                'threat_weight': threat_weight
            })
    
    def get_threat_analytics(self) -> Dict[str, Any]:
        """Get threat detection analytics."""
        return {
            'total_threats_detected': len(self.threat_history),
            'recent_threats': len([t for t in self.threat_history if time.time() - t.timestamp < 3600]),
            'top_threat_types': self._get_top_threat_types(),
            'ip_reputation_summary': self._get_ip_reputation_summary(),
            'behavioral_anomalies': len(self.anomaly_scores),
            'rate_limit_violations': self._count_rate_violations()
        }
    
    def _get_top_threat_types(self) -> Dict[str, int]:
        """Get most common threat types."""
        threat_counts = defaultdict(int)
        for event in list(self.threat_history)[-1000:]:  # Last 1000 events
            threat_counts[event.event_type.value] += 1
        return dict(sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _get_ip_reputation_summary(self) -> Dict[str, Any]:
        """Get IP reputation summary."""
        low_rep_ips = sum(1 for rep in self.ip_reputation.values() if rep['score'] < 50)
        total_ips = len(self.ip_reputation)
        
        return {
            'total_tracked_ips': total_ips,
            'low_reputation_ips': low_rep_ips,
            'reputation_percentage': (total_ips - low_rep_ips) / max(1, total_ips) * 100
        }
    
    def _count_rate_violations(self) -> int:
        """Count recent rate limit violations."""
        current_time = time.time()
        violations = 0
        
        for ip_requests in self.rate_patterns['ip'].values():
            recent = [t for t in ip_requests if current_time - t < 3600]
            if len(recent) > 500:
                violations += 1
        
        return violations


class AccessControlManager:
    """Advanced access control and authorization."""
    
    def __init__(self, encryption: AdvancedEncryption):
        self.encryption = encryption
        self.security_profiles = {}
        self.active_tokens = {}
        self.resource_policies = {}
        self.access_logs = deque(maxlen=10000)
        
        # Default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default resource access policies."""
        self.resource_policies = {
            'system.*': AccessLevel.SYSTEM,
            'admin.*': AccessLevel.ADMIN,
            'data.read.*': AccessLevel.READ,
            'data.write.*': AccessLevel.WRITE,
            'public.*': AccessLevel.NONE
        }
    
    def create_user_profile(
        self,
        user_id: str,
        access_level: AccessLevel,
        allowed_resources: Set[str],
        rate_limits: Dict[str, int] = None
    ) -> SecurityProfile:
        """Create user security profile."""
        profile = SecurityProfile(
            user_id=user_id,
            access_level=access_level,
            allowed_resources=allowed_resources,
            rate_limits=rate_limits or {'requests_per_hour': 1000}
        )
        
        self.security_profiles[user_id] = profile
        logger.info(f"Created security profile for user {user_id}")
        
        return profile
    
    def generate_access_token(
        self,
        user_id: str,
        resource_patterns: Set[str],
        expires_in: int = 3600
    ) -> str:
        """Generate secure access token."""
        if user_id not in self.security_profiles:
            raise ValueError(f"User {user_id} not found")
        
        profile = self.security_profiles[user_id]
        current_time = time.time()
        
        # Validate resource access
        for pattern in resource_patterns:
            if not self._check_resource_access(profile, pattern):
                raise PermissionError(f"User {user_id} lacks access to {pattern}")
        
        # Generate token
        token_id = self.encryption.generate_secure_token()
        token = AccessToken(
            token_id=token_id,
            user_id=user_id,
            access_level=profile.access_level,
            resource_patterns=resource_patterns,
            expires_at=current_time + expires_in,
            created_at=current_time
        )
        
        self.active_tokens[token_id] = token
        
        # Create signed token
        token_data = {
            'token_id': token_id,
            'user_id': user_id,
            'expires_at': token.expires_at
        }
        
        signed_token = self.encryption.encrypt_dict(token_data)
        logger.info(f"Generated access token for user {user_id}")
        
        return signed_token
    
    def validate_token(self, signed_token: str) -> Optional[AccessToken]:
        """Validate and return access token."""
        try:
            token_data = self.encryption.decrypt_dict(signed_token)
            token_id = token_data['token_id']
            
            if token_id not in self.active_tokens:
                return None
            
            token = self.active_tokens[token_id]
            current_time = time.time()
            
            # Check expiration
            if current_time > token.expires_at:
                del self.active_tokens[token_id]
                return None
            
            # Update usage
            token.last_used = current_time
            token.use_count += 1
            
            return token
            
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None
    
    def authorize_access(
        self,
        token: AccessToken,
        resource: str,
        action: str = 'read'
    ) -> bool:
        """Authorize access to specific resource."""
        
        # Check token validity
        if time.time() > token.expires_at:
            return False
        
        # Check resource patterns
        resource_allowed = False
        for pattern in token.resource_patterns:
            if self._match_resource_pattern(pattern, resource):
                resource_allowed = True
                break
        
        if not resource_allowed:
            self._log_access_violation(token, resource, "Resource not in token scope")
            return False
        
        # Check access level requirements
        required_level = self._get_required_access_level(resource, action)
        if not self._access_level_sufficient(token.access_level, required_level):
            self._log_access_violation(token, resource, f"Insufficient access level: {token.access_level.value}")
            return False
        
        # Log successful access
        self.access_logs.append({
            'timestamp': time.time(),
            'user_id': token.user_id,
            'resource': resource,
            'action': action,
            'result': 'granted'
        })
        
        return True
    
    def _check_resource_access(self, profile: SecurityProfile, resource: str) -> bool:
        """Check if user profile allows access to resource."""
        for allowed in profile.allowed_resources:
            if self._match_resource_pattern(allowed, resource):
                return True
        return False
    
    def _match_resource_pattern(self, pattern: str, resource: str) -> bool:
        """Match resource against pattern (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(resource, pattern)
    
    def _get_required_access_level(self, resource: str, action: str) -> AccessLevel:
        """Get required access level for resource and action."""
        # Check specific policies
        for pattern, level in self.resource_policies.items():
            if self._match_resource_pattern(pattern, resource):
                return level
        
        # Default based on action
        if action in ['read', 'get']:
            return AccessLevel.READ
        elif action in ['write', 'post', 'put', 'delete']:
            return AccessLevel.WRITE
        else:
            return AccessLevel.ADMIN
    
    def _access_level_sufficient(self, user_level: AccessLevel, required_level: AccessLevel) -> bool:
        """Check if user access level is sufficient."""
        level_hierarchy = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SYSTEM: 4
        }
        
        return level_hierarchy[user_level] >= level_hierarchy[required_level]
    
    def _log_access_violation(self, token: AccessToken, resource: str, reason: str):
        """Log access violation."""
        self.access_logs.append({
            'timestamp': time.time(),
            'user_id': token.user_id,
            'resource': resource,
            'result': 'denied',
            'reason': reason
        })
        
        logger.warning(f"Access denied for user {token.user_id} to {resource}: {reason}")
    
    def revoke_token(self, token_id: str):
        """Revoke specific access token."""
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
            logger.info(f"Revoked token {token_id}")
    
    def revoke_user_tokens(self, user_id: str):
        """Revoke all tokens for a user."""
        tokens_to_remove = [
            token_id for token_id, token in self.active_tokens.items()
            if token.user_id == user_id
        ]
        
        for token_id in tokens_to_remove:
            del self.active_tokens[token_id]
        
        logger.info(f"Revoked {len(tokens_to_remove)} tokens for user {user_id}")
    
    def get_access_analytics(self) -> Dict[str, Any]:
        """Get access control analytics."""
        current_time = time.time()
        recent_logs = [log for log in self.access_logs if current_time - log['timestamp'] < 3600]
        
        granted = sum(1 for log in recent_logs if log['result'] == 'granted')
        denied = sum(1 for log in recent_logs if log['result'] == 'denied')
        
        return {
            'active_tokens': len(self.active_tokens),
            'registered_users': len(self.security_profiles),
            'recent_access_attempts': len(recent_logs),
            'recent_granted': granted,
            'recent_denied': denied,
            'access_success_rate': granted / max(1, granted + denied) * 100,
            'top_accessed_resources': self._get_top_resources(recent_logs)
        }
    
    def _get_top_resources(self, logs: List[Dict]) -> List[Tuple[str, int]]:
        """Get most accessed resources."""
        resource_counts = defaultdict(int)
        for log in logs:
            resource_counts[log['resource']] += 1
        
        return sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]


class SecurityFramework:
    """Main security framework coordinator."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize security framework."""
        self.encryption = AdvancedEncryption(master_key)
        self.threat_detector = ThreatDetectionEngine()
        self.access_control = AccessControlManager(self.encryption)
        
        self.security_events = deque(maxlen=10000)
        self.incident_responses = {}
        
        # Security monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Auto-response configuration
        self.auto_response_enabled = True
        self.response_thresholds = {
            ThreatLevel.CRITICAL: 0,  # Immediate response
            ThreatLevel.HIGH: 5,      # Response after 5 events
            ThreatLevel.MEDIUM: 10,   # Response after 10 events
            ThreatLevel.LOW: 50       # Response after 50 events
        }
        
        logger.info("Security framework initialized")
    
    def process_request(
        self,
        request_data: Dict[str, Any],
        access_token: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Process and validate request through security framework."""
        
        # Validate access token if provided
        token = None
        if access_token:
            token = self.access_control.validate_token(access_token)
            if not token:
                self._create_security_event(
                    SecurityEventType.AUTHENTICATION_FAILURE,
                    ThreatLevel.HIGH,
                    request_data.get('source_ip'),
                    request_data.get('user_id'),
                    request_data.get('resource', 'unknown'),
                    "Invalid or expired access token"
                )
                return False, ["Authentication failed"]
        
        # Authorize access if token provided
        if token:
            resource = request_data.get('resource', '')
            action = request_data.get('action', 'read')
            
            if not self.access_control.authorize_access(token, resource, action):
                self._create_security_event(
                    SecurityEventType.AUTHORIZATION_VIOLATION,
                    ThreatLevel.HIGH,
                    request_data.get('source_ip'),
                    token.user_id,
                    resource,
                    f"Unauthorized access attempt to {resource}"
                )
                return False, ["Authorization failed"]
        
        # Threat detection
        user_profile = None
        if token:
            user_profile = self.access_control.security_profiles.get(token.user_id)
        
        threat_level, threats = self.threat_detector.analyze_request(request_data, user_profile)
        
        # Create security events for detected threats
        if threats:
            self._create_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                threat_level,
                request_data.get('source_ip'),
                request_data.get('user_id'),
                request_data.get('resource', 'unknown'),
                f"Threats detected: {'; '.join(threats)}"
            )
        
        # Auto-response for high-level threats
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH] and self.auto_response_enabled:
            self._trigger_auto_response(threat_level, request_data, threats)
        
        # Block critical threats
        if threat_level == ThreatLevel.CRITICAL:
            return False, threats
        
        return True, threats if threats else []
    
    def _create_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        source_ip: Optional[str],
        user_id: Optional[str],
        resource: str,
        description: str
    ):
        """Create and log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            resource=resource,
            description=description
        )
        
        self.security_events.append(event)
        self.threat_detector.threat_history.append(event)
        
        # Update IP reputation
        if source_ip:
            threat_weight = {
                ThreatLevel.LOW: 1,
                ThreatLevel.MEDIUM: 3,
                ThreatLevel.HIGH: 7,
                ThreatLevel.CRITICAL: 15
            }[threat_level]
            
            self.threat_detector.update_ip_reputation(source_ip, event_type, threat_weight)
        
        logger.warning(f"Security event: {event_type.value} - {description}")
    
    def _trigger_auto_response(
        self,
        threat_level: ThreatLevel,
        request_data: Dict[str, Any],
        threats: List[str]
    ):
        """Trigger automated security response."""
        
        response_actions = []
        
        if threat_level == ThreatLevel.CRITICAL:
            # Critical threat responses
            source_ip = request_data.get('source_ip')
            user_id = request_data.get('user_id')
            
            if source_ip:
                # Block IP temporarily
                response_actions.append(f"IP {source_ip} temporarily blocked")
                # In a real system, this would integrate with firewall/WAF
            
            if user_id:
                # Revoke user tokens
                self.access_control.revoke_user_tokens(user_id)
                response_actions.append(f"All tokens revoked for user {user_id}")
            
        elif threat_level == ThreatLevel.HIGH:
            # High threat responses
            user_id = request_data.get('user_id')
            
            if user_id and user_id in self.access_control.security_profiles:
                # Increase monitoring for user
                profile = self.access_control.security_profiles[user_id]
                profile.failed_attempts += 1
                
                if profile.failed_attempts >= 5:
                    # Lock account temporarily
                    profile.locked_until = time.time() + 1800  # 30 minutes
                    response_actions.append(f"User {user_id} locked for 30 minutes")
        
        if response_actions:
            logger.info(f"Auto-response triggered: {'; '.join(response_actions)}")
    
    def start_monitoring(self):
        """Start security monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Background security monitoring loop."""
        while self._monitoring_active:
            try:
                # Cleanup expired tokens
                self._cleanup_expired_tokens()
                
                # Analyze security trends
                self._analyze_security_trends()
                
                # Generate alerts for patterns
                self._check_alert_conditions()
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
                time.sleep(60)
    
    def _cleanup_expired_tokens(self):
        """Clean up expired access tokens."""
        current_time = time.time()
        expired_tokens = [
            token_id for token_id, token in self.access_control.active_tokens.items()
            if token.expires_at < current_time
        ]
        
        for token_id in expired_tokens:
            del self.access_control.active_tokens[token_id]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
    
    def _analyze_security_trends(self):
        """Analyze security event trends."""
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type] += 1
        
        # Check for unusual patterns
        for event_type, count in event_counts.items():
            if count > 50:  # More than 50 events of same type in hour
                logger.warning(f"High frequency of {event_type.value}: {count} events in last hour")
    
    def _check_alert_conditions(self):
        """Check conditions that should trigger alerts."""
        current_time = time.time()
        
        # Check for failed authentication spike
        recent_auth_failures = [
            event for event in self.security_events
            if (event.event_type == SecurityEventType.AUTHENTICATION_FAILURE and
                current_time - event.timestamp < 300)  # Last 5 minutes
        ]
        
        if len(recent_auth_failures) > 20:
            logger.critical(f"Authentication failure spike: {len(recent_auth_failures)} failures in 5 minutes")
        
        # Check for critical threats
        recent_critical = [
            event for event in self.security_events
            if (event.threat_level == ThreatLevel.CRITICAL and
                current_time - event.timestamp < 300)
        ]
        
        if len(recent_critical) > 5:
            logger.critical(f"Multiple critical threats: {len(recent_critical)} in 5 minutes")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        current_time = time.time()
        
        # Recent events analysis
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 3600
        ]
        
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
        
        # Access control analytics
        access_analytics = self.access_control.get_access_analytics()
        
        # Threat detection analytics
        threat_analytics = self.threat_detector.get_threat_analytics()
        
        return {
            'system_status': {
                'monitoring_active': self._monitoring_active,
                'auto_response_enabled': self.auto_response_enabled
            },
            'recent_events': {
                'total': len(recent_events),
                'by_threat_level': dict(threat_counts),
                'unresolved': sum(1 for e in recent_events if not e.resolved)
            },
            'access_control': access_analytics,
            'threat_detection': threat_analytics,
            'top_security_events': self._get_top_security_events(),
            'security_score': self._calculate_security_score()
        }
    
    def _get_top_security_events(self) -> List[Dict[str, Any]]:
        """Get top recent security events."""
        recent_events = list(self.security_events)[-20:]  # Last 20 events
        
        return [
            {
                'timestamp': event.timestamp,
                'type': event.event_type.value,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'description': event.description
            }
            for event in reversed(recent_events)
        ]
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score."""
        current_time = time.time()
        
        # Base score
        score = 100.0
        
        # Deduct for recent critical events
        recent_critical = [
            event for event in self.security_events
            if (event.threat_level == ThreatLevel.CRITICAL and
                current_time - event.timestamp < 86400)  # Last 24 hours
        ]
        score -= len(recent_critical) * 10
        
        # Deduct for high threat events
        recent_high = [
            event for event in self.security_events
            if (event.threat_level == ThreatLevel.HIGH and
                current_time - event.timestamp < 86400)
        ]
        score -= len(recent_high) * 5
        
        # Factor in access success rate
        access_analytics = self.access_control.get_access_analytics()
        access_success_rate = access_analytics.get('access_success_rate', 100)
        score = score * (access_success_rate / 100)
        
        return max(0.0, min(100.0, score))


# Global security framework instance
_global_security: Optional[SecurityFramework] = None


def get_security_framework() -> SecurityFramework:
    """Get global security framework."""
    global _global_security
    if _global_security is None:
        _global_security = SecurityFramework()
    return _global_security


def initialize_security_framework(master_key: Optional[bytes] = None) -> SecurityFramework:
    """Initialize and start security framework."""
    global _global_security
    
    _global_security = SecurityFramework(master_key)
    _global_security.start_monitoring()
    
    logger.info("Advanced security framework initialized")
    return _global_security


if __name__ == "__main__":
    async def demo_security_framework():
        # Initialize security framework
        security = initialize_security_framework()
        
        # Create test user
        security.access_control.create_user_profile(
            user_id="test_user",
            access_level=AccessLevel.READ,
            allowed_resources={"data.read.*", "public.*"}
        )
        
        # Generate access token
        token = security.access_control.generate_access_token(
            user_id="test_user",
            resource_patterns={"data.read.*"}
        )
        
        # Test legitimate request
        request_data = {
            'source_ip': '192.168.1.100',
            'user_id': 'test_user',
            'resource': 'data.read.customers',
            'action': 'read',
            'data': {'query': 'SELECT * FROM customers LIMIT 10'}
        }
        
        allowed, messages = security.process_request(request_data, token)
        print(f"Legitimate request: {'Allowed' if allowed else 'Blocked'}")
        if messages:
            print(f"Messages: {messages}")
        
        # Test malicious request
        malicious_request = {
            'source_ip': '10.0.0.1',
            'user_id': 'attacker',
            'resource': 'admin.config',
            'action': 'read',
            'data': {'query': "SELECT * FROM users WHERE '1'='1' OR DROP TABLE users;"}
        }
        
        allowed, messages = security.process_request(malicious_request)
        print(f"Malicious request: {'Allowed' if allowed else 'Blocked'}")
        if messages:
            print(f"Messages: {messages}")
        
        # Wait for monitoring
        await asyncio.sleep(2)
        
        # Get security dashboard
        dashboard = security.get_security_dashboard()
        print("\nSecurity Dashboard:")
        print(json.dumps(dashboard, indent=2, default=str))
        
        security.stop_monitoring()
    
    # Run demo
    asyncio.run(demo_security_framework())