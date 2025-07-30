# Security Policy

## Supported Versions

We actively support the following versions of LLM Tab Cleaner:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please report it responsibly:

### Private Disclosure

**Do not create a public GitHub issue for security vulnerabilities.**

Instead, please:

1. **Email**: Send details to [daniel@terragonlabs.com](mailto:daniel@terragonlabs.com)
2. **Subject Line**: Use "SECURITY: [Brief Description]"
3. **Include**:
   - Detailed description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days for critical issues

### Security Considerations

This project handles potentially sensitive data through LLM processing. Key security areas include:

#### Data Privacy
- Sensitive data sent to external LLM APIs
- Audit trails containing potentially sensitive information
- Temporary data storage during processing

#### API Security  
- LLM API key management and rotation
- Rate limiting and usage monitoring
- Input validation and sanitization

#### Processing Security
- Malicious data injection through cleaning prompts
- Resource exhaustion attacks
- Unauthorized data access

### Security Best Practices

When using LLM Tab Cleaner:

1. **API Keys**: Store LLM API keys securely using environment variables
2. **Data Sensitivity**: Review data before sending to external LLM services
3. **Access Control**: Implement proper access controls for audit logs
4. **Monitoring**: Monitor LLM API usage for anomalies
5. **Updates**: Keep dependencies updated for security patches

### Disclosure Policy

After a security issue is resolved:

1. We will credit the reporter (unless they prefer to remain anonymous)
2. We will publish a security advisory with details and mitigation
3. We will release a patched version with security fixes
4. We will notify users through our release channels

### Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [LLM Security Guidelines](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Python Security Best Practices](https://python.org/dev/security/)

## Contact

For security-related questions or concerns:
- Email: [daniel@terragonlabs.com](mailto:daniel@terragonlabs.com)
- Matrix: Mention responsible disclosure in subject line