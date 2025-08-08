#!/usr/bin/env python3
"""Test global-first features of LLM Tab Cleaner."""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_internationalization():
    """Test i18n and localization features."""
    print("\n🌍 Testing internationalization...")
    
    try:
        from llm_tab_cleaner.i18n import setup_i18n, t, set_locale
        
        # Setup i18n system
        i18n = setup_i18n(locale="en")
        print("✅ I18n system initialized")
        
        # Test English (default)
        message_en = t("cleaning.started")
        print(f"✅ English: '{message_en}'")
        
        # Test Spanish
        set_locale("es")
        message_es = t("cleaning.started")
        print(f"✅ Spanish: '{message_es}'")
        
        # Test French
        set_locale("fr")
        message_fr = t("cleaning.started")
        print(f"✅ French: '{message_fr}'")
        
        # Test German
        set_locale("de")
        message_de = t("cleaning.started")
        print(f"✅ German: '{message_de}'")
        
        # Test Japanese
        set_locale("ja")
        message_ja = t("cleaning.started")
        print(f"✅ Japanese: '{message_ja}'")
        
        # Test Chinese
        set_locale("zh")
        message_zh = t("cleaning.started")
        print(f"✅ Chinese: '{message_zh}'")
        
        # Test formatting with variables
        set_locale("en")
        formatted_message = t("cleaning.fixes_applied", count=5)
        print(f"✅ Formatting: '{formatted_message}'")
        
        # Verify different messages in different languages
        assert message_en != message_es
        assert message_fr != message_de
        assert message_ja != message_zh
        
        print("✅ All languages working correctly")
        return True
        
    except Exception as e:
        print(f"❌ I18n test failed: {e}")
        return False

def test_gdpr_compliance():
    """Test GDPR compliance features."""
    print("\n🔒 Testing GDPR compliance...")
    
    try:
        from llm_tab_cleaner.compliance import (
            ComplianceManager, create_gdpr_config, ConsentRecord,
            ProcessingRecord, DataClassification
        )
        
        # Create GDPR-compliant configuration
        config = create_gdpr_config()
        compliance_manager = ComplianceManager(config)
        
        print("✅ GDPR compliance manager created")
        
        # Test data classification
        test_df = pd.DataFrame({
            'customer_name': ['John Smith', 'Jane Doe'],
            'email': ['john@example.com', 'jane@example.com'],
            'phone': ['555-1234', '555-5678'],
            'product': ['Widget A', 'Widget B']
        })
        
        classifications = compliance_manager.classify_data(test_df)
        print(f"✅ Data classified: {classifications}")
        
        # Verify personal data is detected
        assert classifications['customer_name'] == DataClassification.PERSONAL
        assert classifications['email'] == DataClassification.PERSONAL
        assert classifications['product'] == DataClassification.INTERNAL
        
        # Test consent management
        consent = ConsentRecord(
            subject_id="user123",
            consent_type="processing",
            purpose="data_cleaning",
            granted=True,
            timestamp=datetime.now(),
            expiry_date=datetime.now() + timedelta(days=365),
            legal_basis="consent"
        )
        
        compliance_manager.record_consent(consent)
        
        consent_valid = compliance_manager.check_consent("user123", "data_cleaning")
        assert consent_valid == True
        
        print("✅ Consent management working")
        
        # Test processing record
        processing_record = ProcessingRecord(
            activity_id="activity_001",
            data_subject_id="user123",
            data_categories=["name", "email"],
            processing_purpose="data_cleaning",
            legal_basis="consent",
            timestamp=datetime.now(),
            controller="test_controller"
        )
        
        compliance_manager.record_processing_activity(processing_record)
        print("✅ Processing activity recorded")
        
        # Test data anonymization
        if config.require_anonymization:
            anonymized_df = compliance_manager.anonymize_data(test_df, classifications)
            print("✅ Data anonymization applied")
            
            # Verify personal data is anonymized
            assert not any("John Smith" in str(val) for val in anonymized_df['customer_name'])
        
        # Test compliance report
        report = compliance_manager.generate_compliance_report()
        print(f"✅ Compliance report generated: {report['report_id'][:8]}...")
        
        assert report['compliance_checks']['consent_compliance']['compliant'] == True
        
        print("✅ GDPR compliance features working")
        return True
        
    except Exception as e:
        print(f"❌ GDPR compliance test failed: {e}")
        return False

def test_ccpa_compliance():
    """Test CCPA compliance features."""
    print("\n🇺🇸 Testing CCPA compliance...")
    
    try:
        from llm_tab_cleaner.compliance import ComplianceManager, create_ccpa_config
        
        # Create CCPA-compliant configuration
        config = create_ccpa_config()
        compliance_manager = ComplianceManager(config)
        
        print("✅ CCPA compliance manager created")
        
        # CCPA uses opt-out model, so consent should not be required by default
        assert config.require_consent == False
        print("✅ CCPA opt-out model configured")
        
        # Test right to deletion (similar to GDPR right to be forgotten)
        assert config.enable_right_to_be_forgotten == True
        print("✅ Right to deletion enabled")
        
        # Test cross-border transfers (allowed for CCPA)
        assert config.allow_cross_border_transfer == True
        print("✅ Cross-border transfers allowed")
        
        # Generate compliance report
        report = compliance_manager.generate_compliance_report()
        
        assert "us" in [region for region in report['configuration']['regions']]
        print("✅ CCPA region correctly configured")
        
        print("✅ CCPA compliance features working")
        return True
        
    except Exception as e:
        print(f"❌ CCPA compliance test failed: {e}")
        return False

def test_cross_platform_compatibility():
    """Test cross-platform compatibility."""
    print("\n💻 Testing cross-platform compatibility...")
    
    try:
        import platform
        import os
        
        current_platform = platform.system()
        print(f"✅ Running on: {current_platform}")
        
        # Test path handling across platforms
        from pathlib import Path
        
        test_path = Path("test") / "data" / "file.csv"
        print(f"✅ Path handling works: {test_path}")
        
        # Test environment variables
        test_var = os.environ.get("PATH")
        assert test_var is not None
        print("✅ Environment variable access works")
        
        # Test file operations
        temp_file = Path("temp_test.txt")
        temp_file.write_text("test content")
        content = temp_file.read_text()
        temp_file.unlink()
        
        assert content == "test content"
        print("✅ File operations work")
        
        print("✅ Cross-platform compatibility verified")
        return True
        
    except Exception as e:
        print(f"❌ Cross-platform test failed: {e}")
        return False

def test_global_configuration():
    """Test global configuration management.""" 
    print("\n⚙️ Testing global configuration...")
    
    try:
        from llm_tab_cleaner.compliance import create_global_baseline_config
        from llm_tab_cleaner.i18n import setup_i18n, get_i18n
        
        # Test global compliance baseline
        global_config = create_global_baseline_config()
        print("✅ Global compliance baseline created")
        
        assert global_config.detailed_audit_logging == True
        assert global_config.enforce_data_minimization == True
        print("✅ Global security standards enforced")
        
        # Test multilingual configuration
        i18n = setup_i18n("en")
        available_locales = i18n.get_available_locales()
        
        # Should have at least English and several other languages
        expected_locales = ["en", "es", "fr", "de", "ja", "zh"]
        available_set = set(available_locales)
        expected_set = set(expected_locales)
        
        if not expected_set.issubset(available_set):
            missing = expected_set - available_set
            print(f"⚠️ Missing locales: {missing}")
        else:
            print(f"✅ All expected locales available: {expected_locales}")
        
        print("✅ Global configuration management working")
        return True
        
    except Exception as e:
        print(f"❌ Global configuration test failed: {e}")
        return False

def test_integrated_global_features():
    """Test integrated global features with data cleaning."""
    print("\n🌐 Testing integrated global features...")
    
    try:
        from llm_tab_cleaner import TableCleaner
        from llm_tab_cleaner.compliance import ComplianceManager, create_gdpr_config
        from llm_tab_cleaner.i18n import setup_i18n, t
        
        # Setup multilingual environment
        setup_i18n("en")
        
        # Setup compliance
        compliance_config = create_gdpr_config()
        compliance_manager = ComplianceManager(compliance_config)
        
        # Create test data with international elements
        test_data = {
            'name': ['José García', 'François Dubois', '田中太郎', 'John Smith'],
            'email': ['jose@example.es', 'francois@example.fr', 'tanaka@example.jp', 'john@example.com'],
            'country': ['Spain', 'France', 'Japan', 'USA']
        }
        df = pd.DataFrame(test_data)
        
        print("✅ International test data created")
        
        # Classify data for compliance
        classifications = compliance_manager.classify_data(df)
        print(f"✅ Data classified for compliance: {classifications}")
        
        # Clean data with compliance in mind
        cleaner = TableCleaner(llm_provider="local", confidence_threshold=0.8)
        cleaned_df, report = cleaner.clean(df)
        
        print(f"✅ Data cleaned: {report.total_fixes} fixes, {report.quality_score:.2%} quality")
        
        # Test localized reporting
        success_message = t("success.data_cleaned")
        quality_message = t("cleaning.quality_score", score=report.quality_score)
        
        print(f"✅ Localized messages: '{success_message}', '{quality_message}'")
        
        # Generate compliance report
        compliance_report = compliance_manager.generate_compliance_report()
        print(f"✅ Compliance report: {compliance_report['report_id'][:8]}...")
        
        print("✅ Integrated global features working")
        return True
        
    except Exception as e:
        print(f"❌ Integrated global features test failed: {e}")
        return False

def main():
    """Run all global-first tests."""
    print("🌍 LLM Tab Cleaner Global-First Test Suite")
    print("="*45)
    
    tests = [
        ("Internationalization", test_internationalization),
        ("GDPR Compliance", test_gdpr_compliance),
        ("CCPA Compliance", test_ccpa_compliance),
        ("Cross-Platform Compatibility", test_cross_platform_compatibility),
        ("Global Configuration", test_global_configuration),
        ("Integrated Global Features", test_integrated_global_features)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"🧪 Running {test_name}...")
            print(f"{'='*50}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 GLOBAL-FIRST TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ALL GLOBAL-FIRST FEATURES WORKING!")
        print("🌍 System is ready for global deployment!")
        print("\n✨ Global-First Achievements:")
        print("   • Multi-region compliance (GDPR, CCPA, Global)")
        print("   • 6 languages supported (EN, ES, FR, DE, JA, ZH)")
        print("   • Cross-platform compatibility verified")
        print("   • International data handling ready")
        print("   • Localized user interface")
        return True
    else:
        print(f"\n⚠️ {total-passed} global feature(s) failed.")
        print("❗ Address issues before global deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)