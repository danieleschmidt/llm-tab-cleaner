"""Internationalization and localization support."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class I18nManager:
    """Manages internationalization and localization."""
    
    def __init__(self, locale: str = "en", translations_dir: Optional[str] = None):
        """Initialize i18n manager.
        
        Args:
            locale: Target locale (e.g., 'en', 'es', 'fr', 'de', 'ja', 'zh')
            translations_dir: Directory containing translation files
        """
        self.locale = locale
        self.translations_dir = Path(translations_dir) if translations_dir else self._get_default_translations_dir()
        self.translations: Dict[str, str] = {}
        self.fallback_locale = "en"
        
        self._load_translations()
    
    def _get_default_translations_dir(self) -> Path:
        """Get default translations directory."""
        return Path(__file__).parent / "translations"
    
    def _load_translations(self) -> None:
        """Load translations for current locale."""
        try:
            # Try to load primary locale
            translation_file = self.translations_dir / f"{self.locale}.json"
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
                logger.info(f"Loaded translations for locale: {self.locale}")
            else:
                logger.warning(f"Translation file not found: {translation_file}")
                
                # Fall back to English
                if self.locale != self.fallback_locale:
                    fallback_file = self.translations_dir / f"{self.fallback_locale}.json"
                    if fallback_file.exists():
                        with open(fallback_file, 'r', encoding='utf-8') as f:
                            self.translations = json.load(f)
                        logger.info(f"Loaded fallback translations: {self.fallback_locale}")
                    else:
                        # Use built-in English translations
                        self.translations = self._get_builtin_translations()
                        logger.info("Using built-in English translations")
        except Exception as e:
            logger.error(f"Error loading translations: {e}")
            self.translations = self._get_builtin_translations()
    
    def _get_builtin_translations(self) -> Dict[str, str]:
        """Get built-in English translations as fallback."""
        return {
            # Core messages
            "cleaning.started": "Started data cleaning process",
            "cleaning.completed": "Data cleaning completed",
            "cleaning.failed": "Data cleaning failed",
            "cleaning.fixes_applied": "{count} fixes applied",
            "cleaning.quality_score": "Quality score: {score:.1%}",
            "cleaning.processing_time": "Processing time: {time:.2f}s",
            
            # Column types
            "column.type.text": "Text",
            "column.type.numeric": "Numeric", 
            "column.type.date": "Date",
            "column.type.email": "Email",
            "column.type.phone": "Phone",
            "column.type.category": "Category",
            "column.type.unknown": "Unknown",
            
            # Data quality issues
            "issue.missing_value": "Missing value",
            "issue.invalid_format": "Invalid format",
            "issue.outlier": "Outlier detected",
            "issue.duplicate": "Duplicate record",
            "issue.inconsistent": "Inconsistent data",
            
            # Fix descriptions
            "fix.standardized": "Standardized format",
            "fix.corrected_typo": "Corrected spelling",
            "fix.filled_missing": "Filled missing value",
            "fix.removed_duplicate": "Removed duplicate",
            "fix.normalized": "Normalized data",
            
            # Security messages
            "security.sensitive_data_detected": "Sensitive data detected",
            "security.data_size_exceeded": "Data size limit exceeded",
            "security.validation_failed": "Security validation failed",
            "security.access_denied": "Access denied",
            
            # Performance messages
            "performance.slow_operation": "Operation is running slowly",
            "performance.high_memory_usage": "High memory usage detected",
            "performance.optimization_applied": "Performance optimization applied",
            
            # Error messages
            "error.connection_failed": "Connection failed",
            "error.invalid_data": "Invalid data format",
            "error.processing_error": "Processing error occurred",
            "error.configuration_error": "Configuration error",
            
            # Success messages
            "success.data_cleaned": "Data successfully cleaned",
            "success.report_generated": "Report generated successfully",
            "success.export_completed": "Export completed",
            
            # General terms
            "general.yes": "Yes",
            "general.no": "No",
            "general.cancel": "Cancel",
            "general.continue": "Continue",
            "general.retry": "Retry",
            "general.skip": "Skip"
        }
    
    def t(self, key: str, **kwargs) -> str:
        """Translate a message key.
        
        Args:
            key: Translation key
            **kwargs: Variables for string formatting
            
        Returns:
            Translated and formatted message
        """
        if key in self.translations:
            message = self.translations[key]
        else:
            # Return key as fallback
            logger.warning(f"Translation key not found: {key}")
            message = key
        
        # Format with provided variables
        try:
            if kwargs:
                return message.format(**kwargs)
            return message
        except KeyError as e:
            logger.warning(f"Missing formatting variable {e} for key: {key}")
            return message
        except Exception as e:
            logger.error(f"Error formatting translation: {e}")
            return message
    
    def get_locale(self) -> str:
        """Get current locale."""
        return self.locale
    
    def set_locale(self, locale: str) -> None:
        """Set new locale and reload translations."""
        self.locale = locale
        self._load_translations()
        logger.info(f"Locale changed to: {locale}")
    
    def get_available_locales(self) -> list[str]:
        """Get list of available locales."""
        if not self.translations_dir.exists():
            return [self.fallback_locale]
        
        locales = []
        for file_path in self.translations_dir.glob("*.json"):
            locale = file_path.stem
            locales.append(locale)
        
        return sorted(locales)


# Global i18n manager instance
_global_i18n: Optional[I18nManager] = None


def get_i18n() -> I18nManager:
    """Get global i18n manager instance."""
    global _global_i18n
    if _global_i18n is None:
        _global_i18n = I18nManager()
    return _global_i18n


def set_locale(locale: str) -> None:
    """Set global locale."""
    i18n = get_i18n()
    i18n.set_locale(locale)


def t(key: str, **kwargs) -> str:
    """Translate message using global i18n manager."""
    return get_i18n().t(key, **kwargs)


def create_translation_files():
    """Create translation files for supported locales."""
    translations_dir = Path(__file__).parent / "translations"
    translations_dir.mkdir(exist_ok=True)
    
    # Base English translations
    base_translations = I18nManager()._get_builtin_translations()
    
    # Create translations for different languages
    locale_translations = {
        "en": base_translations,  # English (base)
        
        "es": {  # Spanish
            "cleaning.started": "Proceso de limpieza de datos iniciado",
            "cleaning.completed": "Limpieza de datos completada", 
            "cleaning.failed": "Falló la limpieza de datos",
            "cleaning.fixes_applied": "{count} correcciones aplicadas",
            "cleaning.quality_score": "Puntuación de calidad: {score:.1%}",
            "cleaning.processing_time": "Tiempo de procesamiento: {time:.2f}s",
            
            "column.type.text": "Texto",
            "column.type.numeric": "Numérico",
            "column.type.date": "Fecha",
            "column.type.email": "Email",
            "column.type.phone": "Teléfono",
            "column.type.category": "Categoría",
            "column.type.unknown": "Desconocido",
            
            "issue.missing_value": "Valor faltante",
            "issue.invalid_format": "Formato inválido",
            "issue.outlier": "Valor atípico detectado",
            "issue.duplicate": "Registro duplicado",
            "issue.inconsistent": "Datos inconsistentes",
            
            "security.sensitive_data_detected": "Datos sensibles detectados",
            "security.data_size_exceeded": "Límite de tamaño de datos excedido",
            "security.validation_failed": "Falló la validación de seguridad",
            
            "general.yes": "Sí",
            "general.no": "No",
            "general.cancel": "Cancelar",
            "general.continue": "Continuar",
            "general.retry": "Reintentar",
        },
        
        "fr": {  # French
            "cleaning.started": "Processus de nettoyage des données démarré",
            "cleaning.completed": "Nettoyage des données terminé",
            "cleaning.failed": "Échec du nettoyage des données",
            "cleaning.fixes_applied": "{count} corrections appliquées",
            "cleaning.quality_score": "Score de qualité: {score:.1%}",
            "cleaning.processing_time": "Temps de traitement: {time:.2f}s",
            
            "column.type.text": "Texte",
            "column.type.numeric": "Numérique",
            "column.type.date": "Date",
            "column.type.email": "Email",
            "column.type.phone": "Téléphone",
            "column.type.category": "Catégorie",
            "column.type.unknown": "Inconnu",
            
            "security.sensitive_data_detected": "Données sensibles détectées",
            "security.validation_failed": "Échec de la validation de sécurité",
            
            "general.yes": "Oui",
            "general.no": "Non",
            "general.cancel": "Annuler",
            "general.continue": "Continuer",
            "general.retry": "Réessayer",
        },
        
        "de": {  # German
            "cleaning.started": "Datenbereinigungsprozess gestartet",
            "cleaning.completed": "Datenbereinigung abgeschlossen", 
            "cleaning.failed": "Datenbereinigung fehlgeschlagen",
            "cleaning.fixes_applied": "{count} Korrekturen angewendet",
            "cleaning.quality_score": "Qualitätswert: {score:.1%}",
            "cleaning.processing_time": "Verarbeitungszeit: {time:.2f}s",
            
            "column.type.text": "Text",
            "column.type.numeric": "Numerisch",
            "column.type.date": "Datum",
            "column.type.email": "E-Mail",
            "column.type.phone": "Telefon",
            "column.type.category": "Kategorie",
            "column.type.unknown": "Unbekannt",
            
            "security.sensitive_data_detected": "Sensible Daten erkannt",
            "security.validation_failed": "Sicherheitsvalidierung fehlgeschlagen",
            
            "general.yes": "Ja",
            "general.no": "Nein", 
            "general.cancel": "Abbrechen",
            "general.continue": "Fortfahren",
            "general.retry": "Wiederholen",
        },
        
        "ja": {  # Japanese
            "cleaning.started": "データクリーニングプロセスを開始しました",
            "cleaning.completed": "データクリーニングが完了しました",
            "cleaning.failed": "データクリーニングに失敗しました",
            "cleaning.fixes_applied": "{count}件の修正を適用しました",
            "cleaning.quality_score": "品質スコア: {score:.1%}",
            "cleaning.processing_time": "処理時間: {time:.2f}秒",
            
            "column.type.text": "テキスト",
            "column.type.numeric": "数値",
            "column.type.date": "日付",
            "column.type.email": "メール",
            "column.type.phone": "電話番号",
            "column.type.category": "カテゴリ",
            "column.type.unknown": "不明",
            
            "security.sensitive_data_detected": "機密データが検出されました",
            "security.validation_failed": "セキュリティ検証に失敗しました",
            
            "general.yes": "はい",
            "general.no": "いいえ",
            "general.cancel": "キャンセル",
            "general.continue": "続行",
            "general.retry": "再試行",
        },
        
        "zh": {  # Chinese (Simplified)
            "cleaning.started": "数据清理过程已开始",
            "cleaning.completed": "数据清理已完成",
            "cleaning.failed": "数据清理失败",
            "cleaning.fixes_applied": "已应用{count}项修复",
            "cleaning.quality_score": "质量分数：{score:.1%}",
            "cleaning.processing_time": "处理时间：{time:.2f}秒",
            
            "column.type.text": "文本",
            "column.type.numeric": "数值",
            "column.type.date": "日期",
            "column.type.email": "电子邮件",
            "column.type.phone": "电话",
            "column.type.category": "类别",
            "column.type.unknown": "未知",
            
            "security.sensitive_data_detected": "检测到敏感数据",
            "security.validation_failed": "安全验证失败",
            
            "general.yes": "是",
            "general.no": "否",
            "general.cancel": "取消",
            "general.continue": "继续",
            "general.retry": "重试",
        }
    }
    
    # Write translation files
    for locale, translations in locale_translations.items():
        file_path = translations_dir / f"{locale}.json"
        
        # Merge with base translations for incomplete translations
        if locale != "en":
            complete_translations = base_translations.copy()
            complete_translations.update(translations)
            translations = complete_translations
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created translation file: {file_path}")


def setup_i18n(locale: str = "en", translations_dir: Optional[str] = None) -> I18nManager:
    """Setup internationalization system.
    
    Args:
        locale: Target locale
        translations_dir: Optional custom translations directory
        
    Returns:
        Configured I18nManager instance
    """
    global _global_i18n
    
    # Create translation files if they don't exist
    default_dir = Path(__file__).parent / "translations"
    if not default_dir.exists() or not any(default_dir.glob("*.json")):
        create_translation_files()
    
    _global_i18n = I18nManager(locale=locale, translations_dir=translations_dir)
    return _global_i18n