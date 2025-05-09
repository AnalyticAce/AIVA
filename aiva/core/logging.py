import logging
import re
import sys
import time
from typing import Optional

from aiva.core.config import settings

logger = logging.getLogger("aiva")


class PIIFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.timestamp = time.time()
        
        if isinstance(record.msg, str):
            if "insert into" in record.msg.lower():
                record.msg = self._redact_sql(record.msg)
        
        return True
    
    @staticmethod
    def _redact_api_keys(message: str) -> str:
        api_key_pattern = r'([\'"][a-zA-Z0-9_]{30,}[\'"])'
        return re.sub(api_key_pattern, "'[REDACTED]'", message)
    
    @staticmethod
    def _redact_sql(message: str) -> str:
        if "insert into" in message.lower():
            sql_pattern = r'INSERT INTO.*VALUES.*'
            return re.sub(sql_pattern, "INSERT INTO ... VALUES [REDACTED]", message, flags=re.IGNORECASE)
        return message


class CustomFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[41m',
        'RESET': '\033[0m',
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
        msecs = int(record.msecs)
        
        log_fmt = f"{log_color}{timestamp}.{msecs:03d}{reset} | "
        log_fmt += f"{log_color}{record.levelname:<8}{reset} | "
        log_fmt += f"\033[36m{record.name}:{record.funcName}:{record.lineno}\033[0m | "
        log_fmt += f"{log_color}{record.getMessage()}{reset}"
        
        if record.exc_info:
            exc_text = logging.Formatter().formatException(record.exc_info)
            log_fmt += f"\n{exc_text}"
        
        return log_fmt


def setup_logging(log_level: Optional[str] = None) -> None:
    log_level = log_level or settings.log_level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logger.setLevel(numeric_level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(CustomFormatter())
    
    console_handler.addFilter(PIIFilter())
    
    logger.addHandler(console_handler)
    
    if settings.enable_sentry and settings.sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
            )
            logger.info("Sentry integration enabled")
        except ImportError:
            logger.warning("Sentry SDK not installed but Sentry is enabled in config")
    
    logger.info(f"Logging configured at level {log_level}")


def configure_logging(log_level: Optional[str] = None) -> None:
    setup_logging(log_level)


__all__ = ["logger", "setup_logging", "configure_logging"]