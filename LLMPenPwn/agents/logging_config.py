#!/usr/bin/env python3
"""
Advanced logging configuration for penetration testing agents.
Provides structured logging with file rotation, multiple handlers, and detailed formatting.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "agent_name"):
            log_entry["agent"] = record.agent_name
        
        if hasattr(record, "target_ip"):
            log_entry["target_ip"] = record.target_ip
        
        if hasattr(record, "tool_name"):
            log_entry["tool"] = record.tool_name
        
        if hasattr(record, "task_id"):
            log_entry["task_id"] = record.task_id
        
        return json.dumps(log_entry)


class AgentFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        if hasattr(record, "agent_name"):
            record.name = f"{record.name}[{record.agent_name}]"
        if hasattr(record, "target_ip"):
            record.msg = f"[{record.target_ip}] {record.msg}"
        return super().format(record)


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    enable_json_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure advanced logging for the penetration testing system.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Enable file-based logging
        enable_json_logging: Enable JSON structured logging
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup log files to keep
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(AgentFormatter())
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        main_log_file = log_path / "pentest_main.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(AgentFormatter())
        root_logger.addHandler(file_handler)
    
    if enable_json_logging:
        json_log_file = log_path / "pentest_structured.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(level)
        json_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(json_handler)
    
    agent_log_file = log_path / "agents.log"
    agent_handler = logging.handlers.RotatingFileHandler(
        agent_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    agent_handler.setLevel(logging.DEBUG)
    agent_handler.setFormatter(AgentFormatter())
    agent_logger = logging.getLogger("agents")
    agent_logger.addHandler(agent_handler)
    agent_logger.setLevel(logging.DEBUG)
    
    tools_log_file = log_path / "tools.log"
    tools_handler = logging.handlers.RotatingFileHandler(
        tools_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    tools_handler.setLevel(logging.DEBUG)
    tools_handler.setFormatter(AgentFormatter())
    tools_logger = logging.getLogger("tools")
    tools_logger.addHandler(tools_handler)
    tools_logger.setLevel(logging.DEBUG)
    
    db_log_file = log_path / "database.log"
    db_handler = logging.handlers.RotatingFileHandler(
        db_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    db_handler.setLevel(logging.DEBUG)
    db_handler.setFormatter(AgentFormatter())
    db_logger = logging.getLogger("database")
    db_logger.addHandler(db_handler)
    db_logger.setLevel(logging.DEBUG)


def get_logger(name: str, agent_name: str = None) -> logging.Logger:
    """
    Get a logger with optional agent name context.
    
    Args:
        name: Logger name (usually __name__)
        agent_name: Optional agent identifier for context
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if agent_name:
        logger = logging.LoggerAdapter(logger, {"agent_name": agent_name})
    return logger


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds agent context to log records."""
    
    def __init__(self, logger: logging.Logger, agent_name: str, target_ip: str = None):
        super().__init__(logger, {
            "agent_name": agent_name,
            "target_ip": target_ip,
        })
    
    def process(self, msg, kwargs):
        return msg, kwargs


class ToolLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds tool context to log records."""
    
    def __init__(self, logger: logging.Logger, tool_name: str):
        super().__init__(logger, {"tool_name": tool_name})
    
    def process(self, msg, kwargs):
        return msg, kwargs
