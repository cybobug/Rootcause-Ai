import random
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger("RootCauseAI")

class IncidentSimulator:
    """Generates realistic failure scenarios for demonstration purposes."""
    
    def __init__(self, analyzer_instance):
        self.analyzer = analyzer_instance
        self.scenarios = {
            "database_deadlock": self._gen_db_deadlock,
            "api_cascade": self._gen_api_cascade,
            "memory_leak": self._gen_memory_leak,
            "config_error": self._gen_config_error
        }
        self.log_connector = None
        # Try to get the log connector if it's registered
        try:
            self.log_connector = self.analyzer.registry.get("logs")
        except:
            self.log_connector = None

    def run(self, scenario_name: str = "database_deadlock") -> int:
        """Run a predefined failure scenario and return number of events generated."""
        scenario = self.scenarios.get(scenario_name)
        if scenario:
            logger.info(f"Starting simulation: {scenario_name}")
            return scenario()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

    def _add_event(self, event_data: Dict[str, Any]) -> str:
        """Helper to add an event to the analyzer's store and return its ID."""
        self.analyzer.store.add_event(event_data)
        return event_data.get("event_id", "unknown")

    def _gen_db_deadlock(self) -> int:
        """Simulate a database deadlock caused by a recent code change."""
        base_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        events_added = 0
        
        # 1. Code change that introduces the issue
        commit_time = base_time
        commit_id = self._add_event({
            "event_id": f"sim_commit_{int(time.time())}",
            "source_type": "code",
            "timestamp": commit_time,
            "content": {
                "commit_id": "a1b2c3d4",
                "author": "Dev Smith",
                "message": "feat: add concurrent checkout process with optimistic locking",
                "files": ["src/checkout.py", "src/models/order.py", "src/db/transactions.py"],
                "origin": "github.com/org/ecommerce",
                "raw": {
                    "sha": "a1b2c3d4",
                    "author": {"name": "Dev Smith"},
                    "message": "feat: add concurrent checkout process with optimistic locking",
                    "files": [
                        {"filename": "src/checkout.py", "status": "modified"},
                        {"filename": "src/models/order.py", "status": "modified"},
                        {"filename": "src/db/transactions.py", "status": "modified"}
                    ]
                }
            }
        })
        events_added += 1
        
        # 2. Initial increase in database connections
        metric_time = base_time + timedelta(minutes=5)
        for i in range(5):
            self._add_event({
                "event_id": f"sim_metric_{int(time.time())}_{i}",
                "source_type": "metric",
                "timestamp": metric_time + timedelta(seconds=i*2),
                "content": {
                    "metric_id": "database.connections",
                    "value": 50 + i*5,
                    "tags": {"host": "db-primary-1", "region": "us-west"},
                    "origin": "datadog",
                    "raw": {"metric": "database.connections", "value": 50 + i*5}
                }
            })
            events_added += 1
        
        # 3. First warning signs in logs
        log_time = base_time + timedelta(minutes=8)
        warning_logs = [
            "Database connection pool at 80% capacity",
            "Transaction timeout threshold exceeded for user checkout",
            "Deadlock detected in order processing table"
        ]
        
        for i, msg in enumerate(warning_logs):
            self._add_event({
                "event_id": f"sim_log_{int(time.time())}_{i}",
                "source_type": "log",
                "timestamp": log_time + timedelta(seconds=i*10),
                "content": {
                    "timestamp": log_time + timedelta(seconds=i*10),
                    "severity": "WARN" if i < 2 else "ERROR",
                    "message": msg,
                    "origin": "application",
                    "log_id": f"log_{int(time.time())}_{i}",
                    "raw": {"message": msg, "level": "WARN" if i < 2 else "ERROR"}
                }
            })
            events_added += 1
        
        # 4. Error metrics spike
        error_time = base_time + timedelta(minutes=12)
        for i in range(5):
            self._add_event({
                "event_id": f"sim_metric_err_{int(time.time())}_{i}",
                "source_type": "metric",
                "timestamp": error_time + timedelta(seconds=i*3),
                "content": {
                    "metric_id": "checkout.errors",
                    "value": 10 + i*8,
                    "tags": {"service": "checkout", "error_type": "deadlock"},
                    "origin": "datadog",
                    "raw": {"metric": "checkout.errors", "value": 10 + i*8}
                }
            })
            events_added += 1
        
        # 5. Bug report from user
        bug_time = base_time + timedelta(minutes=15)
        self._add_event({
            "event_id": f"sim_bug_{int(time.time())}",
            "source_type": "bug",
            "timestamp": bug_time,
            "content": {
                "report_id": f"bug_{int(time.time())}",
                "summary": "Checkout process fails with database error",
                "entities": ["checkout", "database", "order"],
                "content": "User reported: 'I was trying to complete my purchase but got an error about database deadlock. This happened multiple times.'",
                "origin": "user_report"
            }
        })
        events_added += 1
        
        logger.info(f"Database deadlock simulation complete. Added {events_added} events.")
        return events_added

    def _gen_api_cascade(self) -> int:
        """Simulate a cascading API failure."""
        # Implementation for API cascade scenario
        base_time = datetime.now(timezone.utc) - timedelta(minutes=20)
        events_added = 0
        
        # 1. Initial service degradation
        for i in range(3):
            self._add_event({
                "event_id": f"sim_metric_api_{int(time.time())}_{i}",
                "source_type": "metric",
                "timestamp": base_time + timedelta(minutes=i*2),
                "content": {
                    "metric_id": "api.latency",
                    "value": 200 + i*100,
                    "tags": {"service": "product-catalog", "region": "us-east"},
                    "origin": "datadog",
                    "raw": {"metric": "api.latency", "value": 200 + i*100}
                }
            })
            events_added += 1
        
        # 2. Dependent service failures
        error_time = base_time + timedelta(minutes=5)
        error_logs = [
            "Upstream service product-catalog not responding",
            "Circuit breaker tripped for product-catalog service",
            "Fallback mechanism failed for product recommendations"
        ]
        
        for i, msg in enumerate(error_logs):
            self._add_event({
                "event_id": f"sim_log_api_{int(time.time())}_{i}",
                "source_type": "log",
                "timestamp": error_time + timedelta(seconds=i*15),
                "content": {
                    "timestamp": error_time + timedelta(seconds=i*15),
                    "severity": "ERROR",
                    "message": msg,
                    "origin": "recommendation-service",
                    "log_id": f"log_api_{int(time.time())}_{i}",
                    "raw": {"message": msg, "level": "ERROR"}
                }
            })
            events_added += 1
            
        logger.info(f"API cascade simulation complete. Added {events_added} events.")
        return events_added

    def _gen_memory_leak(self) -> int:
        """Simulate a memory leak in a service."""
        # Implementation for memory leak scenario
        base_time = datetime.now(timezone.utc) - timedelta(minutes=45)
        events_added = 0
        
        # Gradually increasing memory usage
        for i in range(10):
            self._add_event({
                "event_id": f"sim_metric_mem_{int(time.time())}_{i}",
                "source_type": "metric",
                "timestamp": base_time + timedelta(minutes=i*5),
                "content": {
                    "metric_id": "memory.usage",
                    "value": 40 + i*6,
                    "tags": {"service": "image-processing", "pod": "img-proc-xyz"},
                    "origin": "datadog",
                    "raw": {"metric": "memory.usage", "value": 40 + i*6}
                }
            })
            events_added += 1
        
        # OOM error at the end
        oom_time = base_time + timedelta(minutes=48)
        self._add_event({
            "event_id": f"sim_log_mem_{int(time.time())}",
            "source_type": "log",
            "timestamp": oom_time,
            "content": {
                "timestamp": oom_time,
                "severity": "FATAL",
                "message": "Out of memory: Kill process 12345 (image-processor)",
                "origin": "kubernetes",
                "log_id": f"log_mem_{int(time.time())}",
                "raw": {"message": "Out of memory: Kill process 12345 (image-processor)", "level": "FATAL"}
            }
        })
        events_added += 1
            
        logger.info(f"Memory leak simulation complete. Added {events_added} events.")
        return events_added

    def _gen_config_error(self) -> int:
        """Simulate a configuration error after deployment."""
        # Implementation for configuration error scenario
        base_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        events_added = 0
        
        # Config change
        self._add_event({
            "event_id": f"sim_config_{int(time.time())}",
            "source_type": "code",
            "timestamp": base_time,
            "content": {
                "commit_id": "c0nf1gch4nge",
                "author": "Ops Team",
                "message": "chore: update database connection settings",
                "files": ["config/database.yml"],
                "origin": "github.com/org/ecommerce",
                "raw": {
                    "sha": "c0nf1gch4nge",
                    "author": {"name": "Ops Team"},
                    "message": "chore: update database connection settings",
                    "files": [{"filename": "config/database.yml", "status": "modified"}]
                }
            }
        })
        events_added += 1
        
        # Immediate errors
        error_time = base_time + timedelta(minutes=1)
        for i in range(5):
            self._add_event({
                "event_id": f"sim_log_config_{int(time.time())}_{i}",
                "source_type": "log",
                "timestamp": error_time + timedelta(seconds=i*2),
                "content": {
                    "timestamp": error_time + timedelta(seconds=i*2),
                    "severity": "ERROR",
                    "message": "Database connection refused: invalid credentials",
                    "origin": "application",
                    "log_id": f"log_config_{int(time.time())}_{i}",
                    "raw": {"message": "Database connection refused: invalid credentials", "level": "ERROR"}
                }
            })
            events_added += 1
            
        logger.info(f"Configuration error simulation complete. Added {events_added} events.")
        return events_added