import re
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import pandas as pd
import requests
from datadog_api_client.v1 import ApiClient, ApiException
from datadog_api_client import Configuration, ApiClient
from datadog_api_client.v2 import ApiException
from datadog_api_client.v2.api import metrics_api

logger = logging.getLogger("RootCauseAI")

# === Base Classes ===
class BaseConnector(ABC):
    """Abstract base class for all data connectors."""
    
    @abstractmethod
    def parse(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Parse data from the source and return a list of events."""
        pass

class BaseAPIConnector(BaseConnector):
    """Base class for API connectors with session management."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.session = requests.Session()
        if api_token:
            self.session.headers["Authorization"] = f"token {api_token}"
        self.session.headers["User-Agent"] = "RootCauseAI/1.0"
        
    def _make_request(self, url: str, params: Optional[Dict] = None, method: str = "GET") -> Optional[Any]:
        """Helper method to make HTTP requests with error handling."""
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {url}: {e}")
            return None

    def _parse_link_header(self, link_header: str) -> Dict[str, str]:
        """Parse GitHub's Link header to extract pagination URLs."""
        links = {}
        if not link_header:
            return links
            
        # Split by commas and parse each link
        for link in link_header.split(','):
            link = link.strip()
            # Extract URL and rel value
            url_match = re.search(r'<([^>]+)>', link)
            rel_match = re.search(r'rel="([^"]+)"', link)
            
            if url_match and rel_match:
                url = url_match.group(1)
                rel = rel_match.group(1)
                links[rel] = url
                
        return links

# === Connector Registry ===
class ConnectorRegistry:
    """Registry for managing all available data connectors."""
    
    def __init__(self):
        self._connectors = {}

    def register(self, name: str, connector_instance: BaseConnector):
        """Register a new connector."""
        if not isinstance(connector_instance, BaseConnector):
            raise ValueError("Connector must inherit from BaseConnector")
        self._connectors[name] = connector_instance
        logger.info(f"Registered connector: {name}")

    def get(self, name: str) -> BaseConnector:
        """Get a connector by name."""
        connector = self._connectors.get(name)
        if not connector:
            raise ValueError(f"No connector registered with the name: {name}")
        return connector

    def list_connectors(self) -> List[str]:
        """List all registered connector names."""
        return list(self._connectors.keys())

# === Concrete Connectors ===
class LogConnector(BaseConnector):
    """Connector for parsing log files with various formats."""
    
    SEVERITY_LEVELS = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"]
    LOG_FORMATS = {
        "json": re.compile(r'^{.*}$'),
        "common": re.compile(r'^(?P<timestamp>\S+) (?P<level>\S+) (?P<message>.*)$'),
        "apache": re.compile(r'^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(.+?)" (\d{3}) (\d+)'),
    }

    def _parse_json_log(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON-formatted log entry."""
        try:
            record = json.loads(line)
            if not all(k in record for k in ["timestamp", "message"]):
                return None
                
            severity = str(record.get("level", record.get("severity", "INFO"))).upper()
            if severity not in self.SEVERITY_LEVELS:
                severity = "INFO"
                
            return {
                "timestamp": pd.to_datetime(record["timestamp"], utc=True, errors='coerce'),
                "severity": severity,
                "message": record["message"],
                "origin": record.get("origin", "log_upload"),
                "log_id": record.get("log_id", str(uuid.uuid4())),
                "raw": record
            }
        except (json.JSONDecodeError, TypeError):
            return None

    def _parse_common_log(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a common log format entry."""
        match = re.match(self.LOG_FORMATS["common"], line)
        if not match:
            return None
            
        groups = match.groupdict()
        return {
            "timestamp": pd.to_datetime(groups["timestamp"], utc=True, errors='coerce'),
            "severity": groups["level"].upper(),
            "message": groups["message"],
            "origin": "log_upload",
            "log_id": str(uuid.uuid4()),
            "raw": {"line": line}
        }

    def parse_log_entry(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log entry using available formats."""
        # Try JSON format first
        parsed = self._parse_json_log(line)
        if parsed:
            return parsed
            
        # Try common format
        parsed = self._parse_common_log(line)
        if parsed:
            return parsed
            
        # Fallback: extract timestamp and severity if possible
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)", line)
        sev_match = re.search(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", line, re.IGNORECASE)
        
        if not ts_match:
            return None
            
        severity = sev_match.group(1).upper() if sev_match else "INFO"
        return {
            "timestamp": pd.to_datetime(ts_match.group(1), utc=True, errors='coerce'),
            "severity": severity,
            "message": line.strip(),
            "origin": "log_upload",
            "log_id": str(uuid.uuid4()),
            "raw": {"line": line}
        }

    def parse(self, path: str) -> List[Dict[str, Any]]:
        """Parse a log file and return all valid log entries."""
        events = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = self.parse_log_entry(line.strip())
                        if entry and pd.notna(entry["timestamp"]):
                            events.append(entry)
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to read log file {path}: {e}")
            
        logger.info(f"Parsed {len(events)} log entries from {path}")
        return events


class GitHubConnector(BaseAPIConnector):
    """Connector for fetching commit data from GitHub with pagination support."""
    
    def __init__(self, api_token: Optional[str] = None):
        super().__init__(api_token)
        self.base_url = "https://api.github.com"

    def _fetch_commit_details(self, repo: str, sha: str) -> List[str]:
        """Fetch detailed file information for a specific commit."""
        url = f"{self.base_url}/repos/{repo}/commits/{sha}"
        data = self._make_request(url)
        if not data or "files" not in data:
            return []
        return [f.get("filename") for f in data.get("files", []) if f.get("filename")]

    def _make_paginated_request(self, url: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Make paginated requests to GitHub API and return all results."""
        all_results = []
        current_url = url
        
        while current_url:
            try:
                response = self.session.get(current_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Get the data from this page
                page_data = response.json()
                if isinstance(page_data, list):
                    all_results.extend(page_data)
                else:
                    # If it's not a list, it might be a single object or error
                    break
                
                # Parse the Link header for next page
                link_header = response.headers.get('Link', '')
                links = self._parse_link_header(link_header)
                
                # Set up for next iteration
                current_url = links.get('next')
                params = None  # Parameters are already in the next URL
                
                # Safety check: limit to reasonable number of pages
                if len(all_results) > 1000:  # Limit to ~1000 commits
                    logger.warning(f"Stopping pagination after {len(all_results)} results to avoid excessive API calls")
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed for {current_url}: {e}")
                break
        
        return all_results

    def parse(self, repo: str, since: Optional[datetime] = None, 
              until: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch commits from a GitHub repository within a time range with full pagination."""
        url = f"{self.base_url}/repos/{repo}/commits"
        params = {"per_page": 100}  # Maximum per page
        
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()
        
        # Use paginated request to get all commits
        commits_data = self._make_paginated_request(url, params=params)
        if not commits_data:
            return []
            
        commits = []
        for item in commits_data:
            commit = item.get("commit", {})
            author = commit.get("author", {})
            files = self._fetch_commit_details(repo, item.get("sha", ""))
            
            commit_data = {
                "commit_id": item.get("sha", ""),
                "timestamp": pd.to_datetime(author.get("date"), utc=True, errors='coerce'),
                "author": author.get("name", "Unknown"),
                "message": commit.get("message", ""),
                "files": files,
                "origin": repo,
                "raw": item
            }
            
            if pd.notna(commit_data["timestamp"]):
                commits.append(commit_data)
                
        logger.info(f"Fetched {len(commits)} commits from {repo} across {len(commits_data)} total API results")
        return commits


class DatadogConnector(BaseConnector):
    """Connector for fetching metrics from Datadog."""
    
    def __init__(self, api_key: str, app_key: str, site: str = "datadoghq.com"):
        configuration = Configuration()
        configuration.api_key["apiKeyAuth"] = api_key
        configuration.api_key["appKeyAuth"] = app_key
        configuration.server_variables["site"] = site
        self.api_client = ApiClient(configuration)

    def parse(self, query: str, start_ts: datetime, end_ts: datetime) -> List[Dict[str, Any]]:
        """Query metrics from Datadog within a time range."""
        api_instance = metrics_api.MetricsApi(api_client=self.api_client)
        events = []
        
        try:
            response = api_instance.query_metrics(
                _from=int(start_ts.timestamp()),
                to=int(end_ts.timestamp()),
                query=query
            )
            
            if response.series:
                for series in response.series:
                    metric_name = series.get("metric", "unknown_metric")
                    tags = {f"dd_{k}": v for k, v in series.get("scope", {}).items()}
                    
                    for point in series.get("pointlist", []):
                        timestamp = pd.to_datetime(point[0], unit='ms', utc=True)
                        value = point[1]
                        
                        events.append({
                            "timestamp": timestamp,
                            "metric_id": metric_name,
                            "value": value,
                            "tags": tags,
                            "origin": "datadog",
                            "raw": {"metric": metric_name, "scope": series.get("scope")}
                        })
        except ApiException as e:
            logger.error(f"Datadog API Error: {e}")
            
        logger.info(f"Fetched {len(events)} metric events from Datadog")
        return [e for e in events if pd.notna(e["timestamp"])]


class MetricsConnector(BaseConnector):
    """Connector for parsing metric data from files."""
    
    def _parse_csv(self, path: str) -> List[Dict[str, Any]]:
        """Parse metrics from a CSV file."""
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            events = []
            
            for _, row in df.iterrows():
                events.append({
                    "timestamp": row["timestamp"],
                    "metric_id": row.get("metric_id", "unknown"),
                    "value": row["value"],
                    "tags": {k: row[k] for k in row.index if k not in ("timestamp", "value")},
                    "origin": "metric_upload",
                    "raw": row.to_dict()
                })
                
            return events
        except Exception as e:
            logger.error(f"Failed to parse CSV metrics file {path}: {e}")
            return []

    def _parse_json(self, path: str) -> List[Dict[str, Any]]:
        """Parse metrics from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            events = []
            for rec in data:
                events.append({
                    "timestamp": pd.to_datetime(rec["timestamp"], utc=True, errors='coerce'),
                    "metric_id": rec.get("metric_id", "unknown"),
                    "value": rec["value"],
                    "tags": rec.get("tags", {}),
                    "origin": "metric_upload",
                    "raw": rec
                })
                
            return events
        except Exception as e:
            logger.error(f"Failed to parse JSON metrics file {path}: {e}")
            return []

    def parse(self, path: str) -> List[Dict[str, Any]]:
        """Parse a metrics file based on its extension."""
        if path.endswith(".csv"):
            parsed = self._parse_csv(path)
        elif path.endswith(".json"):
            parsed = self._parse_json(path)
        else:
            logger.error(f"Unsupported metrics file format: {path}")
            return []
            
        valid_events = [p for p in parsed if pd.notna(p["timestamp"])]
        logger.info(f"Parsed {len(valid_events)} metric events from {path}")
        return valid_events


class BugReportConnector(BaseConnector):
    """Connector for parsing bug reports from text files."""
    
    def parse(self, path: str) -> List[Dict[str, Any]]:
        """Parse a bug report file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                
            if not text:
                return []
                
            report_id = str(uuid.uuid4())
            summary = text.split("\n")[0][:100]  # First line as summary, limit length
            
            # Extract entities mentioned in the report
            entities = set()
            entity_patterns = [
                r"\b(service|component|module|error|code|function|endpoint|api)\s*[=:]\s*(\w+)",
                r"\b([A-Z][a-z]+[A-Z][a-zA-Z]+)\b",  # CamelCase words
                r"\b([a-z]+_[a-z]+)\b",  # snake_case words
            ]
            
            for pattern in entity_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        entities.add(match[1].lower())
                    else:
                        entities.add(match.lower())
                        
            return [{
                "timestamp": datetime.utcnow(),
                "report_id": report_id,
                "summary": summary,
                "entities": list(entities),
                "content": text,
                "origin": "bug_report_upload"
            }]
        except Exception as e:
            logger.error(f"Failed to parse bug report {path}: {e}")
            return []