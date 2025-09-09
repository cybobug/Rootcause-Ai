import uuid
import json
import os
import re
import logging
from datetime import datetime
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import pandas as pd
from huggingface_hub import InferenceClient

import networkx as nx
import numpy as np
import openai
from pydantic import BaseModel, Field, validator

# Local imports
from .connectors import ConnectorRegistry, LogConnector, GitHubConnector, MetricsConnector, BugReportConnector, DatadogConnector
from .utils import visualize_chain_graph, log_execution_time


logger = logging.getLogger("RootCauseAI")

# === Data Models ===
class AnalysisStatus(Enum):
    """Status codes for analysis results."""
    SUCCESS = 0
    ERROR_NO_SESSION = 1
    ERROR_LLM_FAILURE = 2
    ERROR_INVALID_DATA = 3
    ERROR_NO_HYPOTHESES = 4

class CausalStep(BaseModel):
    """Model for a step in a causal chain."""
    description: str = Field(..., description="Description of this step in the failure chain")
    evidence_ids: List[str] = Field(..., description="List of evidence IDs supporting this step")

class Hypothesis(BaseModel):
    """Model for a root cause hypothesis."""
    rank: int = Field(..., description="Rank of this hypothesis (1 is most likely)")
    text: str = Field(..., description="Concise description of the root cause")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score from 0-100")
    causal_chain: List[CausalStep] = Field(..., description="Sequence of events leading to the failure")
    recommendations: List[str] = Field(..., description="Recommended actions to fix or mitigate")

    @validator('confidence')
    def validate_confidence(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Confidence must be between 0 and 100')
        return v

class LLMResponse(BaseModel):
    """Model for the expected LLM response format."""
    hypotheses: List[Hypothesis] = Field(..., description="List of root cause hypotheses")

class AnalysisResult(BaseModel):
    """Model for the complete analysis result."""
    status: AnalysisStatus
    status_message: str
    hypotheses: List[Hypothesis] = []
    chain_graph: Any = None
    evidence_cards: Dict[str, Dict] = {}
    mitigation: List[str] = []

# === Data Storage ===
class ArtifactStore:
    """Storage for all system artifacts and events."""
    
    def __init__(self):
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def add_event(self, event: Dict[str, Any]) -> str:
        """Add an event to the store and return its ID."""
        event_id = event.get("log_id") or event.get("commit_id") or \
                   event.get("metric_id") or event.get("report_id") or str(uuid.uuid4())
        
        # Ensure timestamp is a timezone-aware datetime object
        timestamp = event.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            try:
                timestamp = pd.to_datetime(timestamp).tz_localize('UTC')
            except:
                timestamp = datetime.now(pd.UTC)
        elif isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pd.UTC)
        elif isinstance(timestamp, pd.Timestamp) and timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        elif not isinstance(timestamp, (datetime, pd.Timestamp)):
            timestamp = datetime.now(pd.UTC)

        event_obj = {
            "event_id": event_id,
            "source_type": self._infer_type(event),
            "timestamp": timestamp,
            "content": event,
        }
        self.artifacts[event_id] = event_obj
        return event_id

    def _infer_type(self, event: Dict[str, Any]) -> str:
        """Infer the type of an event based on its content."""
        if "log_id" in event: return "log"
        if "commit_id" in event: return "code"
        if "metric_id" in event: return "metric"
        if "report_id" in event: return "bug"
        return "unknown"

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get an event by its ID."""
        return self.artifacts.get(event_id)

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events in the store."""
        return list(self.artifacts.values())

    def group_by_session(self, window_seconds: int = 1800) -> Dict[str, Dict[str, Any]]:
        """Group events into sessions based on temporal proximity."""
        if not self.artifacts:
            return {}
            
        events_sorted = sorted(self.get_all_events(), key=lambda x: x["timestamp"])
        
        self.sessions = {}
        if not events_sorted:
            return {}
            
        current_session_events = [events_sorted[0]]
        
        for i in range(1, len(events_sorted)):
            prev_evt = events_sorted[i-1]
            curr_evt = events_sorted[i]
            if (curr_evt["timestamp"] - prev_evt["timestamp"]).total_seconds() < window_seconds:
                current_session_events.append(curr_evt)
            else:
                self._create_session(current_session_events)
                current_session_events = [curr_evt]
        
        if current_session_events:
            self._create_session(current_session_events)
            
        return self.sessions

    def _create_session(self, events: List[Dict[str, Any]]):
        """Create a session from a list of events."""
        if not events:
            return
            
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "session_id": session_id,
            "events": [e["event_id"] for e in events],
            "context_window": (events[0]["timestamp"], events[-1]["timestamp"]),
        }

class LLMReasoningPipeline:
    """Pipeline for LLM-based reasoning and analysis with OpenAI or Hugging Face."""

    def __init__(self, api_key: Optional[str] = None, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif provider == "huggingface":
            token = api_key or os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("Hugging Face requires a token")
            self.client = openai.OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=token
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @log_execution_time
    def diagnose(self, prompt: str) -> Dict[str, Any]:
        try:
            system_prompt = """You are an expert SRE analyzing system incidents. Your task is to:
            1. Identify the root cause of the incident
            2. Establish the chain of events that led to the failure
            3. Provide actionable recommendations
            
            Look for patterns like:
            - Error logs and their progression
            - Metric anomalies and trends
            - Temporal relationships between events
            - System component dependencies
            
            IMPORTANT: Provide your analysis in this exact JSON format:
            {
                "hypotheses": [
                    {
                        "rank": 1,
                        "text": "Clear description of the root cause",
                        "confidence": 85,
                        "causal_chain": [
                            {
                                "description": "Detailed step in the failure chain",
                                "evidence_ids": ["log_1", "metric_2"]
                            }
                        ],
                        "recommendations": [
                            "Specific, actionable recommendation",
                            "Another concrete action item"
                        ]
                    }
                ]
            }
            
            Include at least one hypothesis. Link every step in your causal chain to specific evidence using the provided IDs.
            Return ONLY the JSON object, no additional text or formatting.
            """
            
            # Prepare the API call parameters
            model = "openai/gpt-oss-120b:cerebras" if self.provider == "huggingface" else self.model
            
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
            }
            
            # Only add response_format for OpenAI
            if self.provider == "openai":
                api_params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**api_params)
            raw_output = response.choices[0].message.content

            return self._parse_output(raw_output)

        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return {"error": str(e)}

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse the raw output from the LLM, handling both JSON and text responses."""
        try:
            # Try to parse as direct JSON first
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to extract JSON from code blocks
            json_match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_output)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to extract JSON without code block markers
            json_match = re.search(r"```\s*([\s\S]+?)\s*```", raw_output)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON-like structure in the response
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                try:
                    return json.loads(raw_output[json_start:json_end+1])
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return error
            logger.error(f"Failed to parse LLM output as JSON: {raw_output[:200]}...")
            return {"error": "Failed to parse LLM output as JSON.", "raw_output": raw_output}
# === Core Analyzer ===
class RootCauseAnalyzer:
    def __init__(self, api_key: str,
                 github_api_token: Optional[str] = None,
                 dd_api_key: Optional[str] = None,
                 dd_app_key: Optional[str] = None,
                 dd_site: Optional[str] = "datadoghq.com",
                 provider: str = "openai",
                 model: str = "gpt-4"):
        """
        RootCauseAnalyzer wraps connectors + LLM pipeline.
        """
        # Initialize core components
        self.store = ArtifactStore()
        self.registry = ConnectorRegistry()
        self.llm_pipeline = LLMReasoningPipeline(
            api_key=api_key,
            provider=provider,
            model=model
        )
        
        # Store API tokens
        self.github_api_token = github_api_token
        self.dd_api_key = dd_api_key
        self.dd_app_key = dd_app_key
        self.dd_site = dd_site

        # Register default connectors
        self._register_default_connectors(github_api_token, dd_api_key, dd_app_key, dd_site)

        logger.info("RootCauseAnalyzer initialized")
    def debug_analysis_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Debug method to inspect analysis context and evidence mapping."""
        # Get session or all events
        if session_id:
            session = self.store.sessions.get(session_id)
            if not session:
                return {"error": "Session not found"}
            events = [self.store.get_event(eid) for eid in session["events"] if self.store.get_event(eid)]
        else:
            events = self.store.get_all_events()
        
        if not events:
            return {"error": "No events found"}
        
        # Build context and evidence mapping
        context, evidence_id_map = self._build_context(events)
        
        # Create debug info
        debug_info = {
            "total_events": len(events),
            "events_by_type": {
                "logs": len(context["logs"]),
                "metrics": len(context["metrics"]), 
                "code": len(context["code"]),
                "bugs": len(context["bugs"])
            },
            "evidence_id_mapping": evidence_id_map,
            "sample_events": {}
        }
        
        # Add sample events from each category
        for category, events_list in context.items():
            if events_list:
                sample = events_list[0]
                debug_info["sample_events"][category] = {
                    "simple_id": sample.get("simple_id"),
                    "event_id": sample.get("event_id"),
                    "timestamp": str(sample.get("timestamp")),
                    "content_keys": list(sample.get("content", {}).keys())
                }
        
        return debug_info

    def debug_prompt_generation(self, session_id: Optional[str] = None) -> str:
        """Generate and return the RAG prompt for debugging."""
        # Get session or create one from all events
        if session_id:
            session = self.store.sessions.get(session_id)
            if not session:
                return "Session not found"
            events = [self.store.get_event(eid) for eid in session["events"] if self.store.get_event(eid)]
        else:
            all_events = self.store.get_all_events()
            if not all_events:
                return "No events found"
            
            timestamps = [e["timestamp"] for e in all_events if "timestamp" in e]
            if timestamps:
                session = {
                    "events": [e["event_id"] for e in all_events],
                    "context_window": (min(timestamps), max(timestamps))
                }
                events = all_events
            else:
                return "No events with timestamps found"
        
        # Build context and generate prompt
        context, evidence_id_map = self._build_context(events)
        prompt = self._build_rag_prompt(context, session, evidence_id_map)
        
        return prompt
    
    def _register_default_connectors(self, github_api_token, dd_api_key, dd_app_key, dd_site):
            """Register all default connectors."""
            self.registry.register("logs", LogConnector())
            self.registry.register("github", GitHubConnector(github_api_token))
            self.registry.register("metrics", MetricsConnector())
            self.registry.register("bug_report", BugReportConnector())
            
            if dd_api_key and dd_app_key:
                self.registry.register("datadog", DatadogConnector(dd_api_key, dd_app_key, dd_site or "datadoghq.com"))

    @log_execution_time
    def ingest_data(self, source_type: str, path: str, **kwargs) -> int:
        """Ingest data from a specific source."""
        try:
            connector = self.registry.get(source_type)
            events = connector.parse(path, **kwargs)
            for event in events:
                self.store.add_event(event)
            return len(events)
        except Exception as e:
            logger.error(f"Failed to ingest data from {source_type}: {e}")
            return 0

    @log_execution_time
    def ingest_live_data(self, source_type: str, **kwargs) -> int:
        """Ingest live data from a source that doesn't use a file path."""
        try:
            connector = self.registry.get(source_type)
            events = connector.parse(**kwargs)
            for event in events:
                self.store.add_event(event)
            return len(events)
        except Exception as e:
            logger.error(f"Failed to ingest live data from {source_type}: {e}")
            return 0

    def group_incidents(self) -> Dict[str, Any]:
        """Group events into incident sessions."""
        return self.store.group_by_session()

    @log_execution_time
    def analyze_session(self, session_id: Optional[str] = None) -> AnalysisResult:
        """Analyze a session to determine root cause hypotheses."""
        # If no session_id provided, analyze all events as one session
        if not session_id:
            all_events = self.store.get_all_events()
            if not all_events:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR_INVALID_DATA,
                    status_message="No events found in store."
                )
            
            # Create a session with all events
            timestamps = [e["timestamp"] for e in all_events if "timestamp" in e]
            if timestamps:
                session = {
                    "events": [e["event_id"] for e in all_events],
                    "context_window": (min(timestamps), max(timestamps))
                }
            else:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR_INVALID_DATA,
                    status_message="No events with timestamps found."
                )
        else:
            session = self.store.sessions.get(session_id)
            if not session:
                return AnalysisResult(
                    status=AnalysisStatus.ERROR_NO_SESSION,
                    status_message="Session not found."
                )

        events = [self.store.get_event(eid) for eid in session["events"] if self.store.get_event(eid)]
        if not events:
            return AnalysisResult(
                status=AnalysisStatus.ERROR_INVALID_DATA,
                status_message="No events found in session."
            )
        
        # Build context and evidence mapping
        context, evidence_id_map = self._build_context(events)
        
        # Build and execute the RAG prompt
        prompt = self._build_rag_prompt(context, session, evidence_id_map)
        llm_output = self.llm_pipeline.diagnose(prompt)
        
        if "error" in llm_output:
            logger.error(f"LLM pipeline error: {llm_output['error']}")
            return AnalysisResult(
                status=AnalysisStatus.ERROR_LLM_FAILURE,
                status_message=f"LLM processing failed: {llm_output['error']}"
            )
        
        # Parse and validate LLM response
        try:
            validated_response = LLMResponse(**llm_output)
            hypotheses = self._rank_hypotheses(validated_response.hypotheses)
        except Exception as e:
            logger.error(f"Failed to validate LLM response: {e}")
            return AnalysisResult(
                status=AnalysisStatus.ERROR_LLM_FAILURE,
                status_message=f"LLM response validation failed: {str(e)}"
            )
        
        if not hypotheses:
            return AnalysisResult(
                status=AnalysisStatus.ERROR_NO_HYPOTHESES,
                status_message="No valid hypotheses generated from analysis."
            )
        
        # Create evidence cards and visualization
        evidence_cards = self._create_evidence_cards(hypotheses, evidence_id_map)
        chain_graph = self._build_event_graph(events)
        mitigation = hypotheses[0].recommendations if hypotheses else []
        
        return AnalysisResult(
            status=AnalysisStatus.SUCCESS,
            status_message="Analysis completed successfully",
            hypotheses=hypotheses,
            chain_graph=chain_graph,
            evidence_cards=evidence_cards,
            mitigation=mitigation
        )

    def _build_context(self, events: List[Dict[str, Any]]) -> Tuple[Dict[str, List], Dict[str, str]]:
        """Build analysis context and evidence mapping."""
        context = {"logs": [], "metrics": [], "code": [], "bugs": []}
        evidence_id_map = {}  # simple_id -> event_id
        simple_counter = {"log": 1, "code": 1, "metric": 1, "bug": 1}
        
        for e in events:
            stype = e["source_type"]
            # Map source types to context categories
            if stype == "log":
                category = "logs"
                simple_prefix = "log"
            elif stype == "code":
                category = "code" 
                simple_prefix = "code"
            elif stype == "metric":
                category = "metrics"
                simple_prefix = "metric"
            elif stype == "bug":
                category = "bugs"
                simple_prefix = "bug"
            else:
                continue  # Skip unknown types
                
            # Create a simple ID for LLM reference
            simple_id = f"{simple_prefix}_{simple_counter[simple_prefix]}"
            simple_counter[simple_prefix] += 1
            evidence_id_map[simple_id] = e["event_id"]
            
            # Add simple_id to the event for the prompt
            e_with_simple_id = e.copy()
            e_with_simple_id["simple_id"] = simple_id
            context[category].append(e_with_simple_id)
            
        return context, evidence_id_map

    def _build_rag_prompt(self, context: Dict[str, Any], session: Dict[str, Any], 
                     evidence_id_map: Dict[str, str]) -> str:
        """Build the RAG prompt for LLM analysis."""
        prompt_parts = [
            "Analyze this incident and determine its root cause. Follow these steps:",
            "1. Review the chronological sequence of events",
            "2. Identify critical errors and anomalies", 
            "3. Establish cause-and-effect relationships",
            "4. Determine the initial trigger and root cause",
            "5. Suggest specific preventive measures",
            "\nFormat your analysis as JSON following the schema provided.",
            f"\nINCIDENT TIMELINE:",
            f"Start: {session['context_window'][0].isoformat()}",
            f"End: {session['context_window'][1].isoformat()}"
        ]

        # Enhanced RAG for Logs: Show complete event sequence with clear IDs
        logs = sorted(context['logs'], key=lambda x: x['timestamp'])
        
        if logs:
            log_context_str = "\n=== LOG EVENTS SEQUENCE ===\n"
            for log in logs:
                timestamp = log['timestamp'].isoformat()
                severity = log['content']['severity']
                msg = log['content']['message']
                log_id = log['simple_id']
                origin = log['content'].get('origin', 'unknown')
                
                log_context_str += f"[{log_id}] {timestamp} | {severity} | {msg}\n"
                log_context_str += f"    Source: {origin}\n\n"
            
            prompt_parts.append(log_context_str.strip())
            
            # Highlight critical errors
            critical_logs = [log for log in logs if log['content'].get('severity') in ['ERROR', 'FATAL', 'CRITICAL']]
            if critical_logs:
                critical_str = "\n=== CRITICAL ERRORS ===\n"
                for log in critical_logs:
                    critical_str += f"[{log['simple_id']}] {log['content']['severity']}: {log['content']['message']}\n"
                prompt_parts.append(critical_str.strip())

        # RAG for Metrics: Detect anomalies and show progression
        if context['metrics']:
            metrics_by_id = {}
            for m in context['metrics']:
                mid = m['content']['metric_id']
                if mid not in metrics_by_id:
                    metrics_by_id[mid] = []
                timestamp = m['timestamp'].isoformat() if isinstance(m['timestamp'], datetime) else m['timestamp']
                metrics_by_id[mid].append((timestamp, float(m['content']['value']), m['simple_id']))
            
            metrics_str = "\n=== METRICS DATA ===\n"
            for mid, values in metrics_by_id.items():
                values.sort(key=lambda x: x[0])
                
                # Calculate statistics
                numeric_values = [v[1] for v in values]
                mean = np.mean(numeric_values)
                std_dev = np.std(numeric_values) if len(numeric_values) > 1 else 0
                threshold = mean + 2 * std_dev
                
                metrics_str += f"Metric: {mid}\n"
                # Show progression
                if len(values) >= 2:
                    start_val = values[0][1]
                    end_val = values[-1][1]
                    max_val = max(v[1] for v in values)
                    metrics_str += f"  Trend: {start_val:.2f} → {max_val:.2f} → {end_val:.2f}\n"
                
                # Show anomalies
                anomalies = [(ts, v, sid) for ts, v, sid in values if v > threshold and std_dev > 0]
                if anomalies:
                    metrics_str += f"  Anomalies (>{threshold:.2f}):\n"
                    for ts, v, sid in anomalies:
                        metrics_str += f"    [{sid}] {ts}: {v:.2f}\n"
                metrics_str += "\n"
                
            if metrics_str.strip() != "=== METRICS DATA ===":
                prompt_parts.append(metrics_str.strip())

        # Code Changes
        if context['code']:
            code_str = "\n=== RECENT CODE CHANGES ===\n"
            for e in context['code'][:5]:
                author = e['content']['author']
                msg = e['content']['message'].splitlines()[0]
                files = ', '.join(e['content'].get('files', [])[:3])
                code_str += f"[{e['simple_id']}] {author}: {msg}\n"
                if files:
                    code_str += f"    Files: {files}\n"
            prompt_parts.append(code_str.strip())

        # Bug reports
        if context['bugs']:
            bug_str = "\n=== USER BUG REPORTS ===\n"
            for e in context['bugs']:
                summary = e['content']['summary']
                entities = ', '.join(e['content'].get('entities', [])[:5])
                bug_str += f"[{e['simple_id']}] {summary}\n"
                if entities:
                    bug_str += f"    Related: {entities}\n"
            prompt_parts.append(bug_str.strip())

        # Enhanced task definition with example
        prompt_parts.append(f"""
    === AVAILABLE EVIDENCE IDS ===
    {', '.join(evidence_id_map.keys())}

    === ANALYSIS TASK ===
    Based on the evidence above, identify the root cause of this incident. Use the exact evidence IDs shown above in your causal chain.

    Respond with a single, valid JSON object in this format:
    {{
    "hypotheses": [
        {{
        "rank": 1,
        "text": "Database connection pool exhaustion caused by slow queries and high load",
        "confidence": 85,
        "causal_chain": [
            {{"description": "Initial database performance degradation with slow queries", "evidence_ids": ["log_4", "log_5"]}},
            {{"description": "Connection pool capacity warnings escalated", "evidence_ids": ["log_3", "log_6"]}},
            {{"description": "Connection timeouts began affecting user operations", "evidence_ids": ["log_7", "log_8"]}},
            {{"description": "Complete pool exhaustion led to service failure", "evidence_ids": ["log_12", "log_13"]}}
        ],
        "recommendations": [
            "Implement connection pool monitoring and alerting",
            "Add query performance optimization",
            "Configure automatic connection pool scaling"
        ]
        }}
    ]
    }}

    CRITICAL: Only use evidence IDs from the list above: {', '.join(evidence_id_map.keys())}
    """)

        final_prompt = "\n".join(prompt_parts)
        logger.info(f"Built RAG prompt with {len(final_prompt)} characters, {len(evidence_id_map)} evidence items")
        return final_prompt

    def _rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Rank hypotheses by confidence and other factors."""
        return sorted(hypotheses, key=lambda h: (-h.confidence, h.rank))

    def _create_evidence_cards(self, hypotheses: List[Hypothesis], evidence_id_map: Dict[str, str]) -> Dict[str, Dict]:
        """Create evidence cards from hypotheses using the evidence mapping."""
        cards = {}
        all_evidence_ids = set()
        
        # Collect all referenced evidence IDs
        for hypo in hypotheses:
            for step in hypo.causal_chain:
                for simple_id in step.evidence_ids:
                    real_event_id = evidence_id_map.get(simple_id)
                    if real_event_id:
                        all_evidence_ids.add(real_event_id)
        
        # Create cards for each evidence item
        for eid in all_evidence_ids:
            event = self.store.get_event(eid)
            if event:
                content = event['content']
                cards[eid] = {
                    "summary": content.get('message', content.get('summary', 'N/A')),
                    "origin": content.get('origin', 'N/A'),
                    "timestamp": event['timestamp'].isoformat() if event['timestamp'] else 'N/A',
                    "type": event['source_type']
                }
        
        return cards
        
    def _build_event_graph(self, events: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build a directed graph of events for visualization."""
        G = nx.DiGraph()
        for e in events:
            G.add_node(e["event_id"], **e)
        
        # Add temporal edges
        for i in range(len(events) - 1):
            e1 = events[i]
            e2 = events[i+1]
            G.add_edge(e1["event_id"], e2["event_id"], type="temporal")
        
        return G

    def visualize_graph(self, graph: nx.DiGraph) -> Any:
        """Generate a visualization of the event graph."""
        return visualize_chain_graph(graph)

    def log_feedback(self, session_id: str, analysis_result: AnalysisResult, is_helpful: bool):
        """Log user feedback to improve the system."""
        try:
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "feedback": "helpful" if is_helpful else "unhelpful",
                "hypothesis": analysis_result.hypotheses[0].dict() if analysis_result.hypotheses else {}
            }
            
            with open("feedback_log.jsonl", "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")
                
            logger.info(f"Feedback logged for session {session_id}: {feedback_entry['feedback']}")
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")