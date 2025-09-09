import tempfile
import logging
import functools
import time
from typing import Any, Callable
import networkx as nx
import matplotlib
# Use a non-interactive backend for Matplotlib to prevent issues on servers.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RootCauseAI")

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f}s")
        return result
    return wrapper

@log_execution_time
def visualize_chain_graph(graph: nx.DiGraph) -> Any:
    """Creates a professional PNG visualization of the failure cause-effect graph."""
    plt.figure(figsize=(14, 10))
    
    # Use a layout that emphasizes hierarchical relationships for better clarity
    try:
        pos = nx.kamada_kawai_layout(graph)
    except:
        pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=100)
    
    # Enhanced color mapping with more distinct colors
    color_map = []
    node_sizes = []
    for node in graph.nodes:
        node_type = graph.nodes[node].get("source_type", "")
        if node_type == "log":
            color_map.append("#ff7f0e")  # Orange
            node_sizes.append(2500)
        elif node_type == "code":
            color_map.append("#2ca02c")  # Green
            node_sizes.append(3000)
        elif node_type == "metric":
            color_map.append("#1f77b4")  # Blue
            node_sizes.append(2800)
        elif node_type == "bug":
            color_map.append("#d62728")  # Red
            node_sizes.append(3200)
        else:
            color_map.append("#7f7f7f")  # Gray
            node_sizes.append(2000)
            
    # Create more informative labels
    labels = {}
    for n in graph.nodes:
        node_data = graph.nodes[n]
        source_type = node_data.get("source_type", "unknown")
        content = node_data.get("content", {})
        
        if source_type == "log":
            labels[n] = f"LOG: {content.get('severity', '')}\n{content.get('message', '')[:20]}..."
        elif source_type == "code":
            labels[n] = f"CODE: {content.get('author', '')}\n{content.get('message', '')[:20]}..."
        elif source_type == "metric":
            labels[n] = f"METRIC: {content.get('metric_id', '')}\n{content.get('value', '')}"
        elif source_type == "bug":
            labels[n] = f"BUG: {content.get('summary', '')[:25]}..."
        else:
            labels[n] = n[:8] + '...' if len(n) > 8 else n

    # Draw with enhanced styling
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color='#444444', width=1.5, alpha=0.7, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')
    
    plt.title("Failure Cause-Effect Chain Analysis", size=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save with higher DPI for better quality
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format="png", dpi=150, bbox_inches='tight')
        plt.close()
        return tmpfile.name