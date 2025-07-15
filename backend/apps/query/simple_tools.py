# CREATE: apps/query/simple_tools.py
"""
Simple tool alternatives for MCP functions
These provide basic functionality while maintaining system stability
"""

def simple_tool_info():
    """Information about available simple tools"""
    return {
        "tools": [
            {
                "name": "extract_numerical",
                "description": "Extract numerical data using RAG search",
                "status": "active"
            },
            {
                "name": "calculate_metrics", 
                "description": "Basic metrics calculation guidance",
                "status": "active"
            },
            {
                "name": "domain_analysis",
                "description": "Document domain analysis via RAG",
                "status": "active"
            },
            {
                "name": "visualize_patterns",
                "description": "Visualization planning (enhanced features coming)",
                "status": "placeholder"
            },
            {
                "name": "generate_insights",
                "description": "AI-powered insights using RAG",
                "status": "enhanced"
            }
        ],
        "migration_status": "MCP tools replaced with RAG-based alternatives",
        "future_plans": "Full MCP integration can be restored when async issues resolved"
    }