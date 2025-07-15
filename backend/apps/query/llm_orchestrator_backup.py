# CREATE: backend/apps/query/llm_orchestrator.py
"""
Intelligent LLM Orchestrator for ReportMiner
Decides which tools to use and orchestrates responses
"""

import logging
from typing import Dict, List, Any, Optional
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from .rag_engine import get_rag_engine
from .mcp_server import (
    handle_extract_numerical_data,
    handle_calculate_metrics,
    handle_domain_analysis,
    handle_visualize_patterns,
    handle_generate_insights
)

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """
    Intelligent orchestrator that:
    1. Analyzes user questions
    2. Selects appropriate tools (RAG + MCP)
    3. Orchestrates tool execution
    4. Synthesizes comprehensive responses
    """
    
    def __init__(self):
        # High-accuracy LLM for orchestration
        self.llm = ChatOpenAI(
            model="gpt-4o",           # Upgrade to full version
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.0,          # Zero for deterministic decisions
            max_tokens=1500
        )
        
        # RAG engine
        self.rag_engine = get_rag_engine()
        
        # Tool mapping
        self.available_tools = {
            'rag_query': self._execute_rag,
            'extract_numerical': handle_extract_numerical_data,
            'calculate_metrics': handle_calculate_metrics,
            'domain_analysis': handle_domain_analysis,
            'visualize_patterns': handle_visualize_patterns,
            'generate_insights': handle_generate_insights
        }
        
        # Decision prompt
        self.decision_prompt = self._create_decision_prompt()
        self.synthesis_prompt = self._create_synthesis_prompt()
    
    def _create_decision_prompt(self) -> PromptTemplate:
        """Create prompt for intelligent tool selection"""
        template = """You are an AI assistant that decides which tools to use for document analysis queries.

Available tools:
1. rag_query - Search and retrieve information from documents
2. extract_numerical - Extract numbers, dates, financial data
3. calculate_metrics - Calculate statistics, trends, performance metrics
4. domain_analysis - Analyze document types and domains
5. visualize_patterns - Create data visualizations
6. generate_insights - Generate AI-powered insights and recommendations

User Question: {question}

Instructions:
1. Analyze the user's question
2. Select 1-3 most appropriate tools
3. For each tool, specify the parameters needed
4. Return as JSON format

Example Response:
{{
    "tools_to_use": [
        {{
            "tool": "rag_query",
            "reason": "Need to search documents for specific information",
            "parameters": {{"include_sources": true}}
        }},
        {{
            "tool": "extract_numerical",
            "reason": "User asking about financial data",
            "parameters": {{"analysis_type": "financial"}}
        }}
    ],
    "reasoning": "User wants financial information, so need RAG search plus numerical extraction"
}}

Response:"""
        return PromptTemplate(template=template, input_variables=["question"])
    
    def _create_synthesis_prompt(self) -> PromptTemplate:
        """Create prompt for synthesizing results"""
        template = """You are an AI assistant that creates comprehensive responses by combining results from multiple analysis tools.

Original Question: {question}

Tool Results:
{tool_results}

Instructions:
1. Create a comprehensive answer that addresses the user's question
2. Integrate insights from all tool results
3. Provide specific details, numbers, and examples
4. Include source attribution when available
5. Structure the response clearly with sections if needed
6. End with actionable insights or recommendations if appropriate

Response Format:
- Start with a direct answer to the question
- Provide supporting details from the analysis
- Include specific data points and sources
- End with key insights

Response:"""
        return PromptTemplate(
            template=template,
            input_variables=["question", "tool_results"]
        )
    
    def process_query(self, question: str, document_ids: List[str] = None) -> Dict[str, Any]:
        """
        Main orchestration method
        """
        try:
            # Step 1: Analyze question and select tools
            logger.info(f"🧠 Orchestrating query: {question}")
            
            tool_plan = self._plan_tool_execution(question)
            logger.info(f"📋 Selected tools: {[t['tool'] for t in tool_plan['tools_to_use']]}")
            
            # Step 2: Execute selected tools
            tool_results = self._execute_tools(tool_plan, question, document_ids)
            
            # Step 3: Synthesize comprehensive response
            final_response = self._synthesize_response(question, tool_results)
            
            return {
                "success": True,
                "answer": final_response,
                "tool_plan": tool_plan,
                "tool_results": tool_results,
                "confidence": self._calculate_confidence(tool_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error processing your request."
            }
    
    def _plan_tool_execution(self, question: str) -> Dict[str, Any]:
        """Use LLM to decide which tools to use"""
        try:
            prompt = self.decision_prompt.format(question=question)
            response = self.llm.invoke(prompt).content
            
            # Parse JSON response
            import json
            plan = json.loads(response)
            return plan
            
        except Exception as e:
            logger.warning(f"Tool planning failed, using fallback: {e}")
            # Fallback: always use RAG + insights
            return {
                "tools_to_use": [
                    {"tool": "rag_query", "parameters": {"include_sources": True}},
                    {"tool": "generate_insights", "parameters": {"analysis_focus": "comprehensive"}}
                ],
                "reasoning": "Fallback: RAG search + insights generation"
            }
    
    def _execute_tools(self, plan: Dict[str, Any], question: str, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Execute selected tools in sequence"""
        results = []
        
        for tool_spec in plan["tools_to_use"]:
            tool_name = tool_spec["tool"]
            parameters = tool_spec.get("parameters", {})
            
            try:
                logger.info(f"🔧 Executing tool: {tool_name}")
                
                if tool_name == "rag_query":
                    result = self._execute_rag(question, parameters, document_ids)
                else:
                    tool_func = self.available_tools.get(tool_name)
                    if tool_func:
                        result = tool_func(parameters)
                        # Convert MCP result format
                        if hasattr(result, '__iter__') and not isinstance(result, str):
                            result = {"success": True, "data": [item.text if hasattr(item, 'text') else str(item) for item in result]}
                        else:
                            result = {"success": True, "data": str(result)}
                    else:
                        result = {"success": False, "error": f"Unknown tool: {tool_name}"}
                
                results.append({
                    "tool": tool_name,
                    "result": result,
                    "parameters": parameters
                })
                
            except Exception as e:
                logger.error(f"❌ Tool {tool_name} failed: {e}")
                results.append({
                    "tool": tool_name,
                    "result": {"success": False, "error": str(e)},
                    "parameters": parameters
                })
        
        return results
    
    def _execute_rag(self, question: str, parameters: Dict[str, Any], document_ids: List[str] = None) -> Dict[str, Any]:
        """Execute RAG query with enhanced parameters"""
        try:
            include_sources = parameters.get("include_sources", True)
            result = self.rag_engine.query(question, include_sources=include_sources)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _synthesize_response(self, question: str, tool_results: List[Dict[str, Any]]) -> str:
        """Create comprehensive response from tool results"""
        try:
            # Format tool results for LLM
            formatted_results = []
            for result in tool_results:
                tool_name = result["tool"]
                tool_data = result["result"]
                
                if tool_data.get("success"):
                    if tool_name == "rag_query" and "answer" in tool_data:
                        formatted_results.append(f"Document Search Results: {tool_data['answer']}")
                        if tool_data.get("sources"):
                            formatted_results.append(f"Sources: {len(tool_data['sources'])} documents")
                    else:
                        data = tool_data.get("data", "No data")
                        formatted_results.append(f"{tool_name.title()} Results: {data}")
                else:
                    formatted_results.append(f"{tool_name.title()}: Error - {tool_data.get('error', 'Unknown error')}")
            
            results_text = "\n".join(formatted_results)
            
            # Synthesize using LLM
            prompt = self.synthesis_prompt.format(
                question=question,
                tool_results=results_text
            )
            
            response = self.llm.invoke(prompt).content
            return response
            
        except Exception as e:
            logger.error(f"❌ Response synthesis failed: {e}")
            return "I was able to gather some information but encountered difficulties creating a comprehensive response."
    
    def _calculate_confidence(self, tool_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on tool success rates"""
        if not tool_results:
            return 0.0
        
        successful_tools = sum(1 for result in tool_results if result["result"].get("success", False))
        confidence = successful_tools / len(tool_results)
        return round(confidence, 2)

# Convenience function
def get_orchestrator() -> LLMOrchestrator:
    """Get initialized orchestrator instance"""
    return LLMOrchestrator()


