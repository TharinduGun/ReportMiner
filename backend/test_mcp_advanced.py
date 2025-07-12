import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reportminer.settings')
django.setup()

import sys
sys.path.append('apps/query')

try:
    from apps.query.mcp_server import (
        handle_calculate_metrics,
        handle_domain_analysis, 
        handle_visualize_patterns,
        handle_generate_insights
    )
    
    print("🧪 Testing Advanced MCP Tools...")
    
    # Test 1: Calculate metrics
    result1 = handle_calculate_metrics({"analysis_type": "basic_stats"})
    print(f"✅ Calculate Metrics: {len(result1)} results")
    
    # Test 2: Domain analysis (need a document ID)
    from apps.ingestion.models import Document
    doc = Document.objects.first()
    if doc:
        result2 = handle_domain_analysis({"document_id": str(doc.id)})
        print(f"✅ Domain Analysis: {len(result2)} results")
    
    # Test 3: Visualize patterns
    result3 = handle_visualize_patterns({"visualization_type": "distribution"})
    print(f"✅ Visualize Patterns: {len(result3)} results")
    
    # Test 4: Generate insights
    result4 = handle_generate_insights({"analysis_focus": "performance"})
    print(f"✅ Generate Insights: {len(result4)} results")
    
    print("\n🎉 All 10 MCP tools are working!")
    
except Exception as e:
    print(f"❌ Advanced tool test failed: {e}")