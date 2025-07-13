# ReportMiner: AI-Powered Data Extraction and Query System for 

## 🚀 Project Overview

ReportMiner is an intelligent document processing and query system that combines advanced AI capabilities with modern web technologies to extract, analyze, and query data from structured and unstructured reports. This is tartgetd to be used as an Universal Tool for Reprots.

## ✨ Current Status (Latest Updates)

### ✅ **COMPLETED - Backend Core (100%)**
- **Django REST API Framework** - Fully implemented with comprehensive endpoints
- **MCP (Model Context Protocol) Integration** - 10 advanced tools implemented and tested
- **LLM Orchestrator** - Intelligent query routing and tool selection
- **Enhanced RAG Engine** - Vector search with pgvector and OpenAI embeddings
- **File Processing Pipeline** - Supports PDF, DOCX, XLSX with full text extraction
- **Database Schema** - PostgreSQL with optimized document storage and embeddings

### ✅ **COMPLETED - API Endpoints (100%)**
- `POST /api/chat/query/` - Main chat interface with orchestrated AI responses
- `POST /api/chat/upload/` - Enhanced file upload with comprehensive validation
- `GET /api/chat/documents/` - Document listing with system health monitoring


### ✅ MCP Tools 
1. **handle_list_recent_documents** - Document retrieval and filtering
2. **handle_extract_numerical_data** - Mathematical data extraction
3. **handle_calculate_metrics** - Statistical analysis and calculations
4. **handle_domain_analysis** - Document domain and content analysis
5. **handle_visualize_patterns** - Data visualization generation
6. **handle_generate_insights** - AI-powered insight generation
7. **handle_search_documents** - Advanced document search
8. **handle_analyze_sentiment** - Sentiment analysis of document content
9. **handle_compare_documents** - Document comparison and diff analysis
10. **handle_extract_tables** - Table extraction and structuring

### 🔄 **IN PROGRESS - Frontend Development
- React TypeScript application setup needed
- Modern UI components with Material-UI
- Real-time chat interface implementation
- Document management dashboard
- Data visualization components

### ⏳ **PENDING - Integration & Testing**
- Frontend-backend integration
- End-to-end testing
- Performance optimization
- Production deployment

## 🏗️ Architecture

### Backend Stack
- **Framework**: Django 4.2 + Django REST Framework
- **Database**: PostgreSQL with pgvector extension
- **AI Integration**: OpenAI GPT-4 + text-embedding-ada-002
- **Vector Search**: Custom RAG engine with semantic search
- **MCP Protocol**: Advanced tool calling and orchestration
- **File Processing**: PyPDF2, python-docx, openpyxl

### Frontend Stack (Planned)
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI)
- **State Management**: React Query + Context API
- **Routing**: React Router v6
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + MUI

## 🚀 Key Features

### ✅ **Document Processing**
- **Multi-format Support**: PDF, DOCX, XLSX files
- **Smart Text Extraction**: Preserves structure and formatting
- **Automatic Embedding**: Vector embeddings for semantic search
- **Structured Data Extraction**: Tables, key-value pairs, and metadata
- **Content Validation**: File type, size, and security checks

### ✅ **AI-Powered Query Engine**
- **Natural Language Processing**: Chat-based document querying
- **Intelligent Tool Selection**: Automatic MCP tool routing
- **Contextual Responses**: RAG-enhanced answer generation
- **Multi-modal Analysis**: Text, numerical, and visual data processing
- **Real-time Processing**: Sub-second response times

### ✅ **Advanced Analytics**
- **Statistical Analysis**: Automated metrics calculation
- **Data Visualization**: Charts, graphs, and pattern analysis
- **Sentiment Analysis**: Document tone and sentiment detection
- **Document Comparison**: Diff analysis and similarity scoring
- **Domain Analysis**: Content categorization and topic modeling

### 🔄 **Modern Web Interface** (In Development)
- **Responsive Design**: Mobile-first approach
- **Real-time Chat**: WebSocket-based communication
- **Document Dashboard**: Upload, manage, and organize files
- **Visual Analytics**: Interactive charts and data visualization
- **System Monitoring**: Health checks and performance metrics

## 📁 Project Structure

```
ReportMiner/
├── backend/                          # Django backend
│   ├── apps/
│   │   ├── ingestion/               # File upload and processing
│   │   │   ├── models.py           # Document, FileUpload models
│   │   │   ├── enhanced_upload_pipeline.py
│   │   │   └── embeddings.py       # Vector processing
│   │   ├── query/                   # AI query processing
│   │   │   ├── chat_views.py       # Main API endpoints ✅
│   │   │   ├── mcp_server.py       # MCP tools implementation ✅
│   │   │   ├── llm_orchestrator.py # AI orchestration ✅
│   │   │   └── rag_engine.py       # Vector search engine ✅
│   │   └── reportminer/            # Django settings
│   ├── test_mcp_manual.py          # MCP testing scripts ✅
│   └── test_mcp_advanced.py        # Advanced tool testing ✅
├── frontend/                        # React frontend 
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/

```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- Node.js 18+ (for frontend)
- OpenAI API key

### Backend Setup ✅ (COMPLETED)
```bash
# Clone repository
git clone <repository-url>
cd ReportMiner

# Create virtual environment
python -m venv rmvenv
source rmvenv/bin/activate  # Windows: rmvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt #backend directory

# Database setup
createdb reportminer
psql reportminer -c "CREATE EXTENSION vector;"

# Django setup
cd backend
python manage.py migrate
python manage.py collectstatic

# Environment configuration
cp .env.example .env
# Add your OpenAI API key and database credentials

# Start development server
python manage.py runserver
```


## 🧪 Testing

### Backend Testing ✅ (COMPLETED)
```bash
# Test MCP tools
python backend/test_mcp_manual.py
python backend/test_mcp_advanced.py

# Run Django tests
python backend/manage.py test

# Test API endpoints
curl -X POST http://localhost:8000/api/chat/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What documents do I have?"}'
```

### API Examples ✅ (READY FOR USE)

#### Upload Document
```bash
curl -X POST http://localhost:8000/api/chat/upload/ \
  -F "file=@document.pdf" \
  -F "filename=My Document"
```

#### Query Documents
```bash
curl -X POST http://localhost:8000/api/chat/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key financial metrics in my reports?",
    "include_tools": true,
    "session_id": "user_123"
  }'
```

#### List Documents
```bash
curl -X GET http://localhost:8000/api/chat/documents/
```


## 📊 Performance Metrics

### Current Backend Performance ✅
- **Query Response Time**: < 2 seconds average
- **File Processing**: PDF (< 30s), DOCX (< 15s), XLSX (< 10s)
- **Embedding Generation**: ~100 documents/hour
- **Concurrent Users**: Tested up to 10 simultaneous queries
- **Database**: Optimized for 10K+ documents

### Target System Performance 🎯
- **Frontend Load Time**: < 3 seconds
- **Real-time Query**: < 1 second response
- **File Upload**: Progress indicators and chunked transfer
- **System Availability**: 99.9% uptime target

## 🔐 Security Features

### ✅ **Implemented Security**
- File type validation and sanitization
- SQL injection prevention (Django ORM)
- Input validation and cleaning
- Error handling without information leakage
- Rate limiting on API endpoints

### 🔄 **Additional Security (Frontend Phase)**
- JWT authentication
- CORS configuration
- XSS protection
- File upload security scanning
- API key management

## 🤝 Contributing



### Git Workflow
- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - Feature development
- `hotfix/*` - Critical fixes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 and embedding models
- **Django Community** for the robust web framework
- **PostgreSQL** for advanced vector search capabilities
- **MCP Protocol** for standardized AI tool integration

---

**Current Status**: Backend Complete ✅ | Frontend Development Phase 🔄

For questions or support, please contact the development team: TharinduGun