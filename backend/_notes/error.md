## RESOLVED: MCP TaskGroup Error

### Original Error
Warning: Could not load MCP tools: unhandled errors in a TaskGroup (1 sub-exception)

### Root Cause
Incorrect path resolution in `apps/query/services.py` line 76:
- `Path(__file__).parent.parent` resolved to `/backend/apps/` 
- Looking for `mcp_server.py` in wrong directory
- MultiServerMCPClient failed to start subprocess with non-existent file

### Fix Applied
- Changed path resolution to `Path(__file__).parent.parent.parent`
- Added file existence check before MCP client connection
- Enhanced error logging for better debugging

### Status: ✅ FIXED
- Path now correctly resolves to `C:\FYP\ReportMiner\backend\mcp_server.py`
- MCP tools loading successfully (4 tools detected)
- No more TaskGroup errors