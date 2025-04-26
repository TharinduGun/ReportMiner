from django.http import JsonResponse

# Basic homepage view
def home(request):
    return JsonResponse({
        "message": "Welcome to ReportMiner API 🚀",
        "status": "Server Running",
        "available_endpoints": {
            "Admin Panel": "/admin/",
            "File Upload API": "/api/ingestion/upload/"
        }
    })
