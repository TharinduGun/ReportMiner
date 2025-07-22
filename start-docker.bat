@echo off
echo Starting ReportMiner with Docker...
echo.

echo Building Docker images...
docker-compose build

echo.
echo Starting services...
docker-compose up -d

echo.
echo Services started! You can access the application at:
echo Backend API: http://localhost:8000
echo.
echo To view logs: docker-compose logs -f
echo To stop services: docker-compose down
echo.
echo Press any key to view logs...
pause > nul
docker-compose logs -f