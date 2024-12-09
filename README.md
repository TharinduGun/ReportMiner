# ReportMiner:AI-Powered Data Extraction and Query System for Structured and Unstructured Reports

## Features

  1.File Ingestion Pipeline
Supports file uploads (e.g., PDFs, Word documents).
Validates and processes input files.

2.AI-Based Data Extraction
Leverages OpenAI's GPT models for extracting key information from text.
Handles structured, loosely structured, and unstructured reports.

3.Scalable Database Design
Stores extracted data in a relational or NoSQL database for efficient querying.

4.Natural Language Query Processing
Allows users to interact with the system using natural language.
Translates queries into database operations and retrieves results.

5.User-Friendly Data Presentation
Displays query results in organized formats, including tables and JSON.


## Technologies Used

## Backend
Python (3.11)
Django (with Django REST Framework)
PostgreSQL (or MongoDB for NoSQL)

## Frontend
Django Templates (or optional React)

## AI Integration
OpenAI API (e.g., GPT-4)

## Other Tools
Docker (for containerization)
PyPDF2, pdfminer.six, python-docx (for file parsing)
Redis (optional, for caching query results)
