"""
Enhanced Text Processing Pipeline for ReportMiner
Handles text segmentation, table detection, and structured data extraction
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from django.utils import timezone
from .models import Document, DocumentTextSegment, DocumentTable, DocumentStructuredData, DocumentKeyValue
from .vector_processor import VectorProcessor

class TextProcessor:
    """Enhanced text processor that segments content and extracts structured data"""
    
    def __init__(self):
        self.sequence_counter = 0
    
    def process_document_text(self, document: Document, raw_text: str) -> Dict[str, Any]:
        """
        Main processing function that coordinates all text processing tasks
        
        Args:
            document: Django Document model instance
            raw_text: Raw extracted text from the document
            
        Returns:
            Dict containing processing results and statistics
        """
        processing_results = {
            'text_segments': 0,
            'tables': 0,
            'key_values': 0,
            'processing_status': 'completed',
            'errors': []
        }
        
        try:
            # Update document processing status
            document.processing_status = 'processing'
            document.processing_started_at = timezone.now()
            document.save()
            
            # 1. Segment the text into manageable chunks
            segments = self._segment_text(raw_text)
            processing_results['text_segments'] = len(segments)
            
            # 2. Create text segments in database
            self._save_text_segments(document, segments)

            # 2.5. Generate embeddings for text segments
            try:
                if self._should_generate_embeddings(segments, document):
                    from .vector_processor import VectorProcessor
                    print(f"Generating embeddings for document: {document.filename}")
                    
                    vector_processor = VectorProcessor()
                    embedding_results = vector_processor.generate_embeddings_for_document(str(document.id))
                    
                    processing_results['embeddings'] = embedding_results['processed_segments']
                    processing_results['embeddings_generated'] = embedding_results['processed_segments']
                    processing_results['embedding_errors'] = embedding_results['failed_segments']
                    
                    if embedding_results['errors']:
                        processing_results['errors'].extend(embedding_results['errors'])
                    
                    print(f"SUCCESS: Generated embeddings for {embedding_results['processed_segments']} segments")
                else:
                    processing_results['embeddings'] = 0
                    processing_results['embeddings_generated'] = 0
                    processing_results['embedding_errors'] = 0
                    print(f"WARNING: Skipped embeddings for cost optimization")   

                    
            except Exception as e:
                error_msg = f"Embedding generation failed: {str(e)}"
                processing_results['errors'].append(error_msg)
                processing_results['embeddings'] = 0
                processing_results['embeddings_generated'] = 0
                print(f"ERROR: {error_msg}")
            

            # 3. Detect and extract tables
            tables = self._detect_tables(raw_text)
            processing_results['tables'] = len(tables)
            
            # 4. Save tables and structured data
            for table_data in tables:
                self._save_table_data(document, table_data)
            
            # 5. Extract key-value pairs
            key_values = self._extract_key_values(raw_text)
            processing_results['key_values_extracted'] = len(key_values)
            
            # 6. Save key-value pairs
            self._save_key_values(document, key_values)
            
            # 7. Update document status
            document.processing_status = 'completed'
            document.processing_completed_at = timezone.now()
            document.extraction_summary = json.dumps(processing_results)
            document.save()
            
        except Exception as e:
            processing_results['processing_status'] = 'failed'
            processing_results['errors'].append(str(e))
            
            document.processing_status = 'failed'
            document.processing_error = str(e)
            document.save()

        print(f"DEBUG: Returning processing_results = {processing_results}")
        return processing_results
    
    def _segment_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment text into logical chunks (paragraphs, headings, or tabular data)
        """
        segments = []
        self.sequence_counter = 0
        
        # Clean up the text (but preserve newlines)
        text = self._clean_text(text)
        
        # Detect if this is tabular data (CSV/XLSX)
        if self._is_tabular_data(text):
            segments = self._segment_tabular_data(text)
        else:
            segments = self._segment_regular_text(text)
        
        return segments

    def _is_tabular_data(self, text: str) -> bool:
        """Detect if text contains tabular data (CSV-like structure)"""
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent delimiter patterns across lines
        comma_lines = sum(1 for line in lines[:10] if ',' in line and len(line.split(',')) > 2)
        tab_lines = sum(1 for line in lines[:10] if '\t' in line and len(line.split('\t')) > 2)
        
        # If majority of first 10 lines have consistent delimiters, it's tabular
        return comma_lines >= 5 or tab_lines >= 5

    def _segment_tabular_data(self, text: str) -> List[Dict[str, Any]]:
        """Segment tabular data (CSV/XLSX) into AI-queryable text segments"""
        segments = []
        lines = text.strip().split('\n')
        
        if len(lines) < 2:
            return segments
        
        # Detect delimiter
        delimiter = ',' if ',' in lines[0] else '\t'
        
        # Get headers
        headers = [col.strip().strip('"') for col in lines[0].split(delimiter)]
        
        # Create a summary segment with all unique values from key columns
        key_columns = self._identify_key_columns(headers)
        if key_columns:
            summary_data = self._create_data_summary(lines[1:], headers, key_columns, delimiter)
            if summary_data:
                segments.append({
                    'sequence_number': self.sequence_counter,
                    'content': summary_data,
                    'segment_type': 'data_summary',
                    'section_title': f'Data Summary: {", ".join(key_columns)}',
                    'content_length': len(summary_data),
                    'word_count': len(summary_data.split())
                })
                self.sequence_counter += 1
        
        # Create segments from individual rows (limit to first 30 for performance)
        for i, line in enumerate(lines[1:31]):
            if line.strip():
                row_segment = self._create_row_segment(line, headers, delimiter, i)
                if row_segment:
                    segments.append({
                        'sequence_number': self.sequence_counter,
                        'content': row_segment,
                        'segment_type': 'data_row',
                        'section_title': None,
                        'content_length': len(row_segment),
                        'word_count': len(row_segment.split())
                    })
                    self.sequence_counter += 1
        
        return segments

    def _identify_key_columns(self, headers: List[str]) -> List[str]:
        """Identify columns that likely contain important searchable data"""
        key_patterns = [
            'name', 'model', 'title', 'product', 'company', 'brand', 
            'type', 'category', 'description', 'item', 'service'
        ]
        
        key_columns = []
        for header in headers:
            header_lower = header.lower()
            if any(pattern in header_lower for pattern in key_patterns):
                key_columns.append(header)
        
        # If no key columns found, use first few columns
        if not key_columns:
            key_columns = headers[:3]
        
        return key_columns

    def _create_data_summary(self, data_lines: List[str], headers: List[str], 
                            key_columns: List[str], delimiter: str) -> str:
        """Create a summary of unique values in key columns"""
        key_indices = [headers.index(col) for col in key_columns if col in headers]
        unique_values = {col: set() for col in key_columns}
        
        for line in data_lines[:100]:  # Process first 100 rows
            values = line.split(delimiter)
            for i, col in enumerate(key_columns):
                if i < len(key_indices) and key_indices[i] < len(values):
                    value = values[key_indices[i]].strip().strip('"')
                    if value and len(value) > 1:  # Skip empty/single char values
                        unique_values[col].add(value)
        
        # Create readable summary
        summary_parts = []
        for col, values in unique_values.items():
            if values:
                sorted_values = sorted(list(values))[:20]  # Top 20 unique values
                summary_parts.append(f"{col} includes: {', '.join(sorted_values)}")
        
        if summary_parts:
            return f"This dataset contains the following data: {'. '.join(summary_parts)}"
        
        return ""

    def _create_row_segment(self, line: str, headers: List[str], delimiter: str, row_num: int) -> str:
        """Create a readable text segment from a data row"""
        values = [val.strip().strip('"') for val in line.split(delimiter)]
        
        # Create human-readable description
        descriptions = []
        for i, value in enumerate(values):
            if i < len(headers) and value and len(value) > 1:
                descriptions.append(f"{headers[i]}: {value}")
        
        if descriptions:
            return f"Record {row_num + 1} - {', '.join(descriptions)}"
        
        return ""

    def _segment_regular_text(self, text: str) -> List[Dict[str, Any]]:
        """Segment regular text (PDF/Word) into paragraphs"""
        segments = []
        
        # Try multiple splitting strategies
        paragraphs = []
        
        # Strategy 1: Split by double newlines (preferred)
        if '\n\n' in text:
            paragraphs = re.split(r'\n\s*\n', text)
        # Strategy 2: Split by single newlines (fallback)
        elif '\n' in text:
            paragraphs = text.split('\n')
        # Strategy 3: Split by sentences (final fallback)
        else:
            paragraphs = re.split(r'[.!?]+\s*', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 50:
                continue

            # Skip very short segments
            if len(paragraph.split()) < 5:
                continue

            # Determine segment type
            segment_type = self._classify_segment(paragraph)
            
            # Extract section information if it's a heading
            section_title = None
            if segment_type in ['heading', 'title']:
                section_title = paragraph[:500]
            
            segment = {
                'sequence_number': self.sequence_counter,
                'content': paragraph,
                'segment_type': segment_type,
                'section_title': section_title,
                'content_length': len(paragraph),
                'word_count': len(paragraph.split())
            }
            
            segments.append(segment)
            self.sequence_counter += 1
        
        return segments
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive spaces and tabs, but KEEP newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces/tabs
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*of\s*\d+', '', text)
        
        # Keep line breaks but normalize them
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _classify_segment(self, text: str) -> str:
        """
        Classify text segment type based on content patterns
        
        Args:
            text: Text content to classify
            
        Returns:
            Segment type string
        """
        text_lower = text.lower().strip()
        
        # Check for headings (short, uppercase, or numbered)
        if len(text) < 100:
            if text.isupper() or re.match(r'^\d+\.?\s+[A-Z]', text):
                return 'heading'
            if re.match(r'^[A-Z][^.!?]*$', text) and len(text.split()) <= 10:
                return 'title'
        
        # Check for list items
        if re.match(r'^\s*[-•*]\s+', text) or re.match(r'^\s*\d+\.?\s+', text):
            return 'list_item'
        
        # Check for table captions
        if re.match(r'^(table|figure|chart)\s+\d+', text_lower):
            return 'table_caption'
        
        # Default to paragraph
        return 'paragraph'
    
    def _save_text_segments(self, document: Document, segments: List[Dict[str, Any]]):
        """Save text segments to database"""
        segment_objects = []
        
        for segment_data in segments:
            segment = DocumentTextSegment(
                document=document,
                sequence_number=segment_data['sequence_number'],
                content=segment_data['content'],
                segment_type=segment_data['segment_type'],
                section_title=segment_data.get('section_title'),
                content_length=segment_data['content_length'],
                word_count=segment_data['word_count']
            )
            segment_objects.append(segment)
        
        # Bulk create for efficiency
        DocumentTextSegment.objects.bulk_create(segment_objects)
    


    def _should_generate_embeddings(self, segments: List[Dict[str, Any]], document: Document) -> bool:
        """Determine if embeddings should be generated for this document"""
        from django.conf import settings
        
        # Check if embeddings are enabled
        if not getattr(settings, 'ENABLE_AUTO_EMBEDDINGS', True):
            return False
        
        # Skip if document has too many segments
        max_segments = getattr(settings, 'MAX_SEGMENTS_PER_DOCUMENT', 50)
        if len(segments) > max_segments:
            if getattr(settings, 'SKIP_LARGE_DOCUMENTS', True):
                print(f"⚠️ Skipping embeddings for {document.filename}: {len(segments)} segments > {max_segments} limit")
                return False
        
        # Skip certain file types for embeddings
        if document.file_type == 'xlsx' and len(segments) > 30:
            print(f"⚠️ Skipping embeddings for large Excel file: {document.filename}")
            return False
    
        return True
    

    def _detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect table-like structures in text
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of detected tables with metadata
        """
        tables = []
        
        if self._is_tabular_data(text):
            lines = text.strip().split('\n')
            if len(lines) > 1:
                delimiter = ',' if ',' in lines[0] else '\t'
                headers = lines[0].split(delimiter)
                
                tables.append({
                    'table_index': 0,           # Added this
                    'table_name': 'Data_Table', # Changed from 'title' to 'table_name'
                    'row_count': len(lines) - 1, # Changed from 'rows' to 'row_count'
                    'column_count': len(headers), # Changed from 'columns' to 'column_count'
                    'has_header': True,         # Added this
                    'table_data': [],           # Added this (empty for now)
                    'table_type': 'csv_data',
                    'content': text[:2000]
                })



        # Look for table patterns (rows with consistent delimiters)
        lines = text.split('\n')
        current_table = []
        table_index = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_table:
                    # End of table
                    table_data = self._parse_table_data(current_table)
                    if table_data:
                        tables.append({
                            'table_index': table_index,
                            'table_name': f'Table_{table_index + 1}',
                            'row_count': len(table_data),
                            'column_count': len(table_data[0]) if table_data else 0,
                            'table_data': table_data,
                            'has_header': True
                        })
                        table_index += 1
                    current_table = []
                continue
            
            # Check if line looks like a table row
            if self._is_table_row(line):
                current_table.append(line)
            else:
                if current_table and len(current_table) >= 2:
                    # Process accumulated table
                    table_data = self._parse_table_data(current_table)
                    if table_data:
                        tables.append({
                            'table_index': table_index,
                            'table_name': f'Table_{table_index + 1}',
                            'row_count': len(table_data),
                            'column_count': len(table_data[0]) if table_data else 0,
                            'table_data': table_data,
                            'has_header': True
                        })
                        table_index += 1
                current_table = []
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line appears to be part of a table"""
        # Look for multiple separators (tabs, pipes, multiple spaces)
        separators = ['\t', '|', '  ', ',']
        
        for sep in separators:
            if line.count(sep) >= 2:
                return True
        
        return False
    
    def _parse_table_data(self, table_lines: List[str]) -> List[List[str]]:
        """Parse table lines into structured data"""
        if not table_lines:
            return []
        
        # Try different delimiters
        delimiters = ['\t', '|', ',']
        best_delimiter = None
        max_columns = 0
        
        for delimiter in delimiters:
            test_row = table_lines[0].split(delimiter)
            if len(test_row) > max_columns:
                max_columns = len(test_row)
                best_delimiter = delimiter
        
        if not best_delimiter or max_columns < 2:
            # Try splitting on multiple spaces
            test_row = re.split(r'\s{2,}', table_lines[0])
            if len(test_row) >= 2:
                table_data = []
                for line in table_lines:
                    row = [cell.strip() for cell in re.split(r'\s{2,}', line)]
                    if row:
                        table_data.append(row)
                return table_data
            return []
        
        # Parse with best delimiter
        table_data = []
        for line in table_lines:
            row = [cell.strip() for cell in line.split(best_delimiter)]
            if row:
                table_data.append(row)
        
        return table_data
    
    def _save_table_data(self, document: Document, table_info: Dict[str, Any]):
        """Save table and its structured data to database"""
        # Create table record
        table = DocumentTable.objects.create(
            document=document,
            table_name=table_info['table_name'],
            table_index=table_info['table_index'],
            row_count=table_info['row_count'],
            column_count=table_info['column_count'],
            has_header=table_info['has_header'],
            table_data_json=json.dumps(table_info['table_data'])
        )
        
        # Create structured data records for each cell
        table_data = table_info['table_data']
        if not table_data:
            return
        
        # Determine column names (use first row if it's a header)
        column_names = []
        data_start_row = 0
        
        if table_info['has_header'] and len(table_data) > 1:
            column_names = table_data[0]
            data_start_row = 1
        else:
            column_names = [f'Column_{i+1}' for i in range(len(table_data[0]))]
        
        # Create cell records
        cell_objects = []
        for row_idx, row in enumerate(table_data[data_start_row:], start=data_start_row):
            for col_idx, cell_value in enumerate(row):
                if col_idx >= len(column_names):
                    continue
                
                # Determine data type and convert value
                typed_values = self._convert_cell_value(cell_value)
                
                cell = DocumentStructuredData(
                    document=document,
                    table=table,
                    row_number=row_idx,
                    column_number=col_idx,
                    column_name=column_names[col_idx] if col_idx < len(column_names) else f'Column_{col_idx+1}',
                    cell_value=cell_value,
                    **typed_values
                )
                cell_objects.append(cell)
        
        # Bulk create cells
        DocumentStructuredData.objects.bulk_create(cell_objects)
    
    def _convert_cell_value(self, value: str) -> Dict[str, Any]:
        """Convert cell value to appropriate data types"""
        value = value.strip()
        result = {'text_value': value}
        
        if not value:
            result['data_type'] = 'text'
            return result
        
        # Try to convert to number
        try:
            # Remove common formatting
            clean_value = re.sub(r'[,$%]', '', value)
            
            if '.' in clean_value:
                numeric_value = float(clean_value)
                result['numeric_value'] = numeric_value
                result['data_type'] = 'decimal'
            else:
                integer_value = int(clean_value)
                result['integer_value'] = integer_value
                result['numeric_value'] = float(integer_value)
                result['data_type'] = 'integer'
                
        except ValueError:
            # Try to parse as date
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            
            for pattern in date_patterns:
                if re.match(pattern, value):
                    result['data_type'] = 'date'
                    break
            else:
                # Check for boolean
                if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
                    result['boolean_value'] = value.lower() in ['true', 'yes', '1']
                    result['data_type'] = 'boolean'
                else:
                    result['data_type'] = 'text'
        
        return result
    
    def _extract_key_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract key-value pairs from text"""
        key_values = []
        
        # Pattern for key-value pairs (e.g., "Total Amount: $1,500")
        patterns = [
            r'([A-Za-z\s]+):\s*([^\n]+)',  # Key: Value
            r'([A-Za-z\s]+)\s*=\s*([^\n]+)',  # Key = Value
            r'([A-Za-z\s]+)\s*-\s*([^\n]+)',  # Key - Value
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Skip if key is too short or value is empty
                if len(key) < 3 or not value:
                    continue
                
                # Convert value to appropriate type
                typed_values = self._convert_cell_value(value)
                
                key_value = {
                    'key_name': key,
                    'value_text': value,
                    'extraction_method': 'regex_pattern',
                    **typed_values
                }
                
                key_values.append(key_value)
        
        return key_values
    
    def _save_key_values(self, document: Document, key_values: List[Dict[str, Any]]):
        """Save key-value pairs to database"""
        kv_objects = []
        
        for kv_data in key_values:
            kv = DocumentKeyValue(
                document=document,
                key_name=kv_data['key_name'],
                value_text=kv_data['value_text'],
                extraction_method=kv_data['extraction_method'],
                data_type=kv_data.get('data_type', 'text')
            )
            
            # Set typed values
            if 'numeric_value' in kv_data:
                kv.value_numeric = kv_data['numeric_value']
            if 'boolean_value' in kv_data:
                kv.value_boolean = kv_data['boolean_value']
                
            kv_objects.append(kv)
        
        # Bulk create
        DocumentKeyValue.objects.bulk_create(kv_objects)