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
            'segments_created': 0,
            'tables_detected': 0,
            'key_values_extracted': 0,
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
            processing_results['segments_created'] = len(segments)
            
            # 2. Create text segments in database
            self._save_text_segments(document, segments)

            # 2.5. Generate embeddings for text segments
            try:
                from .vector_processor import VectorProcessor
                print(f"Generating embeddings for document: {document.filename}")
                
                vector_processor = VectorProcessor()
                embedding_results = vector_processor.generate_embeddings_for_document(str(document.id))
                
                processing_results['embeddings_generated'] = embedding_results['processed_segments']
                processing_results['embedding_errors'] = embedding_results['failed_segments']
                
                if embedding_results['errors']:
                    processing_results['errors'].extend(embedding_results['errors'])
                
                print(f"✅ Generated embeddings for {embedding_results['processed_segments']} segments")
                    
            except Exception as e:
                error_msg = f"Embedding generation failed: {str(e)}"
                processing_results['errors'].append(error_msg)
                processing_results['embeddings_generated'] = 0
                print(f"❌ {error_msg}")
            

            # 3. Detect and extract tables
            tables = self._detect_tables(raw_text)
            processing_results['tables_detected'] = len(tables)
            
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
        
        return processing_results
    
    def _segment_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment text into logical chunks (paragraphs, headings, etc.)
        
        Args:
            text: Raw text to segment
            
        Returns:
            List of text segments with metadata
        """
        segments = []
        self.sequence_counter = 0
        
        # Clean up the text
        text = self._clean_text(text)
        
        # Split by multiple newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Determine segment type
            segment_type = self._classify_segment(paragraph)
            
            # Extract section information if it's a heading
            section_title = None
            if segment_type in ['heading', 'title']:
                section_title = paragraph[:500]  # Truncate long titles
            
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
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*of\s*\d+', '', text)
        
        # Normalize line breaks
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
    
    def _detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect table-like structures in text
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of detected tables with metadata
        """
        tables = []
        
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