from django.db import models

# model for uploading files 
class FileUpload(models.Model): #choice for file types 
    FILE_TYPES = [
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
        ('xlsx', 'Excel Sheet'),
    ]

    filename = models.CharField(max_length=255) #File name
    file = models.FileField(upload_to='uploads/')# The actual file, stored in the 'uploads/' subdirectory inside /media
    type = models.CharField(max_length=10, choices=FILE_TYPES) #type of file
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):   # String representation of the object (for admin panel and logs)
        return self.filename