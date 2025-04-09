import PyPDF2
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
import logging


class PDFText():
    def __init__(self):
        pass

    def _get_editable_pdf_data(self, file: bytes):
        """
        Process an editable PDF and extract text page by page.
        
        Args:
            file (bytes): pdf bytes
        
        Returns:
            Dict[str, str]: dictionary with page numbers as keys and text as values
        """
        result = {}

        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text() or ""
            result[f"page_{page_num + 1}"] = text.strip()

        return result
    
    def _get_editable_pdf_data_from_path(self, file_path: str):
        """
        Process an editable PDF and extract text page by page.
        
        Args:
            file (str): pdf path
        
        Returns:
            Dict[str, str]: dictionary with page numbers as keys and text as values
        """
        result = {}

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
        
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text() or ""
                result[f"page_{page_num + 1}"] = text.strip()

        return result
    
    def _get_scanned_pdf_data(self, file: bytes):
        """
        Process a scanned PDF and extract text page by page using OCR.
        
        Args:
            file (bytes): pdf bytes
        
        Returns:
            Dict[str, str]: dictionary with page numbers as keys and text as values
        """
        result = {}

        # Convert PDF bytes to images
        images = convert_from_bytes(file)
        
        # Process each page image with OCR
        for page_num, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image)
            result[f"page_{page_num}"] = text.strip()
            
        return result
    
    def _get_scanned_pdf_data_from_path(self, file_path: str):
        """
        Process a scanned PDF and extract text page by page using OCR.
        
        Args:
            file_path (str): pdf path
        
        Returns:
            Dict[str, str]: dictionary with page numbers as keys and text as values
        """
        result = {}
        
        # Convert PDF pages to images
        images = convert_from_path(file_path)
        
        # Process each page image with OCR
        for page_num, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image)
            result[f"page_{page_num}"] = text.strip()
            
        return result

    def get_pdf_text(self, file: bytes=None, file_path: str=None):
        """
        Extract text from pdf file. Handles both editable and scanned pdfs.

        Args:
            file (bytes): pdf in bytes. optional.
            file_path (str): path to the PDF file. optional.
        
        Returns:
            Dict: JSON-compatible dictionary with page-wise text
        """
        try:
            if file:
                try:
                    # try with editable way
                    text_data = self._get_editable_pdf_data(file)

                    # If no text is extracted, assume it's a scanned PDF
                    if not any(page_text.strip() for page_text in result.values()):
                        raise Exception("Scanned PDF")
                except:
                    # proceed with ocr extraction
                    text_data = self._get_scanned_pdf_data(file)

                return text_data
            elif file_path:
                try:
                    # try with editable way
                    text_data = self._get_editable_pdf_data_from_path(file_path)

                    # If no text is extracted, assume it's a scanned PDF
                    if not any(page_text.strip() for page_text in result.values()):
                        raise Exception("Scanned PDF")
                except:
                    # proceed with ocr extraction
                    text_data = self._get_scanned_pdf_data_from_path(file_path)

                return text_data
            else:
                raise Exception(f"Invalaid arguments. At least one of file or file_path needs to be provided")
        except Exception as e:
            logging.error(f"Exception while getting pdf text data: {e}")
