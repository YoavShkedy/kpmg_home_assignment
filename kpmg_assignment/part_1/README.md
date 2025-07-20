# Israeli National Insurance Form Extractor

A Streamlit web application that extracts information from Israeli National Insurance Institute (ביטוח לאומי) forms using Azure Document Intelligence for OCR and Azure OpenAI for intelligent field extraction.

## Features

- **Multi-format Support**: Process PDF, JPG, JPEG, PNG, and TIFF files
- **Bilingual Processing**: Handles both Hebrew and English text
- **Intelligent Field Extraction**: Uses Azure OpenAI to extract structured information
- **Data Validation**: Comprehensive validation with warnings for data quality issues
- **User-friendly Interface**: Clean Streamlit web interface
- **JSON Export**: Download extracted data in JSON format

## Prerequisites

- Python 3.8 or higher
- Azure Document Intelligence service
- Azure OpenAI service with GPT-4 deployment

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd part_1
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the `part_1` directory with the following variables:

   ```env
   # Azure Document Intelligence
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-document-intelligence.cognitiveservices.azure.com/
   AZURE_DOCUMENT_INTELLIGENCE_KEY=your_document_intelligence_key

   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   ```

2. **Azure Service Setup**:
   
   **Document Intelligence**:
   - Create an Azure Document Intelligence resource
   - Copy the endpoint and key to your `.env` file
   
   **Azure OpenAI**:
   - Create an Azure OpenAI resource
   - Deploy a GPT-4 model (deployment name: `gpt-4o`)
   - Copy the endpoint and key to your `.env` file

## Running the Application

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically open in your default browser

## Usage Guide

### Processing a Document

1. **Upload a Document**:
   - Click "Choose a document file" in the left panel
   - Select a PDF or image file containing an Israeli National Insurance form
   - Supported formats: PDF, JPG, JPEG, PNG, TIFF

2. **Process the Document**:
   - Click "🚀 Process Document" button
   - Wait for the processing to complete (typically 10-30 seconds)

3. **Review Results**:
   - View extracted fields in the results panel
   - Check validation warnings if any
   - Download the results as JSON

### Extracted Fields

The application extracts the following information:

- **Personal Information**: Name, ID number, gender, date of birth
- **Contact Information**: Address, landline phone, mobile phone
- **Accident Details**: Date/time of injury, location, description, injured body part
- **Employment**: Job type
- **Medical Information**: Health fund membership, nature of accident, medical diagnoses
- **Form Metadata**: Signature, filing dates

### Validation Features

The application performs comprehensive validation:

- **ID Number**: Validates 9-digit Israeli ID format
- **Dates**: Validates day (1-31), month (1-12), and year format
- **Phone Numbers**: Validates phone number format and length
- **Data Types**: Ensures appropriate data types for each field
- **Medical Codes**: Validates 4-character alphanumeric medical diagnosis codes

## File Structure

```
part_1/
├── app.py                 # Main Streamlit application
├── field_extraction.py    # Field extraction logic using Azure OpenAI
├── ocr.py                # OCR processing using Azure Document Intelligence
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .env                 # Environment variables (create this file)
```

## Troubleshooting

### Common Issues

1. **"Azure client setup failed"**:
   - Check that all environment variables are set correctly in `.env`
   - Verify Azure service endpoints and keys
   - Ensure Azure services are deployed and accessible

2. **"OCR processing failed"**:
   - Verify Document Intelligence service is running
   - Check file format is supported (PDF, JPG, JPEG, PNG, TIFF)
   - Ensure file is not corrupted

3. **"Field extraction failed"**:
   - Verify Azure OpenAI service is deployed
   - Check that GPT-4 model is deployed with name `gpt-4o`
   - Ensure API quota is not exceeded

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

### Environment Variables

Make sure your `.env` file contains all required variables:

```bash
# Check if environment variables are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Doc Intelligence:', bool(os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'))); print('OpenAI:', bool(os.getenv('AZURE_OPENAI_API_KEY')))"
```

### Testing Individual Components

**Test OCR only**:
```bash
python ocr.py
```

**Test Field Extraction only**:
```bash
python field_extraction.py
```

## Development

### Code Structure

- **`app.py`**: Main Streamlit application with UI components
- **`ocr.py`**: `DocumentOCRProcessor` class for Azure Document Intelligence
- **`field_extraction.py`**: `FieldExtractor` class with Pydantic models for structured output

### Adding New Fields

1. Update the Pydantic models in `field_extraction.py`
2. Modify the system prompt to include extraction instructions
3. Add validation logic if needed

### Customizing Validation

Validation rules are defined in the `_validate_and_clean_data` method and related helper methods in `field_extraction.py`.

## License

This project is part of a KPMG assignment and is intended for demonstration purposes.

## Support

For issues related to:
- **Azure Services**: Check Azure portal for service status
- **Application Bugs**: Review error logs in the Streamlit interface
- **Feature Requests**: Consider the application's scope and requirements 