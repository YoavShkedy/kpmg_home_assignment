import json
import re
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import logging
from ocr import DocumentOCRProcessor
from dotenv import load_dotenv
import os

# Pydantic models for structured output
class DateInfo(BaseModel):
    """Date information with day, month, year"""
    day: str = Field(default="", description="Day of the date / יום")
    month: str = Field(default="", description="Month of the date / חודש")
    year: str = Field(default="", description="Year of the date / שנה")

class Address(BaseModel):
    """Address information"""
    street: str = Field(default="", description="Street name / רחוב")
    houseNumber: str = Field(default="", description="House number / מספר בית")
    entrance: str = Field(default="", description="Entrance number / כניסה")
    apartment: str = Field(default="", description="Apartment number / דירה")
    city: str = Field(default="", description="City name / ישוב")
    postalCode: str = Field(default="", description="Postal code / מיקוד")
    poBox: str = Field(default="", description="PO Box number / תא דואר")

class MedicalInstitutionFields(BaseModel):
    """Medical institution specific fields"""
    healthFundMember: str = Field(default="", description="Health fund membership status / חבר בקופת חולים")
    natureOfAccident: str = Field(default="", description="Nature of the accident / מהות התאונה")
    medicalDiagnoses: str = Field(default="", description="Medical diagnoses / אבחנות רפואיות")

class ExtractedFields(BaseModel):
    """Complete schema for extracted form fields"""
    lastName: str = Field(default="", description="Last name / שם משפחה")
    firstName: str = Field(default="", description="First name / שם פרטי")
    idNumber: str = Field(default="", description="ID number / מספר זהות")
    gender: str = Field(default="", description="Gender / מין")
    dateOfBirth: DateInfo = Field(default_factory=DateInfo, description="Date of birth / תאריך לידה")
    address: Address = Field(default_factory=Address, description="Address / כתובת")
    landlinePhone: str = Field(default="", description="Landline phone / טלפון קווי")
    mobilePhone: str = Field(default="", description="Mobile phone / טלפון נייד")
    jobType: str = Field(default="", description="Job type / סוג העבודה")
    dateOfInjury: DateInfo = Field(default_factory=DateInfo, description="Date of injury / תאריך הפגיעה")
    timeOfInjury: str = Field(default="", description="Time of injury / שעת הפגיעה")
    accidentLocation: str = Field(default="", description="Accident location / מקום התאונה")
    accidentAddress: str = Field(default="", description="Accident address / כתובת מקום התאונה")
    accidentDescription: str = Field(default="", description="Accident description / תיאור התאונה")
    injuredBodyPart: str = Field(default="", description="Injured body part / האיבר שנפגע")
    signature: str = Field(default="", description="Signature / חתימה")
    formFillingDate: DateInfo = Field(default_factory=DateInfo, description="Form filling date / תאריך מילוי הטופס")
    formReceiptDateAtClinic: DateInfo = Field(default_factory=DateInfo, description="Form receipt date at clinic / תאריך קבלת הטופס בקופה")
    medicalInstitutionFields: MedicalInstitutionFields = Field(default_factory=MedicalInstitutionFields, description="Medical institution fields / למילוי ע\"י המוסד הרפואי")

class FieldExtractor:
    """
    Field extractor using Azure OpenAI for Israeli National Insurance forms
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        """
        Initialize the field extractor
        
        Args:
            llm: Configured AzureChatOpenAI instance
        """
        self.llm = llm
        
        # Setup logging with reduced verbosity
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger('azure').setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Define the expected JSON schema
        self.target_schema = ExtractedFields().model_dump()
        
        # Initialize validation warnings list
        self.validation_warnings = []
    
    def extract_fields(self, ocr_text: str) -> Dict[str, Any]:
        """
        Extract fields from OCR text according to the schema
        
        Args:
            ocr_text: Text extracted from the document
            
        Returns:
            Dictionary containing extracted fields and metadata
        """
        try:
            # Clear validation warnings for new extraction
            self.validation_warnings = []
            
            # Create LLM with structured output
            structured_llm = self.llm.with_structured_output(ExtractedFields)
            
            system_prompt = """# Identity
            
You are an expert at extracting information from Israeli National Insurance Institute (ביטוח לאומי) forms.

# Instructions

Your task is to extract specific fields from the OCR text and return them in valid JSON format. Follow the guidelines below:

<guidelines>
1. Extract information accurately from both Hebrew and English text
2. For any field that is not present or cannot be extracted, use an empty string ""
3. For Signature field, extract the text that is written right under 'חתימה X' or 'חתימהX'. If signature is not present, then leave it empty. \
    For example, if the text is 'חתימה X\nJames', then the signature should be 'James'. \
    If the text is 'חתימה X\n5 למילוי ע״י המוסד הרפואי :selected:', then signature does not appear, and the field should be left empty. \
    Avoid inferring the signature from the text if it is not explicitly present. \
4. For natureOfAccident and medicalDiagnoses fields, extract the text that is written under 'מהות התאונה (אבחנות רפואיות):', only if that field is selected in the form (:selected: מהות התאונה (אבחנות רפואיות):). \
    The value of each of the fields should be an 4-character alphanumeric medical diagnosis code (e.g., ICD code). \
    For example, if the text is ':selected: מהות התאונה (אבחנות רפואיות):\n1234\n5678', then the natureOfAccident field should be '1234' and the medicalDiagnoses field should be '5678'. \
5. Return ONLY valid JSON that matches the specified schema
</guidelines>"""

            user_prompt = f"<ocr_text>\n{ocr_text}\n</ocr_text>"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get structured response
            result = structured_llm.invoke(messages)
            
            # Convert result to dict
            extracted_dict = getattr(result, 'model_dump', lambda: result if isinstance(result, dict) else {})()
            
            # Validate and clean data
            cleaned_data = self._validate_and_clean_data(extracted_dict)
            
            # Return structured response
            return {
                "success": True,
                "extracted_fields": cleaned_data,
                "validation_warnings": self.validation_warnings,
                "error": None
            }
            
        except Exception as e:
            self.logger.error(f"Error during structured field extraction: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_fields": self.target_schema.copy(),
                "validation_warnings": []
            }
    
    
    # --- Validation and cleaning functions ---

    def _validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data
        
        Args:
            data: Raw extracted data
            
        Returns:
            Validated data 
        """

        # Perform validations on the data
        self._validate_id_number(data)
        self._validate_dates(data)
        data = self._validate_phone_numbers(data)
        self._validate_string_fields(data)
        self._validate_numeric_fields(data)
        self._validate_health_fund_member(data)
        self._validate_gender(data)
        self._validate_medical_diagnosis_codes(data)
        
        return data
    
    def _add_validation_warning(self, field: str, message: str, value: str = ""):
        """
        Add a validation warning to the collection
        
        Args:
            field: Field name that has the validation issue
            message: Description of the validation issue
            value: The problematic value (optional)
        """
        warning = {
            "field": field,
            "message": message,
            "value": value
        }
        self.validation_warnings.append(warning)
    
    def _validate_id_number(self, data: Dict[str, Any]):
        """
        Validate Israeli ID number format
        """
        id_number = data.get("idNumber", "")
        if id_number and not re.match(r'^\d{9}$', id_number):
            self.logger.warning(f"Invalid ID number format: {id_number}")
            self._add_validation_warning("idNumber", "Invalid ID number format (should be 9 digits)", id_number)
    
    def _validate_dates(self, data: Dict[str, Any]):
        """
        Validate date fields
        """
        date_fields = ["dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic"]
        for field in date_fields:
            if field in data and isinstance(data[field], dict):
                day = data[field].get("day", "")
                month = data[field].get("month", "")
                year = data[field].get("year", "")
                
                # Basic validation
                if day and not (1 <= int(day) <= 31 if day.isdigit() else False):
                    self.logger.warning(f"Invalid day in {field}: {day}")
                    self._add_validation_warning(f"{field}.day", "Invalid day (should be 1-31)", day)
                if month and not (1 <= int(month) <= 12 if month.isdigit() else False):
                    self.logger.warning(f"Invalid month in {field}: {month}")
                    self._add_validation_warning(f"{field}.month", "Invalid month (should be 1-12)", month)
                if year and len(year) != 4:
                    self.logger.warning(f"Invalid year format in {field}: {year}")
                    self._add_validation_warning(f"{field}.year", "Invalid year format (should be 4 digits)", year)
    
    def _validate_phone_numbers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate phone number formats

        Args:
            data: Raw extracted data

        Returns:
            None
        """
        phone_fields = ["landlinePhone", "mobilePhone"]
        for field in phone_fields:
            phone = data.get(field, "")
            if phone:
                # If phone number is only one character, clear it
                if len(phone) == 1:
                    data[field] = "" # Clear the field
                    self._add_validation_warning(field, "Phone number too short (cleared)", phone)
                # Validate phone number format
                elif not re.match(r'^[\d\-\s\+\(\)]+$', phone):
                    self.logger.warning(f"Invalid phone number format in {field}: {phone}")
                    self._add_validation_warning(field, "Invalid phone number format", phone)
        return data
    
    def _validate_string_fields(self, data: Dict[str, Any]):
        """
        Validate string fields to ensure they contain only valid string characters (no digits)
        """
        # Direct string fields that should not contain digits
        string_fields = [
            "lastName", "firstName", "jobType", 
            "accidentLocation", "accidentDescription",
            "injuredBodyPart", "signature"
        ]
        
        for field in string_fields:
            value = data.get(field, "")
            if value and re.search(r'\d', str(value)):
                self.logger.warning(f"Field {field} should not contain digits but got: {value}")
                self._add_validation_warning(field, "Field should not contain digits", value)
        
        # Address string fields that should not contain digits
        address = data.get("address", {})
        if isinstance(address, dict):
            address_string_fields = ["street", "city"]
            for field in address_string_fields:
                value = address.get(field, "")
                if value and re.search(r'\d', str(value)):
                    self.logger.warning(f"Address field {field} should not contain digits but got: {value}")
                    self._add_validation_warning(f"address.{field}", "Address field should not contain digits", value)
    
    def _validate_numeric_fields(self, data: Dict[str, Any]):
        """
        Validate numeric fields to ensure they contain only digits
        """
        address = data.get("address", {})
        if isinstance(address, dict):
            numeric_fields = ["houseNumber", "postalCode"] # Remaining fields are validated in _validate_string_fields
            for field in numeric_fields:
                value = address.get(field, "")
                if value and not re.match(r'^\d+$', str(value)):
                    self.logger.warning(f"Address field {field} should contain only digits but got: {value}")
                    self._add_validation_warning(f"address.{field}", "Address field should contain only digits", value)
    
    def _validate_health_fund_member(self, data: Dict[str, Any]):
        """
        Validate healthFundMember field to ensure it's one of the allowed values
        """
        medical_fields = data.get("medicalInstitutionFields", {})
        if isinstance(medical_fields, dict):
            health_fund = medical_fields.get("healthFundMember", "")
            valid_health_funds = ["כללית", "מאוחדת", "מכבי", "לאומית", ""]
            
            if health_fund and health_fund not in valid_health_funds:
                self.logger.warning(f"healthFundMember should be one of {valid_health_funds} but got: {health_fund}")
                self._add_validation_warning("medicalInstitutionFields.healthFundMember", f"Should be one of: {', '.join(valid_health_funds[:-1])}", health_fund)
    
    def _validate_gender(self, data: Dict[str, Any]):
        """
        Validate gender field to ensure it's one of the allowed values
        """
        gender = data.get("gender", "")
        valid_genders = ["זכר", "נקבה", ""]
        
        if gender and gender not in valid_genders:
            self.logger.warning(f"gender should be one of {valid_genders} but got: {gender}")
            self._add_validation_warning("gender", f"Should be one of: {', '.join(valid_genders[:-1])}", gender)
    
    def _validate_medical_diagnosis_codes(self, data: Dict[str, Any]):
        """
        Validate natureOfAccident and medicalDiagnoses to ensure they are 4-character alphanumeric
        """
        medical_fields = data.get("medicalInstitutionFields", {})
        if isinstance(medical_fields, dict):
            medical_code_fields = ["natureOfAccident", "medicalDiagnoses"]
            for field in medical_code_fields:
                value = medical_fields.get(field, "")
                if value and not re.match(r'^[A-Za-z0-9]{4}$', str(value)):
                    self.logger.warning(f"Medical field {field} should be a 4-character alphanumeric code but got: {value}")
                    self._add_validation_warning(f"medicalInstitutionFields.{field}", "Should be a 4-character alphanumeric code", value)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Get environment variables with defaults
    doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not doc_intelligence_endpoint or not doc_intelligence_key:
        raise ValueError("Missing required environment variables: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY")

    # Create a client for the Document Intelligence service
    ocr = DocumentOCRProcessor(
        endpoint=doc_intelligence_endpoint,
        api_key=doc_intelligence_key
    )

    # Extract text from the file
    result = ocr.extract_from_file_path("phase1_data/283_ex1.pdf")

    # Create a field extractor
    field_extractor = FieldExtractor(
        llm=AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=0
        )
    )

    # Extract fields from the OCR text using the chain
    fields = field_extractor.extract_fields(result["extracted_text"])
    
    # Write OCR text to a text file
    with open("part_1/ocr_test_result.txt", "w", encoding="utf-8") as f:
        f.write(result["extracted_text"])

    # Write output fields to a text file
    with open("part_1/field_extraction_test_result.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(fields, indent=2, ensure_ascii=False))
    