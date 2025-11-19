import json
import requests

GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
prompt = """
You are an expert Ethiopian National ID card verification system. 
Compare OCR extracted data with user-provided data and make a verification decision.

 => OCR EXTRACTED DATA

CRITICAL VALIDATION RULES:
1. Extracted and user provided FCN must have almost similar digits
2. Identify Ethiopian Calendar dates (EC) and Gregorian Calendar dates (GC)
3. Check if main fields (Name, DOB, Expiry, FCN) are complete and readable

DOCUMENT STRUCTURE:
- Top: Amharic text (may be garbled in OCR) 
- "Ethiopian Digital ID Card" header
- Name section: Amharic above, English below (First, Middle, Surname)
- Date of Birth: Ethiopian date (EC) above, Gregorian date (GC) below
- Expiry Date & Sex: Labels may be misaligned - Ethiopian expiry date first, then Sex value, then Gregorian expiry date
- Citizenship: Abbreviated as "ET" for Ethiopia
- Rigion: Addis ababa or Amhara or Oromia ...
- FCN: must be Digit 


EXTRACTION FIELDS:
- document_type
- full_name_english
- date_of_birth_gc (Gregorian Calendar - DD/MM/YYYY) 
- expiry_date_gc (Gregorian Calendar - DD/MM/YYYY)
- gender
- citizenship
- Rigion
- fcn_number 
- extraction_quality (rating 1-10)
- validation_errors (array of errors)

=>  COMPARISION RULES
FIELD MATCHING RULES:
- FCN: Almost EXACT match required
- Name: Flexible matching (ignore case, minor spelling variations)
- Date: Flexible format matching (DD/MM/YYYY vs YYYY-MM-DD)
- Region: Flexible matching (Addis Ababa = Addisababa = Addis Abeba)
- Gendar:exactly similar

VERIFICATION CRITERIA:
✅ APPROVED: All critical fields match (Name, FCN, DOB, Region)
⚠️ REVIEW: only  clear sespeceus thing and  need human verification otherwise make it rejected eg  gendar at the user data is male and in the extracted data but Female or region in the id addis ababa but in user data Amhara
❌ REJECTED: Critical field mismatch (FCN, Name) , complatly name d/ce or multiple errors or if the extracted data is not complated and have no clear and full info


RESPONSE FORMAT - RETURN ONLY THIS JSON:

{
  "status": "success",
  "verification_decision": "approved|review|rejected",
  "confidence_score": 0.95,
  "comparison_passed": true,
  "note": "Brief reason for decision",
  "extracted_data": {
    "document_type": "Ethiopian National ID Card",
    "full_name_english": "Extracted name here",
    "date_of_birth_gc": "DD/MM/YYYY",
    "expiry_date_gc": "DD/MM/YYYY", 
    "gender": "Male/Female",
    "citizenship": "ET",
    "region": "Extracted region",
    "fcn_number": "digit number",
    "extraction_quality": 8,
    "validation_errors": []
  },
  "user_provided_data": {
    "full_name_english": "User provided name",
    "date_of_birth_gc": "DD/MM/YYYY",
    "gender": "Male/Female", 
    "region": "User provided region",
    "fcn_number": "digit number"
  },
  "field_comparisons": {
    "fcn_match": true,
    "name_match": true,
    "dob_match": true,
    "gender_match": true,
    "region_match": true
  }
}


Return ONLY the JSON above, no other text.
ANALYSIS GUIDELINES:
- Ethiopian dates may have garbled month names (like "90fAn")
- Name should be in "First Middle Surname" format
- FCN validation is MANDATORY -
- Rate quality based on field completeness and readability
- Note any OCR artifacts or misalignments

Return ONLY JSON, no other text.         

The Context ocr raw text  and user true data are :- """


class IDExtractor:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }

    def _call_model(self, raw_text: str, user_data: str, model: str = "openai/gpt-4.1"):
        global prompt
        full_prompt = f"""
                 {prompt}
           OCR RAW TEXT:
                   {raw_text}
    
                      USER PROVIDED DATA:
               {user_data}
    
                    Compare the data and return ONLY the verification decision JSON.
                      """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": full_prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        resp = requests.post(
            GITHUB_MODELS_URL, headers=self.headers, json=payload, timeout=60
        )
        # raise for network/4xx/5xx
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # attach body for debugging (but avoid leaking token)
            raise RuntimeError(
                f"Model API error {resp.status_code}: {resp.text}"
            ) from e

        return resp.json()

    def extract_to_json(
        self, extracted_text: str, user_data: str, model: str = "openai/gpt-4.1"
    ) -> str:
        """
        Calls GitHub Models and returns the assistant content (expected to be JSON).
        """
        raw = self._call_model(extracted_text, user_data, model=model)
        
        
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by model: " + json.dumps(raw))
        # GitHub's response uses message.content inside choices[*].message.content
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("content")
        if not content:
            # fallback: stringify raw response for debugging
            raise RuntimeError(
                "No message content found. Raw response: " + json.dumps(raw)
            )
        # Ensure the model returned JSON — try to parse it; return minified JSON string
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # If model returned text that isn't strict JSON, still return content
            return content
