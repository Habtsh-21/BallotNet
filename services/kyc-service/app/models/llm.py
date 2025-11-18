
import json
import requests

GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
prompt = '''
You are an expert Ethiopian National ID card analyzer. Extract and validate information from the raw OCR text.

CRITICAL VALIDATION RULES:
1. FCN (Fiyda Card Number) MUST be exactly 12 digits. If not, return error JSON.
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
- FCN: 12-digit Family Card Number (MUST BE VALID)

EXTRACTION FIELDS:
- document_type
- full_name_english
- date_of_birth_gc (Gregorian Calendar - DD/MM/YYYY) 
- expiry_date_gc (Gregorian Calendar - DD/MM/YYYY)
- gender
- citizenship
- Rigion
- fcn_number (12 digits)
- extraction_quality (rating 1-10)
- validation_errors (array of errors)

RESPONSE FORMATS:

SUCCESS JSON:
{
  "status": "success",
  "data": {
    "document_type": "Ethiopian Digital ID Card",
    "full_name_english": "Test Test Test",
    "date_of_birth_gc": "01/01/1900",
    "expiry_date_gc": "01/01/1900", 
    "gender": "Male",
    "citizenship": "ET",
    "Rigion": "Addis Ababa"
    "fcn_number": "285473403197",
    "extraction_quality": 7,
    "notes": ["Name appears valid", "Dates need verification", "FCN length issue"]
  }
}

ERROR JSON ( critical data missing):
{
  "status": "error",
  "error_code": "short message",
  "message": "message",
  "validation_errors": [
  ],
}

ANALYSIS GUIDELINES:
- Ethiopian dates may have garbled month names (like "90fAn")
- Name should be in "First Middle Surname" format
- FCN validation is MANDATORY -
- Rate quality based on field completeness and readability
- Note any OCR artifacts or misalignments

Return ONLY JSON, no other text.         

The Context ocr raw text is: '''


class IDExtractor:
    def __init__(self, token: str):
        self.token = token 
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }
 
    def _call_model(self, raw_text: str, model: str = "openai/gpt-4.1"):
        global prompt   
        full_prompt = prompt + raw_text
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": ""

          },
                {"role": "user", "content": full_prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        resp = requests.post(GITHUB_MODELS_URL, headers=self.headers, json=payload, timeout=60)
        # raise for network/4xx/5xx
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # attach body for debugging (but avoid leaking token)
            raise RuntimeError(f"Model API error {resp.status_code}: {resp.text}") from e

        return resp.json()



    def extract_to_json(self, extracted_text: str, model: str = "openai/gpt-4.1") -> str:
        """
        Calls GitHub Models and returns the assistant content (expected to be JSON).
        """
        raw = self._call_model(extracted_text, model= model)
        # defensive navigation of response structure
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by model: " + json.dumps(raw))
        # GitHub's response uses message.content inside choices[*].message.content
        message = choices[0].get("message") or {}
        content = message.get("content") or choices[0].get("content")
        if not content:
            # fallback: stringify raw response for debugging
            raise RuntimeError("No message content found. Raw response: " + json.dumps(raw))
        # Ensure the model returned JSON — try to parse it; return minified JSON string
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            # If model returned text that isn't strict JSON, still return content
            return content
