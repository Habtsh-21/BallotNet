
import json
import requests

GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
prompt = '''
        
        you are an assistant that extracts key information from Ethiopian National ID cards based on OCR text.  

Instructions:
- Extract the following fields when present: 
  document_type, full_name_english, date_of_birth_gc, expiry_date_gc, gender, citizenship, fcn_number
- FCN (Family Card Number) should be exactly 16 digits. If it is not, include a note in validation_errors.
- Identify both Ethiopian Calendar (EC) and Gregorian Calendar (GC) dates where possible.
- Return data as a JSON object. Include validation_errors array if any critical data is missing or invalid.
- Rate extraction quality on a scale from 1 to 10 based on completeness and readability.

Example JSON structure:

Success case:
{
  "status": "success",
  "data": {
    "document_type": "Ethiopian Digital ID Card",
    "full_name_english": "Test Test Test",
    "date_of_birth_gc": "11/09/1991",
    "expiry_date_gc": "11/09/1991",
    "gender": "Male",
    "citizenship": "ET",
    "fcn_number": "1234567890123456",
    "extraction_quality": 7,
    "validation_errors": []
  }
}

Error case (FCN invalid or missing fields):
{
  "status": "error",
  "data": {
    "fcn_found": "123456789012",
    "fcn_length": 12,
    "validation_errors": ["FCN is not 16 digits", "Missing or incomplete fields"]
  }
}          The Context ocr raw text is: '''


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
        global prompt   # tell Python to use the global variable
        full_prompt = prompt + raw_text
        print(full_prompt)
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
