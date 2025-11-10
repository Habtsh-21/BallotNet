import re

def parse_id(raw_text: dict) -> dict:
    """
    Extracts key fields from OCR text.
    """
    text = raw_text.get("combined_text", "").upper()

    # Name (simple heuristic: line containing "NAME" or first line)
    name_match = re.search(r"NAME[:\s]*([A-Z ]+)", text)
    name = name_match.group(1).strip() if name_match else None

    # ID number (alphanumeric)
    id_match = re.search(r"(ID|IDENTITY)[\s#:]*([A-Z0-9]+)", text)
    id_number = id_match.group(2).strip() if id_match else None

    # Date of birth
    dob_match = re.search(r"(BIRTH|DOB)[\s#:]*([0-9]{2}/[0-9]{2}/[0-9]{4})", text)
    date_of_birth = dob_match.group(2) if dob_match else None

    # Expiry date
    exp_match = re.search(r"(EXPIRY|EXPIRES)[\s#:]*([0-9]{2}/[0-9]{2}/[0-9]{4})", text)
    expiry_date = exp_match.group(2) if exp_match else None

    return {
        "name": name,
        "id_number": id_number,
        "date_of_birth": date_of_birth,
        "expiry_date": expiry_date
    }
