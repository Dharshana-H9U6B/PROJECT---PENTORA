"""
Text input processor.

Cleans and normalizes raw text input before analysis.
"""

import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning for analysis input.

    - Normalizes whitespace
    - Strips leading/trailing whitespace
    - Preserves URLs and key content
    """
    if not text:
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def build_structured_text(
    company: str = "",
    role: str = "",
    salary: str = "",
    registration_fee: str = "",
    contact_method: str = "",
    website: str = "",
    description: str = "",
) -> str:
    """
    Convert structured form fields into a normalized analysis text.
    """
    parts = []

    if company:
        parts.append(f"Company: {company}")
    if role:
        parts.append(f"Role/Position: {role}")
    if salary:
        parts.append(f"Salary/Stipend: {salary}")
    if registration_fee:
        parts.append(f"Registration Fee: {registration_fee}")
    if contact_method:
        parts.append(f"Contact Method: {contact_method}")
    if website:
        parts.append(f"Website: {website}")
    if description:
        parts.append(f"\nDescription:\n{description}")

    return "\n".join(parts)


def validate_text_input(text: str) -> tuple[bool, str]:
    """
    Validate that the input text is suitable for analysis.

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter the opportunity message or details."

    if len(text.strip()) < 20:
        return False, "The message is too short for meaningful analysis. Please provide more details."

    if len(text) > 50000:
        return False, "The message is too long. Please trim it to under 50,000 characters."

    return True, ""
