"""Contact-detail validation. Syntax only: no network or deliverability checks."""
from __future__ import annotations

import phonenumbers
from email_validator import EmailNotValidError, validate_email


def _e164(number: phonenumbers.PhoneNumber) -> str:
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)


def validate_phone(text: str, region: str = "GB") -> str | None:
    """Return the number in E.164 form if `text` contains a valid phone number."""
    for match in phonenumbers.PhoneNumberMatcher(text or "", region):
        if phonenumbers.is_valid_number(match.number):
            return _e164(match.number)
    try:
        number = phonenumbers.parse((text or "").strip(), region)
    except phonenumbers.NumberParseException:
        return None
    return _e164(number) if phonenumbers.is_valid_number(number) else None


def validate_email_address(text: str) -> str | None:
    """Return the normalised email if `text` is (or contains) a valid address."""
    for token in (text or "").replace(",", " ").split():
        token = token.strip("<>()[]\"'.;:")
        if "@" not in token:
            continue
        try:
            return validate_email(token, check_deliverability=False).normalized
        except EmailNotValidError:
            continue
    return None
