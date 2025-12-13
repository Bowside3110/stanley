import os
import re
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()


def format_phone_number(phone: str) -> str:
    """
    Format phone number to E.164 format for Australia.
    
    Args:
        phone: Phone number in various formats (e.g., "0412345678", "412345678", "+61412345678")
        
    Returns:
        str: E.164 formatted number (e.g., "+61412345678")
    """
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Handle Australian mobile numbers
    if digits.startswith('61'):
        # Already has country code
        return f"+{digits}"
    elif digits.startswith('0'):
        # Remove leading 0 and add country code
        return f"+61{digits[1:]}"
    else:
        # Assume it's missing the leading 0
        return f"+61{digits}"


def send_race_alert_sms(to_number: str, predictions_df: pd.DataFrame, threshold: float = 0.7) -> bool:
    """
    Send SMS alert for high-confidence race predictions via Twilio.
    
    Args:
        to_number: Recipient phone number (will be formatted to E.164)
        predictions_df: DataFrame with columns including 'win_probability', 'race_number', 
                       'horse_number', etc.
        threshold: Minimum win probability to include in alert (default 0.7)
        
    Returns:
        bool: True if SMS sent successfully, False otherwise
    """
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_number = os.getenv('TWILIO_FROM_PHONE')
    
    if not all([account_sid, auth_token, from_number]):
        raise ValueError("Missing required Twilio environment variables")
    
    # Filter high-confidence predictions
    high_confidence = predictions_df[predictions_df['win_probability'] >= threshold].copy()
    
    if high_confidence.empty:
        print(f"No predictions above threshold {threshold}")
        return False
    
    # Sort by win probability descending
    high_confidence = high_confidence.sort_values('win_probability', ascending=False)
    
    # Build SMS body
    count = len(high_confidence)
    body_lines = [f"Stanley Alert: {count} pick{'s' if count > 1 else ''}"]
    
    for _, row in high_confidence.iterrows():
        race_num = row.get('race_number', '?')
        horse_num = row.get('horse_number', '?')
        prob_pct = row['win_probability'] * 100
        body_lines.append(f"Race {race_num}: Horse #{horse_num} - {prob_pct:.1f}%")
    
    body = "\n".join(body_lines)
    
    try:
        # Format phone number to E.164
        formatted_number = format_phone_number(to_number)
        
        # Send SMS via Twilio
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=from_number,
            to=formatted_number
        )
        
        print(f"SMS sent successfully. SID: {message.sid}")
        return True
        
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return False

