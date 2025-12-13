import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def send_prediction_email(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    attachment_path: Optional[str] = None
) -> bool:
    """
    Send an email with optional HTML body and attachment.
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Plain text email body
        html_body: Optional HTML version of email body
        attachment_path: Optional path to file attachment
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    
    if not all([smtp_server, smtp_username, smtp_password]):
        raise ValueError("Missing required SMTP environment variables")
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = subject
    
    # Attach plain text body
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach HTML body if provided
    if html_body:
        msg.attach(MIMEText(html_body, 'html'))
    
    # Attach file if provided
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        
        encoders.encode_base64(part)
        filename = os.path.basename(attachment_path)
        part.add_header('Content-Disposition', f'attachment; filename={filename}')
        msg.attach(part)
    
    try:
        # Connect to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def send_race_alert(predictions_df: pd.DataFrame, threshold: float = 0.7) -> bool:
    """
    Send email alert for high-confidence race predictions.
    
    Args:
        predictions_df: DataFrame with columns including 'horse_name', 'win_probability', 
                       'race_id', 'race_date', 'course', etc.
        threshold: Minimum win probability to include in alert (default 0.7)
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    alert_email = os.getenv('ALERT_EMAIL')
    
    if not alert_email:
        raise ValueError("ALERT_EMAIL environment variable not set")
    
    # Filter high-confidence predictions
    high_confidence = predictions_df[predictions_df['win_probability'] >= threshold].copy()
    
    if high_confidence.empty:
        print(f"No predictions above threshold {threshold}")
        return False
    
    # Sort by win probability descending
    high_confidence = high_confidence.sort_values('win_probability', ascending=False)
    
    # Build plain text body
    subject = f"🏇 Stanley Racing Alert: {len(high_confidence)} High-Confidence Predictions"
    
    body_lines = [
        f"High-confidence predictions (threshold: {threshold:.0%}):",
        "",
    ]
    
    for _, row in high_confidence.iterrows():
        body_lines.append(
            f"• {row.get('horse_name', 'Unknown')} - "
            f"{row['win_probability']:.1%} win probability"
        )
        if 'race_date' in row:
            body_lines.append(f"  Race: {row.get('race_date', '')} - {row.get('course', '')}")
        body_lines.append("")
    
    body = "\n".join(body_lines)
    
    # Build HTML body with formatted table
    html_body = _build_html_table(high_confidence, threshold)
    
    return send_prediction_email(
        to_email=alert_email,
        subject=subject,
        body=body,
        html_body=html_body
    )


def _build_html_table(predictions_df: pd.DataFrame, threshold: float) -> str:
    """
    Build HTML email body with formatted predictions table.
    
    Args:
        predictions_df: DataFrame with prediction data
        threshold: Threshold used for filtering
        
    Returns:
        str: HTML formatted email body
    """
    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #333;
                line-height: 1.6;
            }}
            h2 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .high-prob {{
                color: #27ae60;
                font-weight: bold;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 12px;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <h2>🏇 Stanley Racing Predictions Alert</h2>
        <p>High-confidence predictions with win probability ≥ {threshold:.0%}</p>
        
        <table>
            <thead>
                <tr>
                    <th>Horse</th>
                    <th>Win Probability</th>
                    <th>Race Date</th>
                    <th>Course</th>
                    <th>Race Number</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in predictions_df.iterrows():
        prob_class = "high-prob" if row['win_probability'] >= 0.8 else ""
        html += f"""
                <tr>
                    <td>{row.get('horse_name', 'Unknown')}</td>
                    <td class="{prob_class}">{row['win_probability']:.1%}</td>
                    <td>{row.get('race_date', 'N/A')}</td>
                    <td>{row.get('course', 'N/A')}</td>
                    <td>{row.get('race_number', 'N/A')}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by Stanley Horse Racing Prediction System</p>
        </div>
    </body>
    </html>
    """
    
    return html

