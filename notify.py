"""
notify.py
Send an SMS when a NEXUS run completes.

Two methods (no paid API required for either):

METHOD 1 — Email-to-SMS gateway (free, works with any carrier).
  Every US carrier has a free SMS email gateway:
    AT&T    → number@txt.att.net
    Verizon → number@vtext.com
    T-Mobile→ number@tmomail.net
    Sprint  → number@messaging.sprintpcs.com
  You send an email; the carrier converts it to an SMS.
  Requires: SMTP access (Gmail, Outlook, your hospital email, etc.)

METHOD 2 — Twilio (reliable, $0.008/msg, needs free account at twilio.com).
  Set env vars TWILIO_SID, TWILIO_AUTH, TWILIO_FROM.

Usage from command line:
  python notify.py --phone 5165551234 --carrier att \\
      --smtp-from you@gmail.com --smtp-pass "your_app_password" \\
      --subject "NEXUS done" --body "F1=0.812"

Usage from Python:
  from notify import send_sms_email, send_sms_twilio
"""

import argparse
import os
import smtplib
from email.mime.text import MIMEText


# ── Carrier gateway map ───────────────────────────────────────────────────────
CARRIERS = {
    "att":      "@txt.att.net",
    "verizon":  "@vtext.com",
    "tmobile":  "@tmomail.net",
    "t-mobile": "@tmomail.net",
    "sprint":   "@messaging.sprintpcs.com",
    "boost":    "@smsmyboostmobile.com",
    "cricket":  "@sms.cricketwireless.net",
    "metro":    "@mymetropcs.com",
    "uscellular":"@email.uscc.net",
}


def send_sms_email(
    phone: str,
    carrier: str,
    message: str,
    smtp_from: str,
    smtp_pass: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    subject: str   = "",
) -> bool:
    """
    Send an SMS via carrier email gateway.
    For Gmail: use an App Password (myaccount.google.com → Security → App passwords).
    For Outlook/hospital email: use your normal password + smtp.office365.com port 587.
    Returns True on success, False on failure.
    """
    carrier = carrier.lower().strip()
    if carrier not in CARRIERS:
        print(f"[Notify] Unknown carrier '{carrier}'. Options: {', '.join(CARRIERS.keys())}")
        return False

    # Strip non-digits from phone number
    digits = "".join(c for c in phone if c.isdigit())[-10:]
    to_addr = digits + CARRIERS[carrier]

    msg = MIMEText(message[:160])  # SMS limit
    msg["From"]    = smtp_from
    msg["To"]      = to_addr
    msg["Subject"] = subject[:40]  # keep header short

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_from, smtp_pass)
            server.sendmail(smtp_from, to_addr, msg.as_string())
        print(f"[Notify] SMS sent to {digits}@{CARRIERS[carrier].lstrip('@')}")
        return True
    except Exception as e:
        print(f"[Notify] Failed to send SMS: {e}")
        return False


def send_sms_twilio(
    to_phone:   str,
    message:    str,
    account_sid: str = None,
    auth_token:  str = None,
    from_phone:  str = None,
) -> bool:
    """
    Send SMS via Twilio ($0.008/message, needs twilio.com account).
    Reads TWILIO_SID, TWILIO_AUTH, TWILIO_FROM from env if not passed.
    """
    try:
        from twilio.rest import Client
    except ImportError:
        print("[Notify] Twilio not installed. Run: pip install twilio")
        return False

    sid   = account_sid or os.environ.get("TWILIO_SID")
    auth  = auth_token  or os.environ.get("TWILIO_AUTH")
    frm   = from_phone  or os.environ.get("TWILIO_FROM")
    if not (sid and auth and frm):
        print("[Notify] Twilio: need TWILIO_SID, TWILIO_AUTH, TWILIO_FROM env vars.")
        return False
    try:
        c = Client(sid, auth)
        c.messages.create(body=message[:1600], from_=frm, to=to_phone)
        print(f"[Notify] Twilio SMS sent to {to_phone}")
        return True
    except Exception as e:
        print(f"[Notify] Twilio error: {e}")
        return False


def format_nexus_message(run_dir: str, best_f1: float, rounds: int,
                          final_f1: float, accepted_changes: list,
                          principles_count: int, engrams_formed: int) -> str:
    """Build a compact SMS-friendly summary of a NEXUS run."""
    changes = len(accepted_changes)
    lines = [
        f"NEXUS run complete ({run_dir})",
        f"Best F1={best_f1:.3f}  Final={final_f1:.3f}  ({rounds} rounds)",
        f"Changes={changes}  Principles={principles_count}  Engrams={engrams_formed}",
    ]
    if accepted_changes:
        lines.append("Last: " + accepted_changes[-1][:40])
    return "\n".join(lines)


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Send a test SMS notification")
    ap.add_argument("--phone",      required=True, help="10-digit phone number")
    ap.add_argument("--carrier",    default="att", help="Carrier (att/verizon/tmobile/sprint)")
    ap.add_argument("--smtp-from",  required=True, help="Sender email address")
    ap.add_argument("--smtp-pass",  required=True, help="Email password or app password")
    ap.add_argument("--smtp-host",  default="smtp.gmail.com")
    ap.add_argument("--smtp-port",  type=int, default=587)
    ap.add_argument("--subject",    default="NEXUS")
    ap.add_argument("--body",       default="NEXUS run complete.")
    ap.add_argument("--twilio",     action="store_true", help="Use Twilio instead")
    args = ap.parse_args()

    if args.twilio:
        send_sms_twilio(to_phone=args.phone, message=args.body)
    else:
        send_sms_email(
            phone=args.phone, carrier=args.carrier,
            message=args.body,
            smtp_from=args.smtp_from, smtp_pass=args.smtp_pass,
            smtp_host=args.smtp_host, smtp_port=args.smtp_port,
            subject=args.subject,
        )
