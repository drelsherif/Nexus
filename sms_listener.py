"""
sms_listener.py
Two-way SMS control for NEXUS via Gmail IMAP.

How it works:
  1. You text commands from your phone (646-404-2406) to your Gmail address
  2. Your carrier converts the text to an email (free, no extra accounts)
  3. This script polls your Gmail every 30s and reads new emails from your number
  4. Commands are written as sentinel files that nexus_run.py watches

Commands you can text:
  KILL    → stops the run gracefully after the current round
  STATUS  → replies with current F1 / round progress
  PAUSE   → pauses after the current round (text RESUME to continue)
  RESUME  → resumes a paused run

Setup (one-time):
  1. Enable IMAP in Gmail: Settings → See all settings → Forwarding and POP/IMAP → Enable IMAP
  2. Create a Gmail App Password: myaccount.google.com → Security → App passwords
  3. Set your carrier in this file (PHONE_CARRIER below)

Run alongside nexus_run.py:
  python sms_listener.py --email you@gmail.com --pass "xxxx xxxx xxxx xxxx" &
  python nexus_run.py --ai-hub --rounds 10 --out-dir run_04 ...
"""

import argparse
import email
import imaplib
import json
import os
import re
import smtplib
import time
from email.header import decode_header
from email.mime.text import MIMEText
from pathlib import Path

# ── Your number (used to filter incoming emails from just you) ────────────────
YOUR_DIGITS   = "6464042406"   # 10 digits, no dashes

# Your carrier (for sending replies back to your phone):
#   att / verizon / tmobile / sprint / boost / cricket / metro
YOUR_CARRIER  = "verizon"      # CHANGE THIS to your carrier

CARRIER_GATES = {
    "att":      "@txt.att.net",
    "verizon":  "@vtext.com",
    "tmobile":  "@tmomail.net",
    "t-mobile": "@tmomail.net",
    "sprint":   "@messaging.sprintpcs.com",
    "boost":    "@smsmyboostmobile.com",
    "cricket":  "@sms.cricketwireless.net",
    "metro":    "@mymetropcs.com",
}

# ── Sentinel files (nexus_run.py watches these) ───────────────────────────────
SENTINEL_DIR  = Path(".")   # same dir as nexus_run.py — overridden by --sentinel-dir
KILL_FILE     = "KILL_NEXUS"
PAUSE_FILE    = "PAUSE_NEXUS"
STATUS_REQ    = "STATUS_REQUEST"
STATUS_RESP   = "STATUS_RESPONSE"

POLL_INTERVAL = 30   # seconds between Gmail checks


class SMSListener:
    def __init__(self, gmail_user: str, gmail_pass: str,
                 phone: str, carrier: str,
                 sentinel_dir: str = ".",
                 smtp_host: str = "smtp.gmail.com",
                 imap_host: str = "imap.gmail.com"):
        self.gmail_user   = gmail_user
        self.gmail_pass   = gmail_pass
        self.phone        = "".join(c for c in phone if c.isdigit())[-10:]
        self.carrier      = carrier.lower()
        self.sentinel_dir = Path(sentinel_dir)
        self.smtp_host    = smtp_host
        self.imap_host    = imap_host
        self.reply_to     = self.phone + CARRIER_GATES.get(self.carrier, "@vtext.com")
        self.running      = True

    # ── Incoming ──────────────────────────────────────────────────────────────

    def _header_text(self, value) -> str:
        """Decode email headers into a plain string."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        parts = []
        for chunk, charset in decode_header(value):
            if isinstance(chunk, bytes):
                parts.append(chunk.decode(charset or "utf-8", errors="ignore"))
            else:
                parts.append(chunk)
        return "".join(parts)

    def _fetch_new_from_phone(self) -> list:
        """
        Connect to Gmail IMAP, fetch unseen emails from this phone number.
        Carrier gateways send from addresses like 6464042406@vtext.com.
        Returns list of (uid, subject, body) tuples. Marks them as seen.
        """
        msgs = []
        try:
            mail = imaplib.IMAP4_SSL(self.imap_host)
            mail.login(self.gmail_user, self.gmail_pass)
            mail.select("INBOX")

            # Search for UNSEEN emails from any carrier gateway for this number
            # We search for the phone digits in the sender field
            _, data = mail.search(None, "UNSEEN")
            if not data or not data[0]:
                mail.logout()
                return msgs

            for uid in data[0].split():
                _, raw = mail.fetch(uid, "(RFC822)")
                if not raw or not raw[0] or not isinstance(raw[0], tuple):
                    continue
                msg = email.message_from_bytes(raw[0][1])
                from_addr = self._header_text(msg.get("From", "")).lower()
                subject = self._header_text(msg.get("Subject", "")).strip()

                # Check the email is from our phone number (carrier gateway)
                if self.phone not in from_addr:
                    # Also accept emails where subject/body contains the number
                    # (some carriers format differently)
                    body = self._get_body(msg)
                    if self.phone not in (subject + body):
                        continue

                body    = self._get_body(msg).strip()
                command = (subject + " " + body).strip().upper()
                msgs.append((uid, subject, body, command))
                # Mark as seen
                mail.store(uid, "+FLAGS", "\\Seen")

            mail.logout()
        except Exception as e:
            print(f"[Listener] IMAP error: {e}")
        return msgs

    def _get_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    raw = part.get_payload(decode=True)
                    if raw:
                        return raw.decode("utf-8", errors="ignore")
            return ""
        raw = msg.get_payload(decode=True)
        if raw:
            return raw.decode("utf-8", errors="ignore")
        # Fallback: payload may already be a string (no encoding)
        p = msg.get_payload()
        return p if isinstance(p, str) else ""

    # ── Outgoing ──────────────────────────────────────────────────────────────

    def reply(self, message: str):
        """Send an SMS reply back to the phone."""
        try:
            msg = MIMEText(message[:160])
            msg["From"]    = self.gmail_user
            msg["To"]      = self.reply_to
            msg["Subject"] = ""
            with smtplib.SMTP(self.smtp_host, 587) as s:
                s.starttls()
                s.login(self.gmail_user, self.gmail_pass)
                s.sendmail(self.gmail_user, self.reply_to, msg.as_string())
            print(f"[Listener] Replied to {self.reply_to}: {message[:60]}")
        except Exception as e:
            print(f"[Listener] Reply failed: {e}")

    # ── Command handling ──────────────────────────────────────────────────────

    def _handle_command(self, command: str):
        sd = self.sentinel_dir

        if "KILL" in command or "STOP" in command:
            (sd / KILL_FILE).write_text("kill requested via SMS")
            self.reply("NEXUS: KILL signal sent. Run will stop after current round.")
            print("[Listener] KILL sentinel written.")

        elif "PAUSE" in command:
            (sd / PAUSE_FILE).write_text("pause requested via SMS")
            self.reply("NEXUS: PAUSE signal sent. Run will pause after current round.")
            print("[Listener] PAUSE sentinel written.")

        elif "RESUME" in command:
            pause_path = sd / PAUSE_FILE
            if pause_path.exists():
                pause_path.unlink()
                self.reply("NEXUS: RESUMED.")
                print("[Listener] PAUSE sentinel removed.")
            else:
                self.reply("NEXUS: Not paused.")

        elif "STATUS" in command:
            # Write status request; nexus_run.py will write response file
            (sd / STATUS_REQ).write_text(str(time.time()))
            # Wait up to 90s for response
            deadline = time.time() + 90
            while time.time() < deadline:
                resp_path = sd / STATUS_RESP
                if resp_path.exists():
                    status = resp_path.read_text()
                    resp_path.unlink()
                    self.reply(status[:160])
                    return
                time.sleep(5)
            self.reply("NEXUS: No status yet (run may be between rounds).")

        elif "HELP" in command or "?" in command:
            self.reply("NEXUS cmds: KILL | PAUSE | RESUME | STATUS")

        else:
            self.reply(f"NEXUS: Unknown command '{command[:30]}'. Text HELP for options.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print(f"[Listener] Started. Polling Gmail every {POLL_INTERVAL}s.")
        print(f"[Listener] Watching for texts from {self.phone} → {self.reply_to}")
        print(f"[Listener] Sentinel dir: {self.sentinel_dir.resolve()}")
        print(f"[Listener] Commands: KILL | PAUSE | RESUME | STATUS")

        while self.running:
            try:
                messages = self._fetch_new_from_phone()
                for uid, subject, body, command in messages:
                    print(f"[Listener] Got command: '{command[:40]}'")
                    self._handle_command(command)
            except Exception as e:
                print(f"[Listener] Poll error: {e}")
            time.sleep(POLL_INTERVAL)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NEXUS SMS listener (runs alongside nexus_run.py)")
    ap.add_argument("--email",        required=True, help="Your Gmail address")
    ap.add_argument("--pass",         dest="password", required=True,
                    help="Gmail App Password (16 chars, no spaces)")
    ap.add_argument("--phone",        default=YOUR_DIGITS,
                    help="Your 10-digit phone number (default: stored number)")
    ap.add_argument("--carrier",      default=YOUR_CARRIER,
                    help="att|verizon|tmobile|sprint  (default set in script)")
    ap.add_argument("--sentinel-dir", default=".",
                    help="Directory to write sentinel files (match --out-dir of nexus_run.py)")
    ap.add_argument("--smtp",         default="smtp.gmail.com")
    ap.add_argument("--imap",         default="imap.gmail.com")
    ap.add_argument("--interval",     type=int, default=POLL_INTERVAL,
                    help="Seconds between Gmail polls (default 30)")
    args = ap.parse_args()

    POLL_INTERVAL = args.interval
    listener = SMSListener(
        gmail_user=args.email,
        gmail_pass=args.password,
        phone=args.phone,
        carrier=args.carrier,
        sentinel_dir=args.sentinel_dir,
        smtp_host=args.smtp,
        imap_host=args.imap,
    )
    listener.run()
