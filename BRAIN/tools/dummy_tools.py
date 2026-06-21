"""
BRAIN/tools/dummy_tools.py — Communication stub tools

get_current_time  : Real — returns actual current time
check_emails      : Stub — needs Gmail OAuth to go live
send_email        : Stub — needs Gmail OAuth to go live
check_whatsapp    : Stub — no official personal WhatsApp API
send_whatsapp     : Stub — no official personal WhatsApp API
check_calendar    : Stub — needs Google Calendar OAuth to go live

Real web/filesystem/execution tools live in web_tools.py, fs_tools.py, exec_tools.py.
"""

import datetime
import json
import random

from BRAIN.tools.registry import ToolEntry, ToolRegistry


# ── Handlers ──────────────────────────────────────────────────────────────

def get_current_time() -> str:
    now = datetime.datetime.now()
    return json.dumps({
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day": now.strftime("%A"),
        "timezone": "IST (Asia/Kolkata)",
    })


def check_emails(query: str = "", max_results: int = 5) -> str:
    emails = [
        {"from": "Aman Gupta", "subject": "RE: Project timeline update", "snippet": "Hey, the deadline moved to next Friday. Can we sync tomorrow?", "unread": True, "time": "2 hours ago"},
        {"from": "AWS Billing", "subject": "Your AWS bill for June", "snippet": "Your estimated charges for this month: $47.23", "unread": True, "time": "5 hours ago"},
        {"from": "Mom", "subject": "Dinner Sunday?", "snippet": "Beta, are you coming for dinner this Sunday? Dad is making biryani.", "unread": True, "time": "yesterday"},
        {"from": "GitHub", "subject": "[anthropics/claude-code] New release v1.0.20", "snippet": "A new release has been published...", "unread": False, "time": "yesterday"},
        {"from": "Spotify", "subject": "Your weekly discovery mix is ready", "snippet": "30 new songs picked just for you", "unread": False, "time": "2 days ago"},
    ]

    if query:
        q = query.lower()
        emails = [e for e in emails if q in e["from"].lower() or q in e["subject"].lower() or q in e["snippet"].lower()]

    return json.dumps(emails[:max_results], indent=2)


def send_email(to: str, subject: str, body: str) -> str:
    return json.dumps({
        "status": "sent",
        "to": to,
        "subject": subject,
        "message_id": f"msg_{random.randint(10000, 99999)}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def check_whatsapp(contact: str = "") -> str:
    messages = [
        {"from": "Aman", "message": "bro check the figma link I sent", "time": "30 min ago", "unread": True},
        {"from": "Mom", "message": "Call me when free", "time": "1 hour ago", "unread": True},
        {"from": "Riya", "message": "haha that meme was gold 😂", "time": "3 hours ago", "unread": False},
    ]

    if contact:
        c = contact.lower()
        messages = [m for m in messages if c in m["from"].lower()]

    return json.dumps(messages, indent=2)


def send_whatsapp(contact: str, message: str) -> str:
    return json.dumps({
        "status": "sent",
        "to": contact,
        "message_preview": message[:50],
        "timestamp": datetime.datetime.now().strftime("%H:%M"),
    })


def check_calendar(date: str = "today") -> str:
    events = [
        {"title": "Standup", "time": "10:00 AM", "duration": "15 min", "location": "Google Meet"},
        {"title": "SOFi Architecture Review", "time": "2:00 PM", "duration": "1 hour", "location": "Zoom"},
        {"title": "Gym", "time": "6:30 PM", "duration": "1 hour", "location": "Cult Fit"},
    ]
    return json.dumps({"date": date, "events": events}, indent=2)


# ── Registration ──────────────────────────────────────────────────────────

def register_dummy_tools(registry: ToolRegistry) -> None:
    """Register all dummy tools into the given registry."""

    registry.register(ToolEntry(
        name="get_current_time",
        description="Get the current date, time, and day of the week.",
        schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=get_current_time,
        category="information",
        capability_name="time_awareness",
        capability_description="Check the current time and date.",
    ))

    registry.register(ToolEntry(
        name="check_emails",
        description="Read emails from Zafar's inbox. Can filter by search query.",
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search filter (e.g., 'from:aman', 'unread', a keyword). Empty returns recent emails.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to return.",
                    "default": 5,
                },
            },
            "required": [],
        },
        handler=check_emails,
        category="communication",
        capability_name="email_read",
        capability_description="Read and search Zafar's emails.",
    ))

    registry.register(ToolEntry(
        name="send_email",
        description="Send an email on Zafar's behalf. Use only when he explicitly asks to send.",
        schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email or name."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
            },
            "required": ["to", "subject", "body"],
        },
        handler=send_email,
        needs_confirmation=True,
        category="communication",
        capability_name="email_send",
        capability_description="Send emails on Zafar's behalf.",
    ))

    registry.register(ToolEntry(
        name="check_whatsapp",
        description="Check recent WhatsApp messages. Can filter by contact name.",
        schema={
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Contact name to filter messages. Empty returns all recent.",
                },
            },
            "required": [],
        },
        handler=check_whatsapp,
        category="communication",
        capability_name="whatsapp_read",
        capability_description="Read Zafar's WhatsApp messages.",
    ))

    registry.register(ToolEntry(
        name="send_whatsapp",
        description="Send a WhatsApp message to a contact. Use only when Zafar explicitly asks.",
        schema={
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Contact name."},
                "message": {"type": "string", "description": "Message text to send."},
            },
            "required": ["contact", "message"],
        },
        handler=send_whatsapp,
        needs_confirmation=True,
        category="communication",
        capability_name="whatsapp_send",
        capability_description="Send WhatsApp messages on Zafar's behalf.",
    ))

    registry.register(ToolEntry(
        name="check_calendar",
        description="Check Zafar's calendar events for a given date.",
        schema={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to check (e.g., 'today', 'tomorrow', '2026-06-15').",
                    "default": "today",
                },
            },
            "required": [],
        },
        handler=check_calendar,
        category="information",
        capability_name="calendar_read",
        capability_description="Check Zafar's calendar and schedule.",
    ))



# Auto-discovery alias — brain.py looks for register(registry)
register = register_dummy_tools
