---
name: daily_briefing
title: Daily Briefing
description: Compile a tight morning summary — weather, calendar, email highlights, top news
requires: [web_search, get_weather, check_emails, check_calendar]
tags: [productivity, morning, daily, summary]
---

# Daily Briefing

When Zafar asks for a morning briefing, daily summary, or "what's on today", compile the following in order:

1. **Weather** — use `get_weather` for current conditions and today's forecast. One line only.

2. **Calendar** — check today's events. List them with times. Flag any conflict or tight back-to-back.

3. **Emails** — unread count and anything flagged, urgent, or from a name Zafar mentions often.
   Skip newsletters unless he explicitly asked about them.

4. **News** — 2-3 headlines relevant to what he's currently working on or interested in.
   Use `web_search` with a targeted query (not generic "top news").

## Format rules

- Bullet format throughout. No prose paragraphs.
- Under 200 words total.
- If a section has nothing useful, omit it — don't write "No emails."
- End with: **"Ready when you are, sir."**

## When a tool is unavailable

If `check_emails` or `check_calendar` isn't available, skip that section silently.
Do not say "I couldn't check your email" — just omit it.
