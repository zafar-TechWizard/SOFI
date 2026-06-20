"""
BRAIN/tools/web_tools.py — Real web tools for SOFi

- web_search  : DuckDuckGo search (no API key required)
- web_fetch   : Fetch + parse any webpage
- get_weather : Real weather via wttr.in (no API key required)

Dependencies (already in requirements.txt except duckduckgo-search):
    pip install duckduckgo-search
"""

import asyncio
import logging
import re

import httpx

_log = logging.getLogger("sofi.brain.tools.web")


# ── Web Search ────────────────────────────────────────────────────────────────

async def web_search(query: str, max_results: int = 5) -> str:
    # Package was renamed: duckduckgo_search → ddgs. Try new name first.
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "web_search unavailable — run: pip install ddgs"

    try:
        loop = asyncio.get_event_loop()

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max(1, min(max_results, 10))))

        results = await loop.run_in_executor(None, _search)

        if not results:
            return f"No results found for: {query}"

        lines = [f"Web search: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = (r.get("body") or "")[:200].strip()
            href = r.get("href", "")
            lines.append(f"{i}. {title}")
            if body:
                lines.append(f"   {body}")
            lines.append(f"   {href}\n")

        return "\n".join(lines)

    except Exception as exc:
        _log.error("web_search error: %s", exc)
        return f"Search failed: {exc}"


# ── Web Fetch ─────────────────────────────────────────────────────────────────

async def web_fetch(url: str, max_chars: int = 4000) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "web_fetch unavailable — run: pip install beautifulsoup4"

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SOFi/1.0; +research)"},
        ) as client:
            r = await client.get(url)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Strip noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                          "iframe", "noscript", "svg", "button", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... {len(text) - max_chars} more characters truncated]"

        return f"Page: {title}\nURL: {url}\n\n{text}"

    except httpx.HTTPStatusError as exc:
        return f"HTTP {exc.response.status_code}: {url}"
    except Exception as exc:
        _log.error("web_fetch error: %s", exc)
        return f"Fetch failed: {exc}"


# ── Weather ───────────────────────────────────────────────────────────────────

async def get_weather(city: str = "Jaipur") -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://wttr.in/{city.replace(' ', '+')}?format=j1",
                headers={"Accept": "application/json"},
            )
        r.raise_for_status()
        data = r.json()

        cur = data["current_condition"][0]
        desc = cur["weatherDesc"][0]["value"]
        temp = cur["temp_C"]
        feels = cur["FeelsLikeC"]
        humidity = cur["humidity"]
        wind = cur["windspeedKmph"]
        wind_dir = cur.get("winddir16Point", "")

        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)
        country = area.get("country", [{}])[0].get("value", "")

        forecast_lines = []
        for day in data.get("weather", [])[:3]:
            date = day["date"]
            max_c = day["maxtempC"]
            min_c = day["mintempC"]
            day_desc = day["hourly"][4]["weatherDesc"][0]["value"]
            forecast_lines.append(f"  {date}: {day_desc}, {min_c}°C – {max_c}°C")

        return (
            f"Weather — {area_name}, {country}\n"
            f"  Now: {desc}, {temp}°C (feels {feels}°C)\n"
            f"  Humidity: {humidity}%  |  Wind: {wind} km/h {wind_dir}\n"
            f"\n3-day forecast:\n" + "\n".join(forecast_lines)
        )

    except Exception as exc:
        _log.error("get_weather error: %s", exc)
        return f"Weather unavailable for {city}: {exc}"


# ── Registration ──────────────────────────────────────────────────────────────

def register_web_tools(registry) -> None:
    from BRAIN.tools.registry import ToolEntry

    registry.register(ToolEntry(
        name="web_search",
        description=(
            "Search the web using DuckDuckGo. Returns real, current results with titles, "
            "snippets, and URLs. Use for current events, news, facts, people, products, "
            "or anything requiring up-to-date information. NOT a mock — actual live search."
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–10, default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=web_search,
        category="information",
        capability_name="web_search",
        capability_description="Search the web in real time using DuckDuckGo.",
        capability_refusal="I can't reach the web right now.",
    ))

    registry.register(ToolEntry(
        name="web_fetch",
        description=(
            "Fetch and read the full content of any webpage or URL. "
            "Extracts clean readable text (strips nav, scripts, ads). "
            "Use to read articles, docs, GitHub READMEs, or any web page Zafar links."
        ),
        schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch (http:// or https://)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 4000)",
                    "default": 4000,
                },
            },
            "required": ["url"],
        },
        handler=web_fetch,
        category="information",
        capability_name="web_fetch",
        capability_description="Fetch and read any webpage's content.",
        capability_refusal="I can't fetch that page right now.",
    ))

    registry.register(ToolEntry(
        name="get_weather",
        description=(
            "Get real current weather and 3-day forecast for any city worldwide. "
            "Live data from wttr.in. Includes temperature, humidity, wind, and daily forecast."
        ),
        schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. 'Jaipur', 'Mumbai', 'London', 'New York')",
                    "default": "Jaipur",
                },
            },
            "required": [],
        },
        handler=get_weather,
        category="information",
        capability_name="weather",
        capability_description="Get real-time weather and 3-day forecasts for any city.",
        capability_refusal="I can't reach the weather service right now.",
    ))
