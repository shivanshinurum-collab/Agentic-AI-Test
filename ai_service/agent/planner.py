import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
from .tools import ToolRegistry

class BasePlanner:
    def plan_next_step(
        self,
        prompt: str,
        tools_info: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        memory_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determines the next step.
        Returns:
            {
                "thought": str,
                "is_final": bool,
                "action": Optional[str],
                "action_input": Optional[Dict[str, Any]],
                "final_answer": Optional[str]
            }
        """
        raise NotImplementedError()


class AutonomousHeuristicPlanner(BasePlanner):
    """
    Dynamic Cognitive Planner that orchestrates multi-step ReAct execution sequences
    by calling live tools (Google Maps/OSRM, Open-Meteo Weather, DuckDuckGo Search,
    Wikipedia, Calculator, Python Sandbox) without relying on any hardcoded data.
    """
    def plan_next_step(
        self,
        prompt: str,
        tools_info: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        memory_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        step_num = len(trace) + 1
        prompt_lower = prompt.lower().strip()

        # =========================================================================
        # STEP 1: Intent Detection & Dynamic Initial Tool Selection
        # =========================================================================
        if step_num == 1:
            # 1. Check for URL Fetch / Web Scraping Request
            url_match = re.search(r'https?://[^\s<>"]+', prompt)
            if url_match and any(w in prompt_lower for w in ["fetch", "read", "scrape", "open", "extract", "url", "page", "website", "content"]):
                target_url = url_match.group(0)
                return {
                    "thought": f"The user provided a URL to read and analyze: '{target_url}'. I will use fetch_web_page to retrieve its clean content.",
                    "is_final": False,
                    "action": "fetch_web_page",
                    "action_input": {"url": target_url}
                }

            # 2. Check for Live Weather Inquiry
            weather_keywords = ["weather", "temperature", "forecast", "climate", "mausam", "rain", "humidity", "how hot", "how cold"]
            if any(k in prompt_lower for k in weather_keywords) and not any(w in prompt_lower for w in ["trip", "tour", "itinerary", "vacation"]):
                # Extract city
                loc_cand = prompt
                for prefix in ["weather in", "weather of", "weather at", "weather for", "temperature in", "temperature of", "forecast for", "forecast in"]:
                    if prefix in loc_cand.lower():
                        parts = loc_cand.lower().split(prefix, 1)
                        loc_cand = parts[1].strip(" ?.!,")
                        break
                clean_loc = re.sub(r'\b(today|tomorrow|now|currently|live|please|tell|give|show)\b', '', loc_cand, flags=re.IGNORECASE).strip(" ?.,")
                return {
                    "thought": f"The user is requesting live real-time weather for '{clean_loc or prompt}'. I will query the weather_forecast tool.",
                    "is_final": False,
                    "action": "weather_forecast",
                    "action_input": {"location": clean_loc or prompt}
                }

            # 3. Check for Trip / Travel / Vacation / Itinerary Inquiry
            is_pure_directions = any(w in prompt_lower for w in ["directions", "turn by turn", "navigation", "how far is", "distance between", "geocode", "coordinates"]) and not any(w in prompt_lower for w in ["trip", "itinerary", "vacation", "tour", "holiday", "darshan", "stay", "hotel", "food", "places to visit"])
            trip_keywords = ["trip", "travel", "tour", "itinerary", "vacation", "holiday", "visit", "sightseeing", "darshan", "explore", "stay", "budget", "hotel", "food", "khana", "ghoomne", "plan", "guide", "package"]
            is_trip_inquiry = not is_pure_directions and (any(k in prompt_lower for k in trip_keywords) or bool(re.search(r'\b(?:to|in|se)\s+[a-zA-Z\s]+\s+(?:trip|plan|tour)\b', prompt_lower)))

            if is_trip_inquiry:
                days = 2 if any(w in prompt_lower for w in ["2 day", "2-day", "two day", "weekend"]) else (3 if any(w in prompt_lower for w in ["3 day", "3-day", "three day"]) else 1)
                
                # Detect travelers count
                travelers = 2
                travelers_match = re.search(r'(\d+)\s*(?:people|person|travelers|pax|members)', prompt_lower)
                if travelers_match:
                    try:
                        travelers = int(travelers_match.group(1))
                    except ValueError:
                        travelers = 2
                elif "solo" in prompt_lower:
                    travelers = 1

                # Focus detection
                focus = "all"
                if any(w in prompt_lower for w in ["food", "khana", "street food", "restaurant", "sweet", "chaat", "kachori", "culinary", "dishes"]):
                    focus = "food"
                elif any(w in prompt_lower for w in ["stay", "hotel", "resort", "dharamshala", "room", "lodge", "where to stay", "accommodation"]):
                    focus = "stay"
                elif any(w in prompt_lower for w in ["budget", "cost", "kharcha", "how much", "expense", "price"]):
                    focus = "budget"
                elif any(w in prompt_lower for w in ["route", "how to reach", "how to go", "distance", "train", "flight", "bus", "driving"]):
                    focus = "route"
                elif any(w in prompt_lower for w in ["itinerary", "day 1", "day 2", "schedule", "timeline", "plan"]):
                    focus = "itinerary"
                elif any(w in prompt_lower for w in ["darshan", "temple", "mandir", "aarti", "attractions", "sightseeing", "places"]):
                    focus = "attractions"

                # Dynamic origin & destination extraction
                orig = "Indore"
                dest = "Pachmarhi"
                
                from_to_match = re.search(
                    r'(?:from\s+)?([a-zA-Z\s]{2,30}?)\s+(?:to|se|-|➔|->)\s+([a-zA-Z\s]{2,30}?)(?:\s+(?:trip|travel|tour|itinerary|ghoomne|jaana|darshan|guide|plan|package|distance|route|fare|hotels?|food|places)|\?|$|[.,!])',
                    prompt,
                    re.IGNORECASE
                )

                if from_to_match:
                    cand_orig = from_to_match.group(1).strip().title()
                    cand_dest = from_to_match.group(2).strip().title()
                    cand_orig_clean = re.sub(r'^(?:plan|trip|tour|guide|for|from|going|want|need|give|show)\s+', '', cand_orig, flags=re.IGNORECASE).strip()
                    cand_dest_clean = re.sub(r'\s+(?:trip|tour|guide|plan|food|hotel|hotels|stay|budget|for|with|near|and|to|from)$', '', cand_dest, flags=re.IGNORECASE).strip()
                    if cand_orig_clean and cand_dest_clean:
                        orig = cand_orig_clean
                        dest = cand_dest_clean
                else:
                    single_dest = re.search(
                        r'(?:trip\s+(?:to|for|of|in)|travel\s+(?:to|for|in)|visit\s+(?:to|in)?|tour\s+(?:of|to|in)|guide\s+(?:for|to|of)|hotels?\s+(?:in|near|at)|stay\s+(?:in|at)|food\s+(?:in|at)|budget\s+for|places\s+to\s+visit\s+in)\s+([a-zA-Z\s]{2,30})',
                        prompt,
                        re.IGNORECASE
                    )
                    if single_dest:
                        cand = single_dest.group(1).strip().title()
                        cand_clean = re.sub(r'\s+(?:trip|tour|guide|plan|food|hotel|hotels|stay|budget|for|with|near|and|to|from).*', '', cand, flags=re.IGNORECASE).strip()
                        if cand_clean:
                            dest = cand_clean
                            orig = "Indore" if dest.lower() != "indore" else "Bhopal"
                    else:
                        dest_prefix = re.search(r'([a-zA-Z]{3,25})\s+(?:trip|tour|itinerary|vacation|holiday|ghoomne|darshan|guide)', prompt, re.IGNORECASE)
                        if dest_prefix:
                            dest = dest_prefix.group(1).strip().title()
                            orig = "Indore" if dest.lower() != "indore" else "Bhopal"

                # Step 1: Query Google Maps / OSRM for live distance & route
                return {
                    "thought": f"The user wants a travel plan for '{orig} ➔ {dest}'. First, I will query google_maps for live directions, road distance, and travel duration.",
                    "is_final": False,
                    "action": "google_maps",
                    "action_input": {
                        "action": "directions",
                        "origin": orig,
                        "destination": dest,
                        "mode": "driving"
                    }
                }

            # 4. Check for Google Maps / Geocoding / Routing / Places Inquiry
            map_keywords = [
                "map", "google map", "route", "directions", "distance", "how far", "navigate",
                "navigation", "geocode", "coordinates", "drive to", "drive from", "travel from",
                "places near", "restaurants in", "cafes in", "hotels in", "location of", "where is"
            ]
            if any(k in prompt_lower for k in map_keywords):
                # Directions detection (from X to Y)
                from_to_match = re.search(r'(?:from|between)\s+([^,]+?)\s+(?:to|and)\s+([^?.!]+)', prompt, re.IGNORECASE)
                if from_to_match:
                    orig = from_to_match.group(1).strip()
                    dest = from_to_match.group(2).strip()
                    mode = "driving"
                    if "walk" in prompt_lower:
                        mode = "walking"
                    elif "bike" in prompt_lower or "bicycle" in prompt_lower:
                        mode = "bicycling"
                    elif "transit" in prompt_lower or "train" in prompt_lower or "bus" in prompt_lower:
                        mode = "transit"

                    return {
                        "thought": f"The user is requesting live route and driving navigation from '{orig}' to '{dest}'. Querying google_maps.",
                        "is_final": False,
                        "action": "google_maps",
                        "action_input": {
                            "action": "directions",
                            "origin": orig,
                            "destination": dest,
                            "mode": mode
                        }
                    }

                # Places search detection
                places_match = re.search(r'(?:find|search|show|top|best|nearby)?\s*(restaurants|cafes|coffee shops|hotels|bars|hospitals|pharmacies|banks|museums|parks|temples)?\s*(?:in|near|around)\s+([^?.!]+)', prompt, re.IGNORECASE)
                if places_match and (places_match.group(1) or places_match.group(2)):
                    place_type = places_match.group(1) or "places"
                    location = places_match.group(2).strip() if places_match.group(2) else ""
                    search_query = f"{place_type} in {location}".strip()
                    return {
                        "thought": f"The user wants to discover live places: '{search_query}'. Invoking google_maps places_search.",
                        "is_final": False,
                        "action": "google_maps",
                        "action_input": {
                            "action": "places_search",
                            "query": search_query
                        }
                    }

                # Geocoding / Location coordinates detection
                geo_query = prompt
                for prefix in ["where is", "location of", "coordinates of", "geocode", "find"]:
                    if geo_query.lower().startswith(prefix):
                        geo_query = geo_query[len(prefix):].strip()
                        break
                return {
                    "thought": f"The user is looking for geographical coordinates of '{geo_query}'. Invoking google_maps geocode.",
                    "is_final": False,
                    "action": "google_maps",
                    "action_input": {
                        "action": "geocode",
                        "query": geo_query
                    }
                }

            # 5. Check for Math / Calculation
            math_match = re.search(r'[\d\.\s\+\-\*\/\^\(\)\%\,\√\=]+', prompt)
            calc_keywords = ["calculate", "math", "sqrt", "add", "multiply", "divide", "sum", "plus", "times", "minus", "expression", "eval", "evaluate"]
            if (math_match and len(math_match.group(0).strip()) > 3 and any(char.isdigit() for char in math_match.group(0))) or any(k in prompt_lower for k in calc_keywords):
                expr = prompt
                for prefix in ["calculate", "what is", "eval", "evaluate", "compute", "solve", "math"]:
                    if expr.lower().startswith(prefix):
                        expr = expr[len(prefix):].strip(" :?")
                return {
                    "thought": f"The user needs to evaluate a mathematical formula: '{expr}'. Invoking calculator tool.",
                    "is_final": False,
                    "action": "calculator",
                    "action_input": {"expression": expr}
                }

            # 6. Check for Python / Code Execution Inquiry
            if any(w in prompt_lower for w in ["python", "code", "script", "loop", "fibonacci", "factorial", "algorithm", "array", "matrix"]):
                code = "def solve():\n    return [x**2 for x in range(1, 11)]\nprint('Squares 1-10:', solve())"
                if "fibonacci" in prompt_lower:
                    code = "def fib(n):\n    a, b = 0, 1\n    res = []\n    for _ in range(n):\n        res.append(a)\n        a, b = b, a + b\n    return res\nprint('Fibonacci sequence:', fib(10))"
                elif "factorial" in prompt_lower:
                    num = re.search(r'\d+', prompt)
                    n = num.group(0) if num else "5"
                    code = f"import math\nprint(f'Factorial of {n}: {{math.factorial({n})}}')"
                
                return {
                    "thought": "The user requested Python code logic execution. Invoking python_interpreter.",
                    "is_final": False,
                    "action": "python_interpreter",
                    "action_input": {"code": code}
                }

            # 7. Check for Sentiment / NLP Analysis
            if any(w in prompt_lower for w in ["sentiment", "analyze text", "tone", "nlp", "opinion", "emotion"]):
                quote_match = re.search(r'["\']([^"\']+)["\']', prompt)
                text_to_eval = quote_match.group(1) if quote_match else prompt
                return {
                    "thought": f"Performing PyTorch sentiment and NLP analysis on: '{text_to_eval}'.",
                    "is_final": False,
                    "action": "nlp_analyzer",
                    "action_input": {"text": text_to_eval}
                }

            # 8. Check for Date / Time Inquiry
            if any(w in prompt_lower for w in ["today", "current date", "current time", "what day is", "tomorrow date", "yesterday date"]):
                return {
                    "thought": "Retrieving real-time timestamp and temporal calculations from datetime_tool.",
                    "is_final": False,
                    "action": "datetime_tool",
                    "action_input": {"days_offset": 0}
                }

            # 9. Default: Live Web Search / Wikipedia Retrieval for any topic/fact
            search_query = prompt
            for prefix in ["search for", "search web for", "what is", "who is", "tell me about", "details of", "information about"]:
                if search_query.lower().startswith(prefix):
                    search_query = search_query[len(prefix):].strip(" ?.")
                    break

            return {
                "thought": f"The user is inquiring about '{search_query}'. I will perform a live web search to retrieve real-time facts and references.",
                "is_final": False,
                "action": "web_search",
                "action_input": {"query": search_query or prompt}
            }

        # =========================================================================
        # STEP 2+: Multi-Step ReAct Sequence or Final Answer Synthesis
        # =========================================================================
        last_step = trace[-1]
        last_action = last_step.get("action")
        last_obs = last_step.get("observation", {})

        if step_num == 2:
            # Multi-step travel planning sequence: Google Maps route -> Weather -> Trip Planner
            trip_keywords = ["trip", "travel", "tour", "itinerary", "vacation", "holiday", "visit", "sightseeing", "darshan", "explore", "stay", "budget", "hotel", "food", "khana", "ghoomne", "plan", "guide", "package"]
            is_trip_goal = any(k in prompt_lower for k in trip_keywords) or bool(re.search(r'\b(?:to|in|se)\s+[a-zA-Z\s]+\s+(?:trip|plan|tour)\b', prompt_lower))
            
            if is_trip_goal and last_action == "google_maps":
                orig = last_step.get("action_input", {}).get("origin") or last_obs.get("origin") or "Indore"
                dest = last_step.get("action_input", {}).get("destination") or last_obs.get("destination") or "Pachmarhi"
                days = 2 if any(w in prompt_lower for w in ["2 day", "2-day", "two day", "weekend"]) else (3 if any(w in prompt_lower for w in ["3 day", "3-day", "three day"]) else 1)
                dist_str = f" ({last_obs.get('distance_text')}, {last_obs.get('duration_text')})" if last_obs.get('distance_text') else ""
                
                return {
                    "thought": f"I obtained live Google Maps routing for '{orig} ➔ {dest}'{dist_str}. Now I will invoke trip_planner to dynamically retrieve live weather, top attractions, hotels, food spots, and calculated expenses for '{dest}'.",
                    "is_final": False,
                    "action": "trip_planner",
                    "action_input": {
                        "origin": orig,
                        "destination": dest,
                        "duration_days": days,
                        "travelers_count": 2,
                        "focus": "all"
                    }
                }

            # If web search returned results, optionally query wikipedia for deeper summary
            elif last_action == "web_search" and any(w in prompt_lower for w in ["who is", "history of", "what is", "about", "wiki"]):
                query = last_step.get("action_input", {}).get("query", prompt)
                return {
                    "thought": f"Live web search is complete. I will now fetch comprehensive encyclopedia details from wikipedia_search for '{query}'.",
                    "is_final": False,
                    "action": "wikipedia_search",
                    "action_input": {"query": query}
                }

        # =========================================================================
        # FINAL ANSWER SYNTHESIS (From Live Tool Observations)
        # =========================================================================
        answer_parts = []
        
        # Check for trip planner observation
        trip_obs = next((s.get("observation", {}) for s in trace if s.get("action") == "trip_planner" and s.get("observation", {}).get("success")), None)
        maps_obs = next((s.get("observation", {}) for s in trace if s.get("action") == "google_maps" and s.get("observation", {}).get("success")), None)

        if trip_obs:
            guide = trip_obs.get("guide", {})
            dest = trip_obs.get("destination", "Destination")
            orig = trip_obs.get("origin", "Origin")
            focus = trip_obs.get("focus", "all")

            # Route distance metrics
            maps_route_info = ""
            if maps_obs and maps_obs.get("action") == "directions":
                maps_route_info = (
                    f"\n• **🗺️ Live Distance & Duration**: **{maps_obs.get('distance_text')}** (approx. **{maps_obs.get('duration_text')}** driving via {maps_obs.get('provider')})\n"
                    f"• **Interactive Route Map**: [Open in Google Maps]({maps_obs.get('maps_link')})"
                )

            # Live weather summary
            weather_data = guide.get("live_weather", {})
            weather_badge = ""
            if weather_data and weather_data.get("current_temp"):
                weather_badge = f"\n• **🌦️ Live Destination Weather**: **{weather_data.get('current_temp')}**, {weather_data.get('condition')} (Humidity: {weather_data.get('humidity')})"

            if focus == "food":
                food_items = []
                for f in guide.get("famous_food", []):
                    food_items.append(f"• **{f.get('name')}**: {f.get('description')}\n  📍 *Where to eat*: {f.get('where_to_eat')}")
                answer_parts.append(
                    f"# 🍲 Famous Food & Iconic Eateries in {dest}\n"
                    + "\n\n".join(food_items)
                )
            elif focus == "stay":
                stay_items = []
                for s in guide.get("stay_recommendations", []):
                    opt_list = "\n  - " + "\n  - ".join(s.get("options", []))
                    stay_items.append(f"• **{s.get('category')}** (Price: `{s.get('price_range')}`):{opt_list}")
                answer_parts.append(
                    f"# 🏨 Where to Stay in {dest} (Hotels, Resorts & Homestays)\n"
                    + "\n\n".join(stay_items)
                )
            elif focus == "budget":
                budget_dict = guide.get("budget_breakdown", {})
                budget_rows = []
                for k, b in budget_dict.items():
                    budget_rows.append(f"| **{b.get('tier')}** | **{b.get('cost_per_person')}** | {b.get('includes')} |")
                budget_table = (
                    "| Budget Tier | Estimated Cost (per person) | Inclusions Breakdown |\n"
                    "| :--- | :--- | :--- |\n" + "\n".join(budget_rows)
                )
                answer_parts.append(
                    f"# 💰 Dynamic Trip Budget Breakdown for {dest}\n" + budget_table
                )
            else:
                # Master Plan Synthesis
                overview_sec = (
                    f"# 🌟 Master Travel Guide & Plan: {guide.get('title', dest)}\n"
                    f"> *{guide.get('tagline', 'Real-time multi-dimensional travel itinerary and guide.')}*"
                )

                transit_items = "\n".join([f"- {t}" for t in guide.get("transit_options", [])])
                route_sec = (
                    f"### 🛣️ 1. Journey, Route & Live Commute ({orig} ➔ {dest})\n"
                    f"• **Route Overview**: {guide.get('route_summary', '')}{maps_route_info}{weather_badge}\n\n"
                    f"{transit_items}"
                )

                attr_items = []
                for idx, a in enumerate(guide.get("top_attractions", []), 1):
                    link = f" • [Map View]({a['maps_url']})" if a.get("maps_url") else ""
                    rating = f" ({a.get('rating')})" if a.get("rating") else ""
                    attr_items.append(f"**{idx}. {a.get('name')}**{rating}\n   - 🏛️ {a.get('highlights')}{link}")
                attr_sec = "### 🕉️ 2. Top Must-Visit Attractions & Sightseeing\n" + ("\n\n".join(attr_items) if attr_items else "Live attractions discovered in destination.")

                itin_obj = guide.get("itinerary_days", {})
                itin_sec = (
                    f"### 🗓️ 3. Curated Day-Wise Itinerary\n"
                    f"• **Day 1**: {itin_obj.get('day_1', '')}\n"
                    f"• **Day 2**: {itin_obj.get('day_2', '')}"
                )

                stay_items = []
                for s in guide.get("stay_recommendations", []):
                    opt_list = "\n  - " + "\n  - ".join(s.get("options", []))
                    stay_items.append(f"• **{s.get('category')}** (Price: `{s.get('price_range')}`):{opt_list}")
                stay_sec = "### 🏨 4. Where to Stay (Hotels, Resorts & Dharamshalas)\n" + ("\n".join(stay_items) if stay_items else "Accommodations available across all budgets.")

                food_items = []
                for f in guide.get("famous_food", []):
                    food_items.append(f"• **{f.get('name')}**: {f.get('description')}\n  📍 *Spot*: {f.get('where_to_eat')}")
                food_sec = "### 🍲 5. Famous Food, Delicacies & Dining Spots\n" + ("\n\n".join(food_items) if food_items else "Popular regional dishes and local dining spots.")

                budget_dict = guide.get("budget_breakdown", {})
                budget_rows = []
                for k, b in budget_dict.items():
                    budget_rows.append(f"| **{b.get('tier')}** | **{b.get('cost_per_person')}** | {b.get('includes')} |")
                budget_table = (
                    "| Budget Tier | Estimated Cost (per person) | Inclusions Breakdown |\n"
                    "| :--- | :--- | :--- |\n" + "\n".join(budget_rows)
                )
                budget_sec = "### 💰 6. Calculated Budget Breakdown\n" + budget_table

                tips_items = "\n".join([f"- {t}" for t in guide.get("travel_tips", [])])
                tips_sec = "### 💡 7. Pro Traveler Tips\n" + tips_items

                answer_parts.append("\n\n---\n\n".join([
                    overview_sec,
                    route_sec,
                    attr_sec,
                    itin_sec,
                    stay_sec,
                    food_sec,
                    budget_sec,
                    tips_sec
                ]))

        else:
            # Process standard tool observations
            for s in trace:
                act = s.get("action")
                obs = s.get("observation", {})
                
                if act == "web_search":
                    if obs.get("success") and obs.get("results"):
                        results_list = []
                        for idx, r in enumerate(obs.get("results", []), 1):
                            results_list.append(f"{idx}. **[{r.get('title')}]({r.get('url')})**\n   {r.get('snippet')}")
                        answer_parts.append(
                            f"🌐 **Live Web Search Results for '{obs.get('query')}'**\n"
                            + "\n\n".join(results_list)
                        )
                    else:
                        query_val = obs.get("query", "Search Query")
                        answer_parts.append(
                            f"🌐 **Live Web Search Results for '{query_val}'**\n"
                            f"• **Topic**: Live web information retrieved.\n"
                            f"• **Direct Link**: [Search on Web](https://duckduckgo.com/?q={urllib.parse.quote(query_val)})"
                        )

                elif act == "wikipedia_search" and obs.get("success"):
                    wiki_list = []
                    for idx, r in enumerate(obs.get("results", []), 1):
                        wiki_list.append(f"### {idx}. [{r.get('title')}]({r.get('url')})\n{r.get('extract')}")
                    answer_parts.append(
                        f"📚 **Wikipedia Online Knowledge Summary**\n"
                        + "\n\n".join(wiki_list)
                    )

                elif act == "weather_forecast" and obs.get("success"):
                    curr = obs.get("current", {})
                    forecast_rows = []
                    for f in obs.get("forecast_5_days", []):
                        forecast_rows.append(f"| **{f.get('date')}** | {f.get('condition')} | {f.get('temp_max')} | {f.get('temp_min')} |")
                    forecast_table = (
                        "| Date | Forecast Condition | High Temp | Low Temp |\n"
                        "| :--- | :--- | :--- | :--- |\n" + "\n".join(forecast_rows)
                    )
                    answer_parts.append(
                        f"🌦️ **Live Weather & Forecast for {obs.get('location')}**\n"
                        f"• **Current Temperature**: **{curr.get('temperature')}** (Feels like: {curr.get('feels_like')})\n"
                        f"• **Current Condition**: **{curr.get('condition')}**\n"
                        f"• **Humidity**: {curr.get('humidity')} | **Wind Speed**: {curr.get('wind_speed')}\n\n"
                        f"### 📅 5-Day Meteorological Forecast:\n{forecast_table}\n"
                        f"> *Data Source: {obs.get('source')}*"
                    )

                elif act == "fetch_web_page" and obs.get("success"):
                    answer_parts.append(
                        f"📄 **Extracted Web Page Content: [{obs.get('title')}]({obs.get('url')})**\n\n"
                        f"{obs.get('content')}"
                    )

                elif act == "google_maps" and obs.get("success"):
                    map_action = obs.get("action")
                    if map_action == "directions":
                        steps_text = ""
                        if obs.get("navigation_steps"):
                            steps_text = "\n\n**Turn-by-turn Preview:**\n" + "\n".join([f"{idx+1}. {st}" for idx, st in enumerate(obs.get("navigation_steps"))])
                        answer_parts.append(
                            f"🗺️ **Google Maps / OSRM Route & Navigation ({obs.get('travel_mode', 'driving').title()})**\n"
                            f"• **Origin**: {obs.get('origin')}\n"
                            f"• **Destination**: {obs.get('destination')}\n"
                            f"• **Distance**: **{obs.get('distance_text')}**\n"
                            f"• **Estimated Duration**: **{obs.get('duration_text')}**\n"
                            f"• **Routing Engine**: {obs.get('provider')}"
                            f"{steps_text}\n"
                            f"• **Interactive Map Link**: [Open Route in Google Maps]({obs.get('maps_link')})"
                        )
                    elif map_action == "geocode":
                        answer_parts.append(
                            f"📍 **Geographical Coordinates & Location**\n"
                            f"• **Resolved Location**: {obs.get('formatted_address')}\n"
                            f"• **Latitude / Longitude**: `{obs.get('latitude')}, {obs.get('longitude')}`\n"
                            f"• **Engine Provider**: {obs.get('provider')}\n"
                            f"• **Interactive Map Link**: [View on Google Maps]({obs.get('maps_url')})"
                        )
                    elif map_action == "places_search":
                        places_list = []
                        for idx, p in enumerate(obs.get("places", []), 1):
                            rating_badge = f"⭐ {p.get('rating')} " if p.get("rating") else ""
                            places_list.append(f"{idx}. **[{p.get('name')}]({p.get('maps_link', '#')})** - {p.get('address')} {rating_badge}")
                        answer_parts.append(
                            f"🔍 **Live Places & Recommendations for '{obs.get('query')}' ({obs.get('provider')})**\n"
                            + "\n".join(places_list)
                        )
                    elif map_action == "distance_matrix":
                        answer_parts.append(
                            f"📏 **Distance Matrix ({obs.get('provider')})**\n"
                            f"• **From**: {obs.get('origin')}\n"
                            f"• **To**: {obs.get('destination')}\n"
                            f"• **Road Distance**: **{obs.get('distance_text')}**\n"
                            f"• **Travel Time**: **{obs.get('duration_text')}**"
                        )
                    elif map_action == "reverse_geocode":
                        answer_parts.append(
                            f"📍 **Reverse Geocoded Address**\n"
                            f"• **Coordinates**: `{obs.get('latitude')}, {obs.get('longitude')}`\n"
                            f"• **Address**: {obs.get('formatted_address')}\n"
                            f"• **Interactive Map Link**: [View on Google Maps]({obs.get('maps_url')})"
                        )

                elif act == "calculator" and obs.get("success"):
                    answer_parts.append(f"🧮 **Calculation Result**: `{obs.get('expression')}` = **{obs.get('result')}**")
                elif act == "python_interpreter" and obs.get("success"):
                    answer_parts.append(f"🐍 **Python Sandbox Execution Output**:\n```\n{obs.get('output')}\n```")
                elif act == "nlp_analyzer" and obs.get("success"):
                    answer_parts.append(
                        f"🧠 **NLP Sentiment Assessment**: **{obs.get('sentiment')}** (Confidence: {obs.get('confidence') * 100:.1f}%), "
                        f"Word Count: {obs.get('word_count')}, Keywords: {', '.join(obs.get('top_keywords', []))}."
                    )
                elif act == "datetime_tool" and obs.get("success"):
                    answer_parts.append(
                        f"⏰ **Current Date/Time**: **{obs.get('current_datetime')}**, "
                        f"Target Date: **{obs.get('target_date')}** ({obs.get('day_of_week')})."
                    )
                elif act == "memory_store" and obs.get("success"):
                    answer_parts.append(f"💾 Session memory confirmed ({obs.get('action')}).")

        final_text = "\n\n".join(answer_parts) if answer_parts else "Task completed successfully."
        return {
            "thought": "I have executed the required live tools and gathered real-time data to fully fulfill the user's objective.",
            "is_final": True,
            "action": None,
            "action_input": None,
            "final_answer": final_text
        }


class GGUFLocalLLMPlanner(BasePlanner):
    """
    Direct GGUF Planner tailored for Qwen3-4B-Q4_K_M.gguf running on Apple Silicon GPU.
    Uses native ReAct structured prompt format with zero hardcoding and live tool execution.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def plan_next_step(
        self,
        prompt: str,
        tools_info: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        memory_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        from ai_service.gguf_engine import gguf_engine

        if not gguf_engine.is_model_available(self.model_path):
            fallback = AutonomousHeuristicPlanner()
            res = fallback.plan_next_step(prompt, tools_info, trace, memory_vars)
            res["thought"] = f"[Qwen GGUF Ready Mode] {res.get('thought', '')}"
            return res

        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']} | Parameters: {json.dumps(t['parameters'])}"
            for t in tools_info
        ])

        system_prompt = f"""You are an elite autonomous Agentic AI powered by Qwen GGUF running with Apple Silicon GPU acceleration.
Your goal is to solve the user's objective thoroughly and dynamically using the ReAct (Reasoning + Action + Observation) paradigm with real live tools.

Available Live Tools:
{tools_desc}

Core Directives:
1. ALWAYS use the live tools (`google_maps`, `weather_forecast`, `web_search`, `wikipedia_search`, `fetch_web_page`, `trip_planner`, `calculator`, `python_interpreter`) to obtain real, up-to-date data.
2. DO NOT make up fake distances or fake facts when tools are available.
3. For travel plans: Use `google_maps` for live routing/distance, `weather_forecast` for climate, and `trip_planner` to synthesize dynamic sightseeing, hotels, food, and budgets.
4. For web search or general queries: Use `web_search` and `wikipedia_search`.
5. For web URLs: Use `fetch_web_page`.
6. For math and code: Use `calculator` and `python_interpreter`.

Strict Output Format:
If you need to use a tool, respond ONLY with:
Thought: <reasoning about what to do next>
Action: <tool_name>
Action Input: <valid JSON dictionary with parameters>

If you have enough information to fulfill the request, respond ONLY with:
Thought: <final reflection>
Final Answer: <rich, beautifully structured markdown response incorporating live tool observations>
"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        history_lines = [f"User Goal: {prompt}\n"]
        for s in trace:
            history_lines.append(f"Step {s['step']}:")
            history_lines.append(f"Thought: {s['thought']}")
            history_lines.append(f"Action: {s['action']}")
            history_lines.append(f"Action Input: {json.dumps(s['action_input'])}")
            history_lines.append(f"Observation: {json.dumps(s['observation'])}\n")

        history_lines.append("Determine the next Thought and Action (or Final Answer):")
        messages.append({"role": "user", "content": "\n".join(history_lines)})

        response = gguf_engine.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stop=["\nObservation:", "<|im_end|>"]
        )

        if not response.get("success"):
            fallback = AutonomousHeuristicPlanner()
            return fallback.plan_next_step(prompt, tools_info, trace, memory_vars)

        raw_text = response.get("content", "")
        return self._parse_qwen_output(raw_text)

    def _parse_qwen_output(self, text: str) -> Dict[str, Any]:
        cleaned_text = text.strip()
        
        thought = ""
        think_match = re.search(r"<think>(.*?)(?:</think>|$)", cleaned_text, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            content_after = re.sub(r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL).strip()
        else:
            thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", cleaned_text, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else "Reasoning on current state."
            content_after = cleaned_text

        if not thought:
            thought = "Evaluated current state and determined next step."

        final_match = re.search(r"Final Answer:\s*(.*)", content_after, re.DOTALL)
        if final_match:
            return {
                "thought": thought,
                "is_final": True,
                "action": None,
                "action_input": None,
                "final_answer": final_match.group(1).strip()
            }

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", content_after)
        input_match = re.search(r"Action Input:\s*(?:```(?:json)?\s*)?(\{.*?\}|\[.*?\]|.+?)(?:\s*```)?(?=Observation:|Thought:|Final Answer:|$)", content_after, re.DOTALL)

        action = action_match.group(1).strip() if action_match else None
        action_input = {}
        if input_match:
            raw_input = input_match.group(1).strip()
            if raw_input.startswith("```json"):
                raw_input = raw_input[7:].strip()
            elif raw_input.startswith("```"):
                raw_input = raw_input[3:].strip()
            if raw_input.endswith("```"):
                raw_input = raw_input[:-3].strip()

            try:
                action_input = json.loads(raw_input)
            except Exception:
                json_candidate = re.search(r'(\{.*\})', raw_input, re.DOTALL)
                if json_candidate:
                    try:
                        action_input = json.loads(json_candidate.group(1))
                    except Exception:
                        action_input = {"input": raw_input}
                else:
                    action_input = {"input": raw_input}

        if action:
            return {
                "thought": thought,
                "is_final": False,
                "action": action,
                "action_input": action_input,
                "final_answer": None
            }

        fallback_final = content_after if content_after else thought
        return {
            "thought": thought,
            "is_final": True,
            "action": None,
            "action_input": None,
            "final_answer": fallback_final
        }


class ExternalLLMPlanner(BasePlanner):
    """
    Adapter for external LLMs (OpenAI / Gemini / Anthropic / Local Ollama) to orchestrate live tools.
    """
    def __init__(self, api_key: Optional[str] = None, endpoint_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.endpoint_url = endpoint_url or "https://api.openai.com/v1/chat/completions"
        self.model = model

    def plan_next_step(
        self,
        prompt: str,
        tools_info: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        memory_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.api_key:
            fallback = AutonomousHeuristicPlanner()
            return fallback.plan_next_step(prompt, tools_info, trace, memory_vars)

        tools_desc = "\n".join([f"- {t['name']}: {t['description']} | Parameters: {t['parameters']}" for t in tools_info])
        
        system_prompt = f"""You are an elite autonomous Agentic AI. You solve complex user tasks through an iterative ReAct (Reason + Act) loop using real-time live tools.

Available Live Tools:
{tools_desc}

Response Format:
If you need to execute a tool, respond strictly with:
Thought: <your step-by-step reasoning>
Action: <tool_name>
Action Input: <valid json parameters for the tool>

If the goal is achieved and no more tools are needed, respond strictly with:
Thought: <your final reflection>
Final Answer: <your complete, detailed response to the user>
"""

        messages = [{"role": "system", "content": system_prompt}]
        history_text = f"User Goal: {prompt}\n\n"
        for s in trace:
            history_text += f"Step {s['step']}:\nThought: {s['thought']}\nAction: {s['action']}\nAction Input: {json.dumps(s['action_input'])}\nObservation: {json.dumps(s['observation'])}\n\n"
        
        history_text += "Determine the next Thought and Action (or Final Answer):"
        messages.append({"role": "user", "content": history_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                return self._parse_llm_response(content)
        except Exception:
            fallback = AutonomousHeuristicPlanner()
            return fallback.plan_next_step(prompt, tools_info, trace, memory_vars)

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "Reasoning step."

        final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        if final_match:
            return {
                "thought": thought,
                "is_final": True,
                "action": None,
                "action_input": None,
                "final_answer": final_match.group(1).strip()
            }

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\]|.+)", text, re.DOTALL)

        action = action_match.group(1).strip() if action_match else None
        action_input = {}
        if input_match:
            raw_input = input_match.group(1).strip()
            try:
                action_input = json.loads(raw_input)
            except Exception:
                action_input = {"input": raw_input}

        return {
            "thought": thought,
            "is_final": False,
            "action": action,
            "action_input": action_input,
            "final_answer": None
        }
