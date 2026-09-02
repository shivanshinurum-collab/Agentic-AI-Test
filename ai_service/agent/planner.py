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
    Built-in Cognitive Planner that operates 100% offline with zero external API dependencies.
    It decomposes compound user goals into logical multi-step ReAct execution sequences.
    """
    def plan_next_step(
        self,
        prompt: str,
        tools_info: List[Dict[str, Any]],
        trace: List[Dict[str, Any]],
        memory_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        step_num = len(trace) + 1
        prompt_lower = prompt.lower()

        # Step 1: Detect intent and formulate initial thoughts
        if step_num == 1:
            # Check if user specifically asked for pure navigation/directions/distance
            is_pure_directions = any(w in prompt_lower for w in ["directions", "turn by turn", "navigation", "how far is", "distance between", "geocode", "coordinates"]) and not any(w in prompt_lower for w in ["trip", "itinerary", "vacation", "tour", "holiday", "darshan", "stay", "hotel", "food"])

            # 1. Check for Trip / Travel / Vacation / Itinerary inquiry (Highest priority for multi-aspect travel planning)
            trip_keywords = ["trip", "travel", "tour", "itinerary", "vacation", "holiday", "visit", "sightseeing", "darshan", "explore", "stay", "budget", "hotel", "food", "khana"]
            is_trip_inquiry = not is_pure_directions and (any(k in prompt_lower for k in trip_keywords) or bool(re.search(r'\b(?:to|in)\s+[a-zA-Z\s]+\s+trip\b', prompt_lower)))

            if is_trip_inquiry:
                orig = "Indore"
                dest = "Ujjain"
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

                # Detect user specific focus
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

                # Extract "X to Y" or "from X to Y" (exclude helper verbs/questions like "where to", "how to")
                orig = "Indore"
                dest = "Ujjain"
                ignore_words = ["where", "how", "want", "plan", "planning", "going", "need", "wish", "guide", "best", "what", "good"]
                action_words = ["stay", "eat", "visit", "go", "reach", "travel", "do", "see", "find", "get", "book"]

                from_to_match = re.search(r'\b(?:from\s+)?([a-zA-Z]{3,25})\s+(?:to|-)\s+([a-zA-Z]{3,25})\b', prompt, re.IGNORECASE)
                if from_to_match and from_to_match.group(1).lower() not in ignore_words and from_to_match.group(2).lower() not in action_words:
                    orig = from_to_match.group(1).strip().title()
                    dest = from_to_match.group(2).strip().title()
                else:
                    # Check known cities first for 100% precision
                    known_cities = ["ujjain", "banaras", "varanasi", "kashi", "ayodhya", "omkareshwar", "maheshwar", "jaipur", "agra", "goa", "delhi", "mumbai", "amritsar", "rishikesh", "haridwar", "manali", "shimla", "indore", "bhopal", "kolkata", "chennai", "bengaluru", "hyderabad", "pune", "ahmedabad", "chandigarh", "lucknow", "kanpur", "prayagraj"]
                    found_city = None
                    for c in known_cities:
                        if re.search(rf'\b{c}\b', prompt_lower):
                            found_city = c.title()
                            break
                    
                    if found_city:
                        dest = found_city
                        if found_city.lower() != "indore":
                            orig = "Indore"
                        else:
                            orig = "Bhopal"
                    else:
                        single_dest = re.search(r'(?:trip to|travel to|visit|tour of|guide for|hotels in|stay in|food in|budget for|in|of)\s+([a-zA-Z\s]+)', prompt, re.IGNORECASE)
                        if single_dest:
                            candidate = single_dest.group(1).strip().title()
                            candidate_clean = re.sub(r'\s+(trip|tour|guide|food|hotel|hotels|places|stay|budget|for|with|near|around|and|to|from).*', '', candidate, flags=re.IGNORECASE).strip()
                            if candidate_clean:
                                dest = candidate_clean

                # For comprehensive master trip queries, perform 2-step ReAct: Step 1 = Google Maps route & distance
                if focus == "all":
                    return {
                        "thought": f"The user is requesting a complete master travel plan for '{orig} ➔ {dest}'. First, I will query google_maps for live directions, route metrics, and transit distance.",
                        "is_final": False,
                        "action": "google_maps",
                        "action_input": {
                            "action": "directions",
                            "origin": orig,
                            "destination": dest,
                            "mode": "driving"
                        }
                    }
                else:
                    # For specific focus queries (e.g. food/hotels/budget), query trip_planner directly
                    return {
                        "thought": f"The user has a focused travel inquiry on '{focus}' for '{dest}' from '{orig}'. I will invoke the trip_planner tool with focus='{focus}'.",
                        "is_final": False,
                        "action": "trip_planner",
                        "action_input": {
                            "origin": orig,
                            "destination": dest,
                            "duration_days": days,
                            "travelers_count": travelers,
                            "focus": focus
                        }
                    }

            # 2. Check for Google Maps / Geocoding / Routing / Places inquiry
            map_keywords = [
                "map", "google map", "route", "directions", "distance", "how far", "navigate",
                "navigation", "geocode", "coordinates", "drive to", "drive from", "travel from",
                "places near", "restaurants in", "cafes in", "hotels in", "location of", "where is"
            ]
            is_map_inquiry = any(k in prompt_lower for k in map_keywords)

            if is_map_inquiry:
                # Directions / Route detection (e.g. "from X to Y" or "between X and Y")
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
                        "thought": f"The user is requesting route/navigation and distance from '{orig}' to '{dest}' using {mode} mode. I will query the google_maps tool.",
                        "is_final": False,
                        "action": "google_maps",
                        "action_input": {
                            "action": "directions",
                            "origin": orig,
                            "destination": dest,
                            "mode": mode
                        }
                    }

                # Places search detection (e.g., "cafes in Paris", "find hotels near Eiffel Tower")
                places_match = re.search(r'(?:find|search|show|top|best|nearby)?\s*(restaurants|cafes|coffee shops|hotels|bars|hospitals|pharmacies|banks|museums|parks)?\s*(?:in|near|around)\s+([^?.!]+)', prompt, re.IGNORECASE)
                if places_match and (places_match.group(1) or places_match.group(2)):
                    place_type = places_match.group(1) or "places"
                    location = places_match.group(2).strip() if places_match.group(2) else ""
                    search_query = f"{place_type} in {location}".strip()
                    return {
                        "thought": f"The user wants to find local places: '{search_query}'. I will use google_maps to search places.",
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
                    "thought": f"The user is looking for geographic location coordinates of '{geo_query}'. I will query google_maps geocode.",
                    "is_final": False,
                    "action": "google_maps",
                    "action_input": {
                        "action": "geocode",
                        "address": geo_query
                    }
                }

            # 3. Check for Math / Calculation
            math_match = re.search(r'[\d\.\s\+\-\*\/\^\(\)\%\,\√\=]+', prompt)
            calc_keywords = ["calculate", "math", "sqrt", "add", "multiply", "divide", "sum", "plus", "times", "minus", "expression", "eval", "evaluate"]
            if (math_match and len(math_match.group(0).strip()) > 3 and any(char.isdigit() for char in math_match.group(0))) or any(k in prompt_lower for k in calc_keywords):
                expr = prompt
                for prefix in ["calculate", "what is", "eval", "evaluate", "compute", "solve", "math"]:
                    if expr.lower().startswith(prefix):
                        expr = expr[len(prefix):].strip(" :?")
                return {
                    "thought": f"The user needs to evaluate a mathematical expression: '{expr}'. I will invoke the calculator tool.",
                    "is_final": False,
                    "action": "calculator",
                    "action_input": {"expression": expr}
                }

            # Check for Python / Code execution inquiry
            if any(w in prompt_lower for w in ["python", "code", "script", "loop", "fibonacci", "factorial", "algorithm", "array", "matrix"]):
                code = "def solve():\n    return [x**2 for x in range(1, 11)]\nprint('Squares 1-10:', solve())"
                if "fibonacci" in prompt_lower:
                    code = "def fib(n):\n    a, b = 0, 1\n    res = []\n    for _ in range(n):\n        res.append(a)\n        a, b = b, a + b\n    return res\nprint('Fibonacci sequence:', fib(10))"
                elif "factorial" in prompt_lower:
                    num = re.search(r'\d+', prompt)
                    n = num.group(0) if num else "5"
                    code = f"import math\nprint(f'Factorial of {n}: {{math.factorial({n})}}')"
                
                return {
                    "thought": "The request asks for Python logic or algorithmic execution. I will execute the script using the python_interpreter tool.",
                    "is_final": False,
                    "action": "python_interpreter",
                    "action_input": {"code": code}
                }

            # Check for Sentiment / NLP inquiry
            if any(w in prompt_lower for w in ["sentiment", "analyze text", "tone", "nlp", "opinion", "emotion"]):
                # Extract text target
                quote_match = re.search(r'["\']([^"\']+)["\']', prompt)
                text_to_eval = quote_match.group(1) if quote_match else prompt
                return {
                    "thought": f"I will perform sentiment and NLP analysis on: '{text_to_eval}'.",
                    "is_final": False,
                    "action": "nlp_analyzer",
                    "action_input": {"text": text_to_eval}
                }

            # Default to Knowledge Search for concepts/facts
            search_query = prompt.replace("search for", "").replace("what is", "").replace("who is", "").replace("tell me about", "").strip(" ?.")
            return {
                "thought": f"I need to search the knowledge base for relevant facts and information regarding: '{search_query}'.",
                "is_final": False,
                "action": "knowledge_search",
                "action_input": {"query": search_query or prompt}
            }

        # Step 2: Multi-step check or Final Answer Synthesis
        last_step = trace[-1]
        last_action = last_step.get("action")
        last_obs = last_step.get("observation", {})

        if step_num == 2:
            # If step 1 was google_maps (directions for a trip) AND prompt is a trip inquiry, step 2 retrieves deep trip_planner intelligence
            trip_keywords = ["trip", "travel", "tour", "itinerary", "vacation", "holiday", "visit", "sightseeing", "darshan", "explore", "stay", "budget", "hotel", "food", "khana"]
            is_trip_goal = any(k in prompt_lower for k in trip_keywords) or bool(re.search(r'\b(?:to|in)\s+[a-zA-Z\s]+\s+trip\b', prompt_lower))
            
            if is_trip_goal and last_action == "google_maps" and last_obs.get("action") == "directions":
                orig = last_obs.get("origin", "Indore")
                dest = last_obs.get("destination", "Ujjain")
                days = 2 if any(w in prompt_lower for w in ["2 day", "2-day", "two day", "weekend"]) else 1
                return {
                    "thought": f"I have gathered Google Maps route data ({last_obs.get('distance_text')}, {last_obs.get('duration_text')}). Now I will invoke trip_planner to retrieve top attractions, stays, food spots, itinerary, and budget breakdown.",
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
            elif "sentiment" in prompt_lower and last_action != "nlp_analyzer":
                quote_match = re.search(r'["\']([^"\']+)["\']', prompt)
                text_to_eval = quote_match.group(1) if quote_match else prompt
                return {
                    "thought": f"Now that step 1 ({last_action}) is complete, I will proceed to the sentiment and NLP analysis on: '{text_to_eval}'.",
                    "is_final": False,
                    "action": "nlp_analyzer",
                    "action_input": {"text": text_to_eval}
                }
            elif any(w in prompt_lower for w in ["save", "store", "remember", "memory"]) and last_action != "memory_store":
                result_val = str(last_obs.get("result", last_obs.get("output", "task_output")))
                return {
                    "thought": f"I will save the result from the previous step into session memory.",
                    "is_final": False,
                    "action": "memory_store",
                    "action_input": {"action": "set", "key": "last_calculation", "value": result_val}
                }

        # Synthesize final answer based on trace observations
        observations_summary = []
        for s in trace:
            step_idx = s.get("step")
            act = s.get("action")
            obs = s.get("observation")
            observations_summary.append(f"- Step {step_idx} ({act}): {json.dumps(obs)}")

        # Check if trip planning trace exists
        trip_obs = next((s.get("observation", {}) for s in trace if s.get("action") == "trip_planner" and s.get("observation", {}).get("success")), None)
        maps_obs = next((s.get("observation", {}) for s in trace if s.get("action") == "google_maps" and s.get("observation", {}).get("success")), None)

        answer_parts = []
        
        if trip_obs:
            guide = trip_obs.get("guide", {})
            dest = trip_obs.get("destination", "Destination")
            orig = trip_obs.get("origin", "Origin")
            focus = trip_obs.get("focus", "all")

            # Route distance metrics from google maps if available
            maps_route_info = ""
            if maps_obs and maps_obs.get("action") == "directions":
                maps_route_info = (
                    f"\n• **🗺️ Live Distance & Duration**: **{maps_obs.get('distance_text')}** (approx. **{maps_obs.get('duration_text')}** driving)\n"
                    f"• **Interactive Route Map**: [Open in Google Maps]({maps_obs.get('maps_link')})"
                )

            if focus == "food":
                food_items = []
                for f in guide.get("famous_food", []):
                    food_items.append(
                        f"• **{f.get('name')}**\n"
                        f"  - *Taste Profile*: {f.get('description')}\n"
                        f"  - 📍 *Iconic Spot*: **{f.get('where_to_eat')}**"
                    )
                answer_parts.append(
                    f"# 🍲 Famous Food & Iconic Street Food Spots in {dest}\n"
                    f"> *Authentic local delicacies, breakfast favorites, sweets, and prasad.*\n\n"
                    + "\n\n".join(food_items)
                )

            elif focus == "stay":
                stay_items = []
                for s in guide.get("stay_recommendations", []):
                    opt_list = "\n  - " + "\n  - ".join(s.get("options", []))
                    stay_items.append(f"• **{s.get('category')}** (Price: `{s.get('price_range')}`):{opt_list}")
                answer_parts.append(
                    f"# 🏨 Best Places to Stay in {dest} (Hotels, Resorts & Dharamshalas)\n"
                    f"> *Handpicked accommodations near prime landmarks & ghats across all budgets.*\n\n"
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
                    f"# 💰 Estimated Trip Budget Breakdown for {dest}\n"
                    f"{budget_table}"
                )

            elif focus == "route":
                transit_items = "\n".join([f"- {t}" for t in guide.get("transit_options", [])])
                answer_parts.append(
                    f"# 🛣️ Route & Commute Options: {orig} ➔ {dest}\n"
                    f"• **Overview**: {guide.get('route_summary', '')}{maps_route_info}\n\n"
                    f"### Multi-Modal Transit Breakdown:\n{transit_items}"
                )

            else:
                # Complete Master Plan
                overview_sec = (
                    f"# 🌟 Master Trip Plan & Complete Travel Guide: {guide.get('title', dest)}\n"
                    f"> *{guide.get('tagline', 'Your complete travel, sightseeing, stay, and culinary roadmap.')}*"
                )

                transit_items = "\n".join([f"- {t}" for t in guide.get("transit_options", [])])
                route_sec = (
                    f"### 🛣️ 1. Journey, Route & Commute Options ({orig} ➔ {dest})\n"
                    f"• **Route Overview**: {guide.get('route_summary', '')}{maps_route_info}\n"
                    f"{transit_items}"
                )

                attr_items = []
                for idx, a in enumerate(guide.get("top_attractions", []), 1):
                    attr_items.append(
                        f"**{idx}. {a.get('name')}**\n"
                        f"   - 🏛️ *Highlights*: {a.get('highlights')}\n"
                        f"   - ⏰ *Timings & Tickets*: {a.get('timings')}"
                    )
                attr_sec = "### 🕉️ 2. Top Must-Visit Attractions & Sightseeing\n" + "\n\n".join(attr_items)

                itin_1 = "\n".join([f"- {step}" for step in guide.get("itinerary_1_day", [])])
                itin_2_obj = guide.get("itinerary_2_day", {})
                itin_2 = f"• **Day 1**: {itin_2_obj.get('day_1', '')}\n• **Day 2**: {itin_2_obj.get('day_2', '')}"
                itin_sec = (
                    f"### 🗓️ 3. Curated Day-Wise Itinerary Timelines\n"
                    f"#### ⚡ Option A: 1-Day Express Itinerary (Morning to Night):\n{itin_1}\n\n"
                    f"#### 🌅 Option B: 2-Day Complete Leisure & Darshan Plan:\n{itin_2}"
                )

                stay_items = []
                for s in guide.get("stay_recommendations", []):
                    opt_list = "\n  - " + "\n  - ".join(s.get("options", []))
                    stay_items.append(f"• **{s.get('category')}** (Price: `{s.get('price_range')}`):{opt_list}")
                stay_sec = "### 🏨 4. Where to Stay (Hotels, Resorts & Dharamshalas)\n" + "\n".join(stay_items)

                food_items = []
                for f in guide.get("famous_food", []):
                    food_items.append(
                        f"• **{f.get('name')}**: {f.get('description')}\n"
                        f"  📍 *Where to Eat*: **{f.get('where_to_eat')}**"
                    )
                food_sec = "### 🍲 5. Famous Food, Local Delicacies & Street Food Spots\n" + "\n\n".join(food_items)

                budget_dict = guide.get("budget_breakdown", {})
                budget_rows = []
                for k, b in budget_dict.items():
                    budget_rows.append(f"| **{b.get('tier')}** | **{b.get('cost_per_person')}** | {b.get('includes')} |")
                budget_table = (
                    "| Budget Tier | Estimated Cost (per person) | Inclusions Breakdown |\n"
                    "| :--- | :--- | :--- |\n" + "\n".join(budget_rows)
                )
                budget_sec = "### 💰 6. Estimated Budget Breakdown\n" + budget_table

                tips_items = "\n".join([f"- {t}" for t in guide.get("travel_tips", [])])
                tips_sec = "### 💡 7. Pro Traveler Tips & Darshan Guidelines\n" + tips_items

                full_guide_text = "\n\n---\n\n".join([
                    overview_sec,
                    route_sec,
                    attr_sec,
                    itin_sec,
                    stay_sec,
                    food_sec,
                    budget_sec,
                    tips_sec
                ])
                answer_parts.append(full_guide_text)

        else:
            # Process standard tools observations
            for s in trace:
                act = s.get("action")
                obs = s.get("observation", {})
                if act == "calculator" and obs.get("success"):
                    answer_parts.append(f"Calculation Result for `{obs.get('expression')}` = **{obs.get('result')}**.")
                elif act == "python_interpreter" and obs.get("success"):
                    answer_parts.append(f"Python Execution Output:\n```\n{obs.get('output')}\n```")
                elif act == "knowledge_search" and obs.get("success"):
                    if obs.get("results"):
                        details = "\n".join([f"• **{r['title']}**: {r['content']}" for r in obs['results']])
                        answer_parts.append(f"Knowledge Search Insights:\n{details}")
                    else:
                        answer_parts.append(f"Search result: {obs.get('message', 'Information retrieved.')}")
                elif act == "nlp_analyzer" and obs.get("success"):
                    answer_parts.append(
                        f"NLP & Sentiment Assessment: **{obs.get('sentiment')}** (Confidence: {obs.get('confidence') * 100:.1f}%), "
                        f"Word Count: {obs.get('word_count')}, Top Keywords: {', '.join(obs.get('top_keywords', []))}."
                    )
                elif act == "datetime_tool" and obs.get("success"):
                    answer_parts.append(
                        f"Temporal Details: Current Date/Time: **{obs.get('current_datetime')}**, "
                        f"Target Date: **{obs.get('target_date')}** ({obs.get('day_of_week')})."
                    )
                elif act == "google_maps" and obs.get("success"):
                    map_action = obs.get("action")
                    if map_action == "directions":
                        steps_text = ""
                        if obs.get("navigation_steps"):
                            steps_text = "\n\n**Turn-by-turn Preview:**\n" + "\n".join([f"{idx+1}. {st}" for idx, st in enumerate(obs.get("navigation_steps"))])
                        answer_parts.append(
                            f"🗺️ **Google Maps Route & Navigation ({obs.get('travel_mode', 'driving').title()})**\n"
                            f"• **Origin**: {obs.get('origin')}\n"
                            f"• **Destination**: {obs.get('destination')}\n"
                            f"• **Distance**: **{obs.get('distance_text')}**\n"
                            f"• **Estimated Duration**: **{obs.get('duration_text')}**"
                            f"{steps_text}\n"
                            f"• **Interactive Map Link**: [Open Route in Google Maps]({obs.get('maps_link')})"
                        )
                    elif map_action == "geocode":
                        answer_parts.append(
                            f"📍 **Geographical Coordinates & Location Details**\n"
                            f"• **Resolved Location**: {obs.get('formatted_address')}\n"
                            f"• **Latitude / Longitude**: `{obs.get('latitude')}, {obs.get('longitude')}`\n"
                            f"• **Engine Provider**: {obs.get('provider')}\n"
                            f"• **Interactive Map Link**: [View on Google Maps]({obs.get('maps_url')})"
                        )
                    elif map_action == "places_search":
                        places_list = []
                        for idx, p in enumerate(obs.get("places", []), 1):
                            rating_badge = f"⭐ {p.get('rating')}" if p.get("rating") else ""
                            reviews_badge = f"({p.get('user_ratings_total')} reviews)" if p.get("user_ratings_total") else ""
                            open_badge = " • 🟢 Open Now" if p.get("open_now") is True else ""
                            places_list.append(f"{idx}. **{p.get('name')}** - {p.get('address')} {rating_badge} {reviews_badge}{open_badge}")
                        places_summary = "\n".join(places_list)
                        answer_parts.append(
                            f"🔍 **Top Places & Local Recommendations for '{obs.get('query')}'**\n"
                            f"{places_summary}"
                        )
                    elif map_action == "distance_matrix":
                        answer_parts.append(
                            f"📏 **Distance Matrix Result**\n"
                            f"• **From**: {obs.get('origin')}\n"
                            f"• **To**: {obs.get('destination')}\n"
                            f"• **Distance**: **{obs.get('distance_text')}**\n"
                            f"• **Travel Time**: **{obs.get('duration_text')}**"
                        )
                    elif map_action == "reverse_geocode":
                        answer_parts.append(
                            f"📍 **Reverse Geocoding Address**\n"
                            f"• **Coordinates**: `{obs.get('latitude')}, {obs.get('longitude')}`\n"
                            f"• **Address**: {obs.get('formatted_address')}\n"
                            f"• **Interactive Map Link**: [View on Google Maps]({obs.get('maps_url')})"
                        )
                elif act == "memory_store" and obs.get("success"):
                    answer_parts.append(f"Memory update confirmed ({obs.get('action')}).")

        if not answer_parts:
            final_text = f"Task completed successfully across {len(trace)} steps. Summary:\n" + "\n".join(observations_summary)
        else:
            final_text = "\n\n".join(answer_parts)

        return {
            "thought": "I have collected all necessary observations and tools outputs to fully resolve the user's objective.",
            "is_final": True,
            "action": None,
            "action_input": None,
            "final_answer": final_text
        }


class GGUFLocalLLMPlanner(BasePlanner):
    """
    Direct GGUF Planner tailored for Qwen3-4B-Q4_K_M.gguf running on Apple Silicon GPU.
    Uses native ReAct structured prompt format with zero latency.
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

        # Check if model file is available
        if not gguf_engine.is_model_available(self.model_path):
            # Graceful fallback to heuristic planner if file is not yet placed
            fallback = AutonomousHeuristicPlanner()
            res = fallback.plan_next_step(prompt, tools_info, trace, memory_vars)
            res["thought"] = f"[Qwen GGUF Ready Mode] {res.get('thought', '')}"
            return res

        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']} | Parameters: {json.dumps(t['parameters'])}"
            for t in tools_info
        ])

        system_prompt = f"""You are an elite autonomous Agentic AI powered by Qwen GGUF running with Apple Silicon GPU acceleration.
Your goal is to solve the user's request thoroughly and exhaustively using the ReAct (Reasoning + Action + Observation) paradigm.

Available Tools:
{tools_desc}

Core Directives & Behavioral Standards:
1. TRIP & TRAVEL INQUIRIES: When the user asks for a trip plan, vacation, tour, itinerary, or visit to any city/temple (e.g. 'indore to ujjain trip', 'plan trip to Goa'):
   - Use `trip_planner` (and/or `google_maps`) to retrieve full travel details.
   - In your Final Answer, you MUST provide an exhaustive, multi-dimensional Master Travel Guide covering ALL of these aspects:
     a) 🛣️ Route & Commute Options (Driving distance, Train, Intercity Bus, Cab fares)
     b) 🕉️ Top Sightseeing & Must-Visit Attractions (with timings and highlights)
     c) 🗓️ Day-wise Itinerary (1-Day Express & 2-Day Complete Plan)
     d) 🏨 Where to Stay (Luxury, Mid-Range, Budget & Dharamshala recommendations with prices)
     e) 🍲 Famous Local Food & Iconic Eateries (Signature dishes, street food, sweets, prasad)
     f) 💰 Comprehensive Budget Breakdown (Budget, Moderate, Luxury tiers)
     g) 💡 Pro Traveler Tips (Aarti/Darshan booking, dress codes, luggage lockers, best season)
   - NEVER give a short 1-line answer for travel/trip questions. Provide the complete master itinerary.

Strict Output Format:
If you need to use a tool, respond ONLY with:
Thought: <detailed reasoning about what to do next>
Action: <tool_name>
Action Input: <valid JSON dictionary with parameters>

If you have enough information to fulfill the request, respond ONLY with:
Thought: <final reasoning and reflection>
Final Answer: <complete, rich, beautifully structured markdown answer with all details>
"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Construct conversational trace
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
            # Fallback on runtime failure
            fallback = AutonomousHeuristicPlanner()
            return fallback.plan_next_step(prompt, tools_info, trace, memory_vars)

        raw_text = response.get("content", "")
        return self._parse_qwen_output(raw_text)

    def _parse_qwen_output(self, text: str) -> Dict[str, Any]:
        cleaned_text = text.strip()
        
        # 1. Extract thought from <think>...</think> or Thought: ...
        thought = ""
        think_match = re.search(r"<think>(.*?)(?:</think>|$)", cleaned_text, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            # Content outside <think>
            content_after = re.sub(r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL).strip()
        else:
            thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", cleaned_text, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else "Reasoning on current state."
            content_after = cleaned_text

        if not thought:
            thought = "Evaluated current state and determined next step."

        # 2. Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.*)", content_after, re.DOTALL)
        if final_match:
            return {
                "thought": thought,
                "is_final": True,
                "action": None,
                "action_input": None,
                "final_answer": final_match.group(1).strip()
            }

        # 3. Check for Action and Action Input
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

        # If action was found, return action step
        if action:
            return {
                "thought": thought,
                "is_final": False,
                "action": action,
                "action_input": action_input,
                "final_answer": None
            }

        # If no explicit action was emitted, use content_after or thought as final answer
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
    Adapter for external LLMs (OpenAI / Gemini / Anthropic / Local Ollama).
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
        # If no API key configured, fallback safely to AutonomousHeuristicPlanner
        if not self.api_key:
            fallback = AutonomousHeuristicPlanner()
            return fallback.plan_next_step(prompt, tools_info, trace, memory_vars)

        tools_desc = "\n".join([f"- {t['name']}: {t['description']} | Parameters: {t['parameters']}" for t in tools_info])
        
        system_prompt = f"""You are an elite autonomous Agentic AI. You solve complex user tasks through an iterative ReAct (Reason + Act) loop.

Available Tools:
{tools_desc}

Core Directives:
1. TRIP & TRAVEL INQUIRIES: When the user asks for a trip plan, vacation, tour, itinerary, or visit to any city/temple (e.g. 'indore to ujjain trip', 'plan trip to Goa'):
   - Use `trip_planner` (and/or `google_maps`) to retrieve full travel details.
   - In your Final Answer, you MUST provide an exhaustive, multi-dimensional Master Travel Guide covering ALL 7 aspects:
     1. 🛣️ Route & Commute Options (Distance, Train, Intercity Bus, Cab fares)
     2. 🕉️ Top Sightseeing & Must-Visit Attractions (with timings and highlights)
     3. 🗓️ Day-wise Itinerary (1-Day Express & 2-Day Complete Plan)
     4. 🏨 Where to Stay (Luxury, Mid-Range, Budget & Dharamshalas with prices)
     5. 🍲 Famous Food & Iconic Eateries (Dal Bafla, Street food, Sweets, Prasad)
     6. 💰 Comprehensive Budget Breakdown (Budget, Moderate, Luxury)
     7. 💡 Pro Traveler Tips (Aarti/Darshan booking, dress codes, lockers, best season)
   - NEVER give a short 1-line answer for travel/trip questions.

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
            # Fallback to local heuristic planner on network/API failure
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
