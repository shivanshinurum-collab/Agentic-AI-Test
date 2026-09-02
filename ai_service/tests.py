import json
from django.test import TestCase, Client
from ai_service.gguf_engine import gguf_engine
from ai_service.agent import (
    CalculatorTool,
    PythonCodeTool,
    WebSearchTool,
    WikipediaTool,
    WeatherTool,
    WebFetcherTool,
    GoogleMapsTool,
    TripPlannerTool,
    KnowledgeSearchTool,
    NLPAnalyzerTool,
    DateTimeTool,
    MemoryTool,
    ToolRegistry,
    default_registry,
    agent_executor,
    MemoryManager,
    GGUFLocalLLMPlanner,
)

class RealAgenticAITestCase(TestCase):
    def setUp(self):
        self.client = Client()

    # ----------------------------------------------------
    # 1. System & Engine Tests
    # ----------------------------------------------------
    def test_status_endpoint(self):
        response = self.client.get('/api/status/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('agentic_ai', {}).get('enabled'))
        self.assertGreaterEqual(data.get('agentic_ai', {}).get('tools_registered'), 8)

    def test_benchmark_endpoint(self):
        response = self.client.post(
            '/api/benchmark/',
            data=json.dumps({'matrix_size': 100}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('matrix_shape'), [100, 100])

    def test_predict_endpoint(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({'text': 'This Agentic AI framework with live multi-platform tools is excellent!'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('sentiment'), 'POSITIVE')

    # ----------------------------------------------------
    # 2. Live Tools Unit Tests
    # ----------------------------------------------------
    def test_calculator_tool(self):
        calc = CalculatorTool()
        res = calc.execute(expression="1000 * (1 + 0.05)**2")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 1102.5)

        res_sqrt = calc.execute(expression="sqrt(144) + 10")
        self.assertTrue(res_sqrt["success"])
        self.assertEqual(res_sqrt["result"], 22)

    def test_python_code_tool(self):
        py_tool = PythonCodeTool()
        res = py_tool.execute(code="nums = [1, 2, 3, 4, 5]\nprint('Sum:', sum(nums))")
        self.assertTrue(res["success"])
        self.assertIn("Sum: 15", res["output"])

    def test_web_search_tool(self):
        search = WebSearchTool()
        res = search.execute(query="Artificial Intelligence", max_results=3)
        self.assertTrue(res["success"])
        self.assertGreater(res["total_results"], 0)
        self.assertIn("title", res["results"][0])

    def test_wikipedia_tool(self):
        wiki = WikipediaTool()
        res = wiki.execute(query="Python (programming language)", limit=2)
        self.assertTrue(res["success"])
        self.assertGreater(res["results_count"], 0)
        self.assertIn("Python", res["results"][0]["title"])

    def test_weather_tool(self):
        weather = WeatherTool()
        res = weather.execute(location="Bhopal")
        self.assertTrue(res["success"])
        self.assertIn("current", res)
        self.assertIn("temperature", res["current"])
        self.assertIn("forecast_5_days", res)

    def test_google_maps_tool_geocode(self):
        maps_tool = GoogleMapsTool()
        res = maps_tool.execute(action="geocode", query="Eiffel Tower Paris")
        self.assertTrue(res["success"])
        self.assertIn("latitude", res)
        self.assertIn("longitude", res)
        self.assertAlmostEqual(res["latitude"], 48.85, delta=0.1)

    def test_google_maps_tool_directions(self):
        maps_tool = GoogleMapsTool()
        res = maps_tool.execute(
            action="directions",
            origin="San Francisco",
            destination="San Jose",
            mode="driving"
        )
        self.assertTrue(res["success"])
        self.assertIn("distance_km", res)
        self.assertIn("duration_text", res)
        self.assertGreater(res["distance_km"], 40)

    def test_google_maps_tool_places_search(self):
        maps_tool = GoogleMapsTool()
        res = maps_tool.execute(action="places_search", query="cafes in Paris")
        self.assertTrue(res["success"])
        self.assertGreater(res["total_found"], 0)

    def test_dynamic_trip_planner_tool_arbitrary_city(self):
        trip_tool = TripPlannerTool()
        # Test on an arbitrary global destination without hardcoded data
        res = trip_tool.execute(origin="Delhi", destination="Shimla", duration_days=2, travelers_count=2)
        self.assertTrue(res["success"])
        self.assertEqual(res["destination"], "Shimla")
        guide = res["guide"]
        self.assertIn("Shimla", guide["title"])
        self.assertIn("route_summary", guide)
        self.assertIn("live_weather", guide)
        self.assertIn("budget_breakdown", guide)
        self.assertIn("stay_recommendations", guide)

    def test_nlp_analyzer_tool(self):
        nlp = NLPAnalyzerTool()
        res = nlp.execute(text="The autonomous agentic AI performed wonderfully!")
        self.assertTrue(res["success"])
        self.assertEqual(res["sentiment"], "POSITIVE")

    def test_datetime_tool(self):
        dt = DateTimeTool()
        res = dt.execute(days_offset=3)
        self.assertTrue(res["success"])
        self.assertEqual(res["days_offset"], 3)
        self.assertIn("current_date", res)

    def test_memory_tool(self):
        mem = MemoryTool()
        set_res = mem.execute(action="set", key="user_goal", value="autonomous execution")
        self.assertTrue(set_res["success"])
        get_res = mem.execute(action="get", key="user_goal")
        self.assertTrue(get_res["success"])
        self.assertEqual(get_res["value"], "autonomous execution")

    # ----------------------------------------------------
    # 3. Agent Executor ReAct Workflow Tests
    # ----------------------------------------------------
    def test_agent_executor_weather_goal(self):
        result = agent_executor.run(
            prompt="What is the live weather and temperature in Paris?",
            session_id="test_weather_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["total_steps"], 0)
        self.assertIn("Live Weather", result["final_answer"])

    def test_agent_executor_travel_goal(self):
        result = agent_executor.run(
            prompt="Plan a 2-day trip from Indore to Pachmarhi",
            session_id="test_pachmarhi_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["total_steps"], 0)
        self.assertIn("Master Travel Guide", result["final_answer"])
        self.assertIn("Pachmarhi", result["final_answer"])
        self.assertIn("Budget Breakdown", result["final_answer"])

    def test_agent_executor_web_search_goal(self):
        result = agent_executor.run(
            prompt="Search for Quantum Computing fundamentals",
            session_id="test_web_search_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue("Live Web Search" in result["final_answer"] or "Knowledge" in result["final_answer"] or "Quantum" in result["final_answer"])

    def test_agent_executor_navigation_goal(self):
        result = agent_executor.run(
            prompt="Directions from Mumbai to Pune",
            session_id="test_nav_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertIn("Route & Navigation", result["final_answer"])
        self.assertIn("Mumbai", result["final_answer"])

    def test_agent_executor_math_goal(self):
        result = agent_executor.run(
            prompt="Calculate 450 * 12 + sqrt(144)",
            session_id="test_math_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertIn("5412", result["final_answer"])

    # ----------------------------------------------------
    # 4. API Endpoints Tests
    # ----------------------------------------------------
    def test_agent_tools_api(self):
        response = self.client.get('/api/agent/tools/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tools", data)
        self.assertGreaterEqual(data["count"], 8)
        tool_names = [t["name"] for t in data["tools"]]
        self.assertIn("google_maps", tool_names)
        self.assertIn("weather_forecast", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertIn("wikipedia_search", tool_names)
        self.assertIn("trip_planner", tool_names)

    def test_agent_run_api_live(self):
        response = self.client.post(
            '/api/agent/run/',
            data=json.dumps({
                'prompt': 'Calculate 50 * 8 and get current date',
                'max_iterations': 4,
                'planner_mode': 'heuristic'
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'completed')
        self.assertTrue(len(data.get('trace', [])) >= 1)

    def test_chat_and_dashboard_views(self):
        dashboard_res = self.client.get('/api/agent/dashboard/')
        self.assertEqual(dashboard_res.status_code, 200)

        chat_res = self.client.get('/chat/')
        self.assertEqual(chat_res.status_code, 200)

        root_res = self.client.get('/')
        self.assertEqual(root_res.status_code, 200)
