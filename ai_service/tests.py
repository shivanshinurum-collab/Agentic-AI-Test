import json
from django.test import TestCase, Client
from ai_service.gguf_engine import gguf_engine
from ai_service.agent import (
    CalculatorTool,
    PythonCodeTool,
    KnowledgeSearchTool,
    NLPAnalyzerTool,
    DateTimeTool,
    MemoryTool,
    GoogleMapsTool,
    TripPlannerTool,
    ToolRegistry,
    default_registry,
    agent_executor,
    MemoryManager,
    GGUFLocalLLMPlanner,
)

class AIAPITestCase(TestCase):
    def setUp(self):
        self.client = Client()

    # ----------------------------------------------------
    # System & Model Tests
    # ----------------------------------------------------
    def test_status_endpoint(self):
        response = self.client.get('/api/status/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('agentic_ai', {}).get('enabled'))

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
            data=json.dumps({'text': 'This Django backend with PyTorch is super fast and awesome!'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('sentiment'), 'POSITIVE')

    # ----------------------------------------------------
    # GGUF Model Engine Tests
    # ----------------------------------------------------
    def test_model_status_api(self):
        response = self.client.get('/api/model/status/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get('llama_cpp_installed'))
        self.assertEqual(data.get('target_model_name'), 'Qwen3-4B-Q4_K_M.gguf')

    def test_model_load_api_missing_file(self):
        response = self.client.post(
            '/api/model/load/',
            data=json.dumps({'model_path': 'non_existent_file.gguf'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data.get('success'))

    # ----------------------------------------------------
    # Agent Tools Unit Tests
    # ----------------------------------------------------
    def test_calculator_tool(self):
        calc = CalculatorTool()
        res = calc.execute(expression="1000 * (1 + 0.05)**2")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 1102.5)

        # Test functions
        res_sqrt = calc.execute(expression="sqrt(144) + 10")
        self.assertTrue(res_sqrt["success"])
        self.assertEqual(res_sqrt["result"], 22)

        # Test error handling
        res_err = calc.execute(expression="__import__('os').system('ls')")
        self.assertFalse(res_err["success"])

    def test_python_code_tool(self):
        py_tool = PythonCodeTool()
        res = py_tool.execute(code="nums = [1, 2, 3, 4]\nprint('Sum:', sum(nums))")
        self.assertTrue(res["success"])
        self.assertIn("Sum: 10", res["output"])

    def test_knowledge_search_tool(self):
        search = KnowledgeSearchTool()
        res = search.execute(query="Agentic AI and ReAct")
        self.assertTrue(res["success"])
        self.assertGreater(res["results_count"], 0)

    def test_nlp_analyzer_tool(self):
        nlp = NLPAnalyzerTool()
        res = nlp.execute(text="The AI execution performance is fantastic and lovely!")
        self.assertTrue(res["success"])
        self.assertEqual(res["sentiment"], "POSITIVE")

    def test_datetime_tool(self):
        dt = DateTimeTool()
        res = dt.execute(days_offset=7)
        self.assertTrue(res["success"])
        self.assertEqual(res["days_offset"], 7)
        self.assertIn("current_date", res)

    def test_memory_tool(self):
        mem = MemoryTool()
        set_res = mem.execute(action="set", key="test_key", value="test_val")
        self.assertTrue(set_res["success"])

        get_res = mem.execute(action="get", key="test_key")
        self.assertTrue(get_res["success"])
        self.assertEqual(get_res["value"], "test_val")

    def test_google_maps_tool_geocode(self):
        maps_tool = GoogleMapsTool()
        res = maps_tool.execute(action="geocode", query="Eiffel Tower Paris")
        self.assertTrue(res["success"])
        self.assertIn("latitude", res)
        self.assertIn("longitude", res)
        self.assertAlmostEqual(res["latitude"], 48.8584, places=1)
        self.assertAlmostEqual(res["longitude"], 2.2945, places=1)

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
        self.assertGreater(res["distance_km"], 50)

    def test_google_maps_tool_places_search(self):
        maps_tool = GoogleMapsTool()
        res = maps_tool.execute(action="places_search", query="cafes in Paris")
        self.assertTrue(res["success"])
        self.assertGreater(res["total_found"], 0)
        self.assertTrue(len(res["places"]) > 0)

    def test_trip_planner_tool(self):
        trip_tool = TripPlannerTool()
        res = trip_tool.execute(origin="Indore", destination="Ujjain", duration_days=2)
        self.assertTrue(res["success"])
        self.assertIn("guide", res)
        guide = res["guide"]
        self.assertIn("Mahakaleshwar", str(guide["top_attractions"]))
        self.assertIn("itinerary_1_day", guide)
        self.assertIn("stay_recommendations", guide)
        self.assertIn("famous_food", guide)
        self.assertIn("budget_breakdown", guide)

    # ----------------------------------------------------
    # Agent Executor ReAct Flow Tests
    # ----------------------------------------------------
    def test_agent_executor_trip_goal(self):
        result = agent_executor.run(
            prompt="indore to ujjain trip",
            session_id="test_ujjain_trip_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["total_steps"], 0)
        self.assertIn("Master Trip Plan", result["final_answer"])
        self.assertIn("Mahakaleshwar", result["final_answer"])
        self.assertIn("Dal Bafla", result["final_answer"])
        self.assertIn("Budget Breakdown", result["final_answer"])

    def test_agent_executor_maps_goal(self):
        result = agent_executor.run(
            prompt="Provide driving directions from San Francisco to San Jose",
            session_id="test_maps_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["total_steps"], 0)
        self.assertIn("Google Maps Route & Navigation", result["final_answer"])
        self.assertIn("San Francisco", result["final_answer"])

    def test_agent_executor_math_goal(self):
        result = agent_executor.run(
            prompt="Calculate sqrt(144) + 36",
            session_id="test_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["total_steps"], 0)
        self.assertIsNotNone(result["final_answer"])
        self.assertIn("48", result["final_answer"])

    def test_agent_executor_search_goal(self):
        result = agent_executor.run(
            prompt="Search for Agentic AI and ReAct details",
            session_id="test_search_session",
            planner_mode="heuristic"
        )
        self.assertEqual(result["status"], "completed")
        self.assertIn("Knowledge Search Insights", result["final_answer"])

    def test_gguf_planner_parsing(self):
        planner = GGUFLocalLLMPlanner()
        parsed = planner._parse_qwen_output(
            "<think>Need to calculate 25*4</think>\nAction: calculator\nAction Input: {\"expression\": \"25 * 4\"}"
        )
        self.assertEqual(parsed["action"], "calculator")
        self.assertEqual(parsed["action_input"]["expression"], "25 * 4")
        self.assertEqual(parsed["thought"], "Need to calculate 25*4")

    # ----------------------------------------------------
    # Agent API Endpoints Tests
    # ----------------------------------------------------
    def test_agent_tools_api(self):
        response = self.client.get('/api/agent/tools/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tools", data)
        self.assertGreaterEqual(data["count"], 6)

    def test_agent_run_api_success(self):
        response = self.client.post(
            '/api/agent/run/',
            data=json.dumps({
                'prompt': 'Calculate 25 * 4 and analyze sentiment of great',
                'max_iterations': 5,
                'planner_mode': 'heuristic'
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'completed')
        self.assertTrue(len(data.get('trace', [])) >= 1)

    def test_agent_run_api_empty_prompt(self):
        response = self.client.post(
            '/api/agent/run/',
            data=json.dumps({'prompt': ''}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_agent_dashboard_view(self):
        response = self.client.get('/api/agent/dashboard/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Agentic AI Platform")
        self.assertContains(response, "Qwen3-4B-Q4_K_M.gguf")
