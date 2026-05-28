import time

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import shopping
from app.models.schemas import Product, ProductAnalysis, ShoppingReport
from app.services.task_manager import task_store


class FakeAdvisor:
    def __init__(self, report=None, error=None):
        self.report = report
        self.error = error

    def analyze_product(self, request, tracer=None):
        if self.error:
            raise self.error
        if tracer:
            event_id = tracer.start_step("candidate", "候选产品抽取")
            tracer.finish_step(
                event_id=event_id,
                step_key="candidate",
                step_name="候选产品抽取",
                ok=True,
                message="候选产品抽取完成",
            )
        return self.report


def build_api_report():
    return ShoppingReport(
        query="手机",
        category="手机",
        products=[
            ProductAnalysis(
                product=Product(name="Apple iPhone 15", brand="Apple", model="iPhone 15", price_range="信息不足"),
                common_pros=["体验稳定"],
                common_cons=["价格偏高"],
                red_flags=["未检索到明确负面证据"],
                controversy_points=[],
                verdict="看需求",
                verdict_reason="测试返回",
            )
        ],
        comparison_summary="对比完成",
        final_recommendation="建议按需求选择",
        budget_advice=None,
        general_tips=["多平台交叉验证"],
    )


def test_analyze_api_returns_success_response(monkeypatch):
    import app.api.main as main_module

    monkeypatch.setattr(main_module, "validate_config", lambda: True)
    monkeypatch.setattr(main_module, "print_config", lambda: None)
    monkeypatch.setattr(shopping, "get_shopping_advisor", lambda: FakeAdvisor(report=build_api_report()))

    with TestClient(app) as client:
        response = client.post(
            "/api/shopping/analyze",
            json={"product_name": "手机", "brand_preferences": [], "usage_scenario": "", "concerns": [], "free_text_input": ""},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["query"] == "手机"


def test_analyze_api_returns_500_on_unhandled_error(monkeypatch):
    import app.api.main as main_module

    monkeypatch.setattr(main_module, "validate_config", lambda: True)
    monkeypatch.setattr(main_module, "print_config", lambda: None)
    monkeypatch.setattr(shopping, "get_shopping_advisor", lambda: FakeAdvisor(error=RuntimeError("boom")))

    with TestClient(app) as client:
        response = client.post(
            "/api/shopping/analyze",
            json={"product_name": "手机", "brand_preferences": [], "usage_scenario": "", "concerns": [], "free_text_input": ""},
        )

    assert response.status_code == 500
    assert "生成避雷报告失败" in response.json()["detail"]


def test_analysis_task_status_and_trace(monkeypatch):
    import app.api.main as main_module

    task_store.clear()
    monkeypatch.setattr(main_module, "validate_config", lambda: True)
    monkeypatch.setattr(main_module, "print_config", lambda: None)
    monkeypatch.setattr(shopping, "get_shopping_advisor", lambda: FakeAdvisor(report=build_api_report()))

    with TestClient(app) as client:
        response = client.post(
            "/api/shopping/tasks",
            json={"product_name": "手机", "brand_preferences": [], "usage_scenario": "", "concerns": [], "free_text_input": ""},
        )

        assert response.status_code == 200
        task_id = response.json()["task_id"]

        task_body = None
        for _ in range(30):
            task_response = client.get(f"/api/shopping/tasks/{task_id}")
            assert task_response.status_code == 200
            task_body = task_response.json()
            if task_body["status"] in {"succeeded", "partial", "failed"}:
                break
            time.sleep(0.05)

        assert task_body is not None
        assert task_body["status"] == "succeeded"
        assert task_body["progress"] == 100
        assert task_body["report"]["query"] == "手机"

        trace_response = client.get(f"/api/shopping/tasks/{task_id}/trace")
        assert trace_response.status_code == 200
        trace_body = trace_response.json()
        assert trace_body["success"] is True
        assert trace_body["trace"][0]["step_key"] == "candidate"
        assert trace_body["trace"][0]["duration_ms"] is not None
