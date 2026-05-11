from app.agents.shopping_advisor_agent import JsonRepairError, ModelTimeoutError, ToolTimeoutError

from tests.helpers import StubAgent, build_report_json


def test_retry_succeeds_on_second_attempt(advisor_factory, shopping_request):
    advisor = advisor_factory(review_responses=[RuntimeError("mcp tool unavailable"), "测评成功"])

    report = advisor.analyze_product(shopping_request)

    assert report.query == "手机"
    assert advisor.review_agent.calls == 2


def test_tool_timeout_is_classified_correctly(advisor_factory):
    advisor = advisor_factory()

    step_result = advisor._execute_agent_step(
        step_name="价格对比",
        agent=StubAgent([RuntimeError("timeout while fetching prices")]),
        query="价格查询",
        timeout_seconds=1,
        retries=0,
        uses_tools=True,
    )

    assert step_result.ok is False
    assert isinstance(step_result.error, ToolTimeoutError)


def test_model_timeout_is_classified_correctly(advisor_factory):
    advisor = advisor_factory(delays={"report": 0.2})

    try:
        advisor._run_agent_with_timeout(
            step_name="报告生成",
            agent=advisor.report_agent,
            query="生成报告",
            timeout_seconds=0.01,
            uses_tools=False,
        )
    except Exception as exc:
        assert isinstance(exc, ModelTimeoutError)
    else:
        raise AssertionError("Expected ModelTimeoutError to be raised")


def test_partial_success_returns_partial_report(advisor_factory, shopping_request):
    advisor = advisor_factory(report_responses=[RuntimeError("llm unavailable"), RuntimeError("llm unavailable")])

    report = advisor.analyze_product(shopping_request)

    assert "部分成功" in report.comparison_summary
    assert "可恢复" in report.final_recommendation


def test_json_repair_pass_recovers_invalid_json(advisor_factory, shopping_request):
    advisor = advisor_factory(
        report_responses=[
            '{"query": "手机"',
            build_report_json(final_recommendation="repair 成功后可继续返回"),
        ]
    )

    report = advisor.analyze_product(shopping_request)

    assert report.final_recommendation == "repair 成功后可继续返回"
    assert advisor.report_agent.calls == 2


def test_json_repair_pass_raises_error_when_repair_fails(advisor_factory):
    advisor = advisor_factory(report_responses=['{"query": "手机"', '{"query": "手机"'])

    try:
        advisor._parse_response('{"query": "手机"', None)
    except Exception as exc:
        assert isinstance(exc, JsonRepairError)
    else:
        raise AssertionError("Expected JsonRepairError to be raised")
