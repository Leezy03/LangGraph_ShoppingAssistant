import json
import time

from app.agents.shopping_advisor_agent import MultiAgentShoppingAdvisor


class StubAgent:
    def __init__(self, responses, delay=0.0):
        self.responses = list(responses)
        self.delay = delay
        self.calls = 0

    def run(self, query):
        if self.delay:
            time.sleep(self.delay)
        index = self.calls if self.calls < len(self.responses) else len(self.responses) - 1
        self.calls += 1
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response


def build_candidate_json(name="Apple iPhone 15"):
    return (
        "```json\n"
        + json.dumps(
            {
                "category": "手机",
                "candidates": [
                    {
                        "name": name,
                        "brand": "Apple",
                        "model": "iPhone 15",
                        "reason": "主流候选",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )


def build_report_json(name="Apple iPhone 15", comparison_summary="对比完成", final_recommendation="建议按需求选择"):
    return (
        "```json\n"
        + json.dumps(
            {
                "query": "手机",
                "category": "手机",
                "products": [
                    {
                        "product": {
                            "name": name,
                            "brand": "Apple",
                            "model": "iPhone 15",
                            "price_range": "信息不足",
                            "rating": None,
                            "image_url": None,
                            "specs": None,
                        },
                        "reviews": [],
                        "common_pros": ["手感稳定"],
                        "common_cons": ["价格偏高"],
                        "red_flags": ["未检索到明确负面证据"],
                        "controversy_points": [],
                        "verdict": "看需求",
                        "verdict_reason": "测试用例",
                    }
                ],
                "comparison_summary": comparison_summary,
                "final_recommendation": final_recommendation,
                "budget_advice": None,
                "general_tips": ["多平台交叉验证"],
            },
            ensure_ascii=False,
        )
        + "\n```"
    )


def build_test_advisor(candidate_responses=None, review_responses=None, price_responses=None, red_flag_responses=None, report_responses=None, delays=None):
    delays = delays or {}
    advisor = MultiAgentShoppingAdvisor.__new__(MultiAgentShoppingAdvisor)
    advisor.candidate_agent = StubAgent(candidate_responses or [build_candidate_json()], delay=delays.get("candidate", 0.0))
    advisor.review_agent = StubAgent(review_responses or ["测评成功"], delay=delays.get("review", 0.0))
    advisor.price_agent = StubAgent(price_responses or ["价格成功"], delay=delays.get("price", 0.0))
    advisor.red_flag_agent = StubAgent(red_flag_responses or ["避雷成功"], delay=delays.get("red_flag", 0.0))
    advisor.report_agent = StubAgent(report_responses or [build_report_json()], delay=delays.get("report", 0.0))
    return advisor