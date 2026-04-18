"""多智能体避雷购物顾问系统"""

import json
from typing import Dict, Any, List
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool
from ..services.llm_service import get_llm
from ..models.schemas import ShoppingRequest, ShoppingReport, ProductAnalysis, Product, ReviewSource
from ..config import get_settings

# ============ Agent提示词 ============

REVIEW_COLLECTOR_PROMPT = """你是购物测评搜集专家。你的任务是使用搜索工具查找各平台（B站、小红书、知乎）对指定产品的真实测评和用户反馈。

**重要提示:**
你必须使用搜索工具来查找测评！不要自己编造信息！

**搜索策略:**
1. 搜索"[产品名] B站 测评 真实体验"
2. 搜索"[产品名] 小红书 避雷 踩坑"
3. 搜索"[产品名] 知乎 推荐 评价"
4. 搜索"[产品名] 缺点 吐槽"

**工具调用格式:**
使用brave_web_search工具时,必须严格按照以下格式:
`[TOOL_CALL:search_brave_web_search:query=搜索关键词]`

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 参数用逗号分隔
4. 请尽可能搜索多个角度的信息
"""

PRICE_COLLECTOR_PROMPT = """你是价格对比专家。你的任务是使用搜索工具查找产品在各大电商平台的价格信息和型号对比。

**重要提示:**
你必须使用搜索工具来查找价格！不要自己编造价格！

**搜索策略:**
1. 搜索"[产品名] 价格 京东 淘宝"
2. 搜索"[产品名] 型号 参数 对比"
3. 搜索"[产品名] 哪个型号性价比高"

**工具调用格式:**
`[TOOL_CALL:search_brave_web_search:query=搜索关键词]`

**注意:**
1. 必须使用工具,不要直接回答
2. 关注不同型号的价格区间和性价比
"""

RED_FLAG_DETECTOR_PROMPT = """你是产品避雷专家。你的任务是使用搜索工具专门查找产品的负面信息、投诉、已知缺陷和恰饭测评。

**重要提示:**
你必须使用搜索工具来查找避雷信息！不要自己编造！

**搜索策略:**
1. 搜索"[产品名] 缺点 问题 质量"
2. 搜索"[产品名] 投诉 售后 差评"
3. 搜索"[产品名] 翻车 踩坑 千万别买"
4. 搜索"[产品名] 恰饭 广告 软文"

**工具调用格式:**
`[TOOL_CALL:search_brave_web_search:query=搜索关键词]`

**注意:**
1. 必须使用工具,不要直接回答
2. 特别关注：质量问题、售后体验、虚假宣传、恰饭测评
"""

REPORT_GENERATOR_PROMPT = """你是避雷购物报告撰写专家。你的任务是根据测评信息、价格信息和避雷信息,生成结构化的购物避雷报告。

请严格按照以下JSON格式返回报告:
```json
{
  "query": "用户查询的产品",
  "category": "产品品类",
  "products": [
    {
      "product": {
        "name": "产品全名",
        "brand": "品牌",
        "model": "型号",
        "price_range": "1000-2000元",
        "rating": 8.5,
        "image_url": null,
        "specs": {"关键参数1": "值1", "关键参数2": "值2"}
      },
      "reviews": [
        {
          "platform": "B站",
          "author": "博主名称",
          "title": "测评标题",
          "url": null,
          "stance": "推荐",
          "is_sponsored": false,
          "key_points": ["观点1", "观点2"],
          "credibility_score": 8.0
        }
      ],
      "common_pros": ["公认优点1", "公认优点2"],
      "common_cons": ["公认缺点1", "公认缺点2"],
      "red_flags": ["避雷点1", "避雷点2"],
      "controversy_points": ["争议点1"],
      "verdict": "推荐",
      "verdict_reason": "综合评价理由"
    }
  ],
  "comparison_summary": "横向对比总结",
  "final_recommendation": "最终购买建议",
  "budget_advice": "预算建议",
  "general_tips": ["选购建议1", "选购建议2"]
}
```

**重要提示:**
1. 必须分析每个博主/测评者的立场是否客观
2. 标记疑似恰饭(广告)内容,is_sponsored设为true
3. 对比不同信息源的矛盾之处,记录在controversy_points中
4. 给出明确的verdict: "推荐" / "不推荐" / "看需求"
5. red_flags(避雷点)是最重要的部分,要重点列出
6. credibility_score根据博主专业度、是否恰饭、内容详实度评判(1-10分)
7. 如果用户指定了预算范围,给出具体的预算建议
8. 分析至少2-3个主流产品/型号
9. general_tips给出该品类的通用选购建议
"""


class MultiAgentShoppingAdvisor:
    """多智能体避雷购物顾问系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体避雷购物系统...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            # 创建共享的搜索MCP工具
            print("  - 创建共享搜索工具...")
            self.search_tool = MCPTool(
                name="search",
                description="网络搜索服务,支持全网搜索测评、价格、用户反馈等信息",
                server_command=["npx", "-y", "@anthropic/mcp-server-brave-search"],
                env={"BRAVE_API_KEY": settings.search_api_key},
                auto_expand=True
            )
            self.search_tool.expandable = True

            # 创建测评搜集Agent
            print("  - 创建测评搜集Agent...")
            self.review_agent = SimpleAgent(
                name="测评搜集专家",
                llm=self.llm,
                system_prompt=REVIEW_COLLECTOR_PROMPT
            )
            self.review_agent.add_tool(self.search_tool)

            # 创建价格对比Agent
            print("  - 创建价格对比Agent...")
            self.price_agent = SimpleAgent(
                name="价格对比专家",
                llm=self.llm,
                system_prompt=PRICE_COLLECTOR_PROMPT
            )
            self.price_agent.add_tool(self.search_tool)

            # 创建避雷检测Agent
            print("  - 创建避雷检测Agent...")
            self.red_flag_agent = SimpleAgent(
                name="避雷检测专家",
                llm=self.llm,
                system_prompt=RED_FLAG_DETECTOR_PROMPT
            )
            self.red_flag_agent.add_tool(self.search_tool)

            # 创建报告生成Agent(不需要工具)
            print("  - 创建报告生成Agent...")
            self.report_agent = SimpleAgent(
                name="报告生成专家",
                llm=self.llm,
                system_prompt=REPORT_GENERATOR_PROMPT
            )

            print(f"✅ 多智能体避雷购物系统初始化成功")
            print(f"   测评搜集Agent: {len(self.review_agent.list_tools())} 个工具")
            print(f"   价格对比Agent: {len(self.price_agent.list_tools())} 个工具")
            print(f"   避雷检测Agent: {len(self.red_flag_agent.list_tools())} 个工具")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def analyze_product(self, request: ShoppingRequest) -> ShoppingReport:
        """
        使用多智能体协作分析产品

        Args:
            request: 购物避雷请求

        Returns:
            购物避雷报告
        """
        try:
            print(f"\n{'='*60}")
            print(f"🛡️ 开始多智能体协作分析产品...")
            print(f"产品: {request.product_name}")
            if request.budget_min or request.budget_max:
                print(f"预算: {request.budget_min or '不限'}-{request.budget_max or '不限'}元")
            print(f"品牌偏好: {', '.join(request.brand_preferences) if request.brand_preferences else '无'}")
            print(f"关注要点: {', '.join(request.concerns) if request.concerns else '无'}")
            print(f"{'='*60}\n")

            # 步骤1: 测评搜集Agent搜索各平台测评
            print("📝 步骤1: 搜集各平台测评...")
            review_query = self._build_review_query(request)
            review_response = self.review_agent.run(review_query)
            print(f"测评搜集结果: {review_response[:200]}...\n")

            # 步骤2: 价格对比Agent搜索价格信息
            print("💰 步骤2: 对比价格信息...")
            price_query = self._build_price_query(request)
            price_response = self.price_agent.run(price_query)
            print(f"价格对比结果: {price_response[:200]}...\n")

            # 步骤3: 避雷检测Agent搜索负面信息
            print("🚩 步骤3: 检测避雷信息...")
            red_flag_query = self._build_red_flag_query(request)
            red_flag_response = self.red_flag_agent.run(red_flag_query)
            print(f"避雷检测结果: {red_flag_response[:200]}...\n")

            # 步骤4: 报告生成Agent整合所有信息
            print("📊 步骤4: 生成避雷报告...")
            report_query = self._build_report_query(
                request, review_response, price_response, red_flag_response
            )
            report_response = self.report_agent.run(report_query)
            print(f"报告生成结果: {report_response[:300]}...\n")

            # 解析最终报告
            report = self._parse_response(report_response, request)

            print(f"{'='*60}")
            print(f"✅ 避雷报告生成完成!")
            print(f"{'='*60}\n")

            return report

        except Exception as e:
            print(f"❌ 生成避雷报告失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_report(request)

    def _build_review_query(self, request: ShoppingRequest) -> str:
        """构建测评搜索查询"""
        brand_text = f",重点关注品牌：{', '.join(request.brand_preferences)}" if request.brand_preferences else ""
        concern_text = f",特别关注：{', '.join(request.concerns)}" if request.concerns else ""

        query = f"""请搜索"{request.product_name}"的真实测评信息{brand_text}{concern_text}。
请分别在B站、小红书、知乎等平台搜索测评内容。
[TOOL_CALL:search_brave_web_search:query={request.product_name} 测评 B站 小红书 真实体验]"""
        return query

    def _build_price_query(self, request: ShoppingRequest) -> str:
        """构建价格搜索查询"""
        budget_text = ""
        if request.budget_min or request.budget_max:
            budget_text = f",预算范围{request.budget_min or '不限'}-{request.budget_max or '不限'}元"

        query = f"""请搜索"{request.product_name}"的价格信息{budget_text}。
对比不同型号和不同平台的价格。
[TOOL_CALL:search_brave_web_search:query={request.product_name} 价格 型号 对比 京东 淘宝]"""
        return query

    def _build_red_flag_query(self, request: ShoppingRequest) -> str:
        """构建避雷搜索查询"""
        query = f"""请搜索"{request.product_name}"的负面信息、投诉和已知问题。
[TOOL_CALL:search_brave_web_search:query={request.product_name} 避雷 踩坑 缺点 千万别买 投诉]"""
        return query

    def _build_report_query(self, request: ShoppingRequest, reviews: str, prices: str, red_flags: str) -> str:
        """构建报告生成查询"""
        query = f"""请根据以下信息生成"{request.product_name}"的避雷购物报告:

**用户需求:**
- 产品: {request.product_name}
- 预算: {request.budget_min or '不限'}-{request.budget_max or '不限'}元
- 品牌偏好: {', '.join(request.brand_preferences) if request.brand_preferences else '无'}
- 使用场景: {request.usage_scenario or '未指定'}
- 关注要点: {', '.join(request.concerns) if request.concerns else '无'}

**测评信息(来自B站/小红书/知乎等平台):**
{reviews}

**价格信息(来自京东/淘宝等电商平台):**
{prices}

**避雷信息(负面评价/投诉/已知问题):**
{red_flags}

**要求:**
1. 分析至少2-3个主流产品/型号
2. 每个产品列出公认优缺点
3. 重点标出避雷点和争议点
4. 分析博主/测评者立场是否客观,是否恰饭
5. 给出明确的购买建议
6. 返回完整的JSON格式数据
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query

    def _parse_response(self, response: str, request: ShoppingRequest) -> ShoppingReport:
        """解析Agent响应"""
        try:
            # 尝试从响应中提取JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到JSON数据")

            data = json.loads(json_str)
            report = ShoppingReport(**data)
            return report

        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成报告")
            return self._create_fallback_report(request)

    def _create_fallback_report(self, request: ShoppingRequest) -> ShoppingReport:
        """创建备用报告(当Agent失败时)"""
        return ShoppingReport(
            query=request.product_name,
            category=request.product_name,
            products=[
                ProductAnalysis(
                    product=Product(
                        name=f"{request.product_name}（待分析）",
                        brand="未知",
                        model="未知",
                        price_range="暂无数据"
                    ),
                    common_pros=["暂无数据,请稍后重试"],
                    common_cons=["暂无数据,请稍后重试"],
                    red_flags=["分析服务暂时不可用,请稍后重试"],
                    controversy_points=[],
                    verdict="待定",
                    verdict_reason="由于服务异常,暂时无法给出建议,请稍后重试"
                )
            ],
            comparison_summary="分析服务暂时不可用,请稍后重试",
            final_recommendation="请稍后重试,或手动搜索相关测评信息进行判断",
            general_tips=[
                "建议多看不同博主的测评,交叉验证",
                "注意区分恰饭内容和真实测评",
                "重点关注售后评价和长期使用体验",
                "不要只看好评,差评往往更有参考价值"
            ]
        )


# 全局多智能体系统实例
_shopping_advisor = None


def get_shopping_advisor() -> MultiAgentShoppingAdvisor:
    """获取多智能体购物顾问实例(单例模式)"""
    global _shopping_advisor

    if _shopping_advisor is None:
        _shopping_advisor = MultiAgentShoppingAdvisor()

    return _shopping_advisor
