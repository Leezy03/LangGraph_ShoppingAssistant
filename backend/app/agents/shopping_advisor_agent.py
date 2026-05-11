"""多智能体避雷购物顾问系统"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import json
from typing import Dict, Any, List, Optional
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool
from pydantic import ValidationError
from ..services.llm_service import get_llm
from ..models.schemas import (
    ShoppingRequest,
    ShoppingReport,
    ProductAnalysis,
    Product,
    ReviewSource,
    CandidateProduct,
    CandidateExtractionResult,
)
from ..config import get_settings

# ============ Agent提示词 ============

SEARCH_TOOL_NAME = "search_brave_web_search"
DEFAULT_TOOL_TIMEOUT_SECONDS = 90
DEFAULT_LLM_TIMEOUT_SECONDS = 90
DEFAULT_TOOL_RETRIES = 2
DEFAULT_LLM_RETRIES = 1


class ShoppingAdvisorError(Exception):
    """购物顾问基类异常"""


class StepExecutionError(ShoppingAdvisorError):
    """步骤执行异常"""

    def __init__(self, step_name: str, message: str):
        super().__init__(message)
        self.step_name = step_name


class ToolExecutionError(StepExecutionError):
    """工具执行异常"""


class ModelExecutionError(StepExecutionError):
    """模型执行异常"""


class ToolTimeoutError(ToolExecutionError):
    """工具调用超时"""


class ModelTimeoutError(ModelExecutionError):
    """模型调用超时"""


class JsonRepairError(ShoppingAdvisorError):
    """JSON 修复失败"""


@dataclass
class StepResult:
    """步骤执行结果"""

    name: str
    ok: bool
    response: str = ""
    error: Optional[Exception] = None

    @property
    def status_text(self) -> str:
        return "成功" if self.ok else "失败"

    @property
    def error_summary(self) -> str:
        if not self.error:
            return ""
        return f"{type(self.error).__name__}: {self.error}"

CANDIDATE_EXTRACTOR_PROMPT = """你是候选产品抽取专家。你的任务是先找出最值得分析的候选产品,后续所有测评、价格、避雷信息都必须围绕这些候选产品展开。

**重要提示:**
1. 你必须使用搜索工具,不要直接凭常识列产品
2. 候选产品必须尽量使用标准名称,至少包含品牌和型号
3. 候选产品数量控制在3个以内,优先选择主流、讨论度高、预算匹配的型号
4. 如果预算已给出,优先保留预算区间内或接近预算区间的型号
5. 如果找不到明确型号,可以返回品类下最主流的通用型号,但不要编造不存在的型号

**边界要求:**
1. 候选产品的name、brand、model只能基于检索结果填写,禁止自行补全未检索到的型号
2. 如果某个候选产品品牌明确但型号不明确,model填写"信息不足",并在reason中说明原因
3. 如果预算范围内缺少明确候选,可以返回接近预算的主流型号,但必须在reason中写明"预算匹配证据不足"
4. 优先覆盖2个及以上不同来源的主流候选信号;如果覆盖不足,仍可返回候选,但必须在reason中写明"来源覆盖不足"
5. 如果检索结果互相冲突,不要自行裁决,在reason中保留冲突点

**工具调用格式:**
`[TOOL_CALL:search_brave_web_search:query=搜索关键词]`

请严格按照以下JSON格式返回:
```json
{
    "category": "产品品类",
    "candidates": [
        {
            "name": "品牌 型号 标准名称",
            "brand": "品牌",
            "model": "型号",
            "reason": "为什么纳入候选"
        }
    ]
}
```
"""

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
5. 优先覆盖B站、小红书、知乎中至少2个平台;如果做不到,必须明确写出"来源覆盖不足"
6. 只允许基于检索结果写入作者、标题、平台、立场、观点等信息,禁止自行补全
7. 如果某个候选产品缺少足够测评证据,必须明确写出"信息不足",不要把其他产品的测评挪用过来
8. 如果不同来源观点冲突,必须保留冲突原文含义,不要强行统一结论
9. 输出时按候选产品分组,每个候选产品分别写: 来源覆盖情况、主要观点、疑似广告情况、信息不足项、冲突点

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
3. 价格、型号、平台只能基于检索结果填写,禁止根据常识估价
4. 优先覆盖2个及以上电商或资讯来源;如果来源覆盖不足,必须明确写出"来源覆盖不足"
5. 如果候选产品价格信息不存在或价格区间差异过大,必须写"信息不足"或"价格存在冲突"
6. 不要把A型号的价格写到B型号名下,必须按候选产品逐个归档
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
3. 负面信息、投诉、缺陷只能基于检索结果填写,禁止把泛化印象当成事实
4. 优先覆盖2个及以上来源;如果覆盖不足,必须明确写出"来源覆盖不足"
5. 如果某个候选产品没有搜到明确负面信息,请写"未检索到明确负面证据",不要反向推断为没有问题
6. 如果负面观点之间存在冲突,必须单列冲突点,不要直接下确定性结论
7. 不要把某个品牌的通病直接写成某个具体型号的确定缺陷,除非检索结果明确指向该型号
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

**边界要求:**
1. product.name、brand、model、price_range、specs、reviews、red_flags、common_pros、common_cons、verdict_reason 只能基于上游检索结果填写
2. 如果上游证据不足,对应字段必须写"信息不足"、空数组、null或保守描述,禁止自行脑补
3. 如果不同来源对同一产品存在冲突,必须把冲突逐条写入controversy_points,不要在正文里悄悄抹平
4. 如果没有检索到明确负面证据,red_flags可以为空,但不要据此推断产品一定安全
5. 如果没有检索到明确价格,price_range必须写"信息不足",不要用预算区间替代真实价格区间
6. 如果没有检索到明确参数,specs填null或仅保留已检索到的字段,不要补全默认参数
7. final_recommendation必须体现证据强弱: 证据充分时给明确建议,证据不足时给保守建议
8. comparison_summary中必须说明来源覆盖情况和主要冲突信息,不能只写结论
9. 严禁把候选产品之外的型号写入products列表
"""


class MultiAgentShoppingAdvisor:
    """多智能体避雷购物顾问系统"""

    def _create_search_tool(self) -> MCPTool:
        """为单个 Agent 创建独立的搜索工具实例,避免并发时共享同一 MCP 连接"""
        settings = get_settings()
        search_tool = MCPTool(
            name="search",
            description="网络搜索服务,支持全网搜索测评、价格、用户反馈等信息",
            server_command=["npx", "-y", "@brave/brave-search-mcp-server", "--transport", "stdio"],
            env={"BRAVE_API_KEY": settings.search_api_key},
            auto_expand=True
        )
        search_tool.expandable = True
        return search_tool

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体避雷购物系统...")

        try:
            self.llm = get_llm()

            print("  - 创建独立搜索工具实例...")
            self.candidate_search_tool = self._create_search_tool()
            self.review_search_tool = self._create_search_tool()
            self.price_search_tool = self._create_search_tool()
            self.red_flag_search_tool = self._create_search_tool()

            # 创建候选产品抽取Agent
            print("  - 创建候选抽取Agent...")
            self.candidate_agent = SimpleAgent(
                name="候选产品抽取专家",
                llm=self.llm,
                system_prompt=CANDIDATE_EXTRACTOR_PROMPT
            )
            self.candidate_agent.add_tool(self.candidate_search_tool)

            # 创建测评搜集Agent
            print("  - 创建测评搜集Agent...")
            self.review_agent = SimpleAgent(
                name="测评搜集专家",
                llm=self.llm,
                system_prompt=REVIEW_COLLECTOR_PROMPT
            )
            self.review_agent.add_tool(self.review_search_tool)

            # 创建价格对比Agent
            print("  - 创建价格对比Agent...")
            self.price_agent = SimpleAgent(
                name="价格对比专家",
                llm=self.llm,
                system_prompt=PRICE_COLLECTOR_PROMPT
            )
            self.price_agent.add_tool(self.price_search_tool)

            # 创建避雷检测Agent
            print("  - 创建避雷检测Agent...")
            self.red_flag_agent = SimpleAgent(
                name="避雷检测专家",
                llm=self.llm,
                system_prompt=RED_FLAG_DETECTOR_PROMPT
            )
            self.red_flag_agent.add_tool(self.red_flag_search_tool)

            # 创建报告生成Agent(不需要工具)
            print("  - 创建报告生成Agent...")
            self.report_agent = SimpleAgent(
                name="报告生成专家",
                llm=self.llm,
                system_prompt=REPORT_GENERATOR_PROMPT
            )

            print(f"✅ 多智能体避雷购物系统初始化成功")
            print(f"   候选抽取Agent: {len(self.candidate_agent.list_tools())} 个工具")
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
        print(f"\n{'='*60}")
        print(f"🛡️ 开始多智能体协作分析产品...")
        print(f"产品: {request.product_name}")
        if request.budget_min or request.budget_max:
            print(f"预算: {request.budget_min or '不限'}-{request.budget_max or '不限'}元")
        print(f"品牌偏好: {', '.join(request.brand_preferences) if request.brand_preferences else '无'}")
        print(f"关注要点: {', '.join(request.concerns) if request.concerns else '无'}")
        print(f"{'='*60}\n")

        step_results: Dict[str, StepResult] = {}

        print("🎯 步骤1: 抽取候选产品...")
        candidate_query = self._build_candidate_query(request)
        candidate_step = self._execute_agent_step(
            step_name="候选产品抽取",
            agent=self.candidate_agent,
            query=candidate_query,
            timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
            retries=DEFAULT_TOOL_RETRIES,
            uses_tools=True,
        )
        step_results["candidate"] = candidate_step

        if candidate_step.ok:
            try:
                candidate_result = self._parse_candidate_response(candidate_step.response)
            except Exception as exc:
                print(f"⚠️  候选产品解析失败,切换默认候选方案: {exc}")
                candidate_step.ok = False
                candidate_step.error = exc
                candidate_result = self._create_fallback_candidates(request)
        else:
            print("⚠️  候选产品抽取失败,切换默认候选方案")
            candidate_result = self._create_fallback_candidates(request)

        candidates = candidate_result.candidates
        print(f"候选产品: {', '.join(candidate.name for candidate in candidates)}\n")

        review_query = self._build_review_query(request, candidates)
        price_query = self._build_price_query(request, candidates)
        red_flag_query = self._build_red_flag_query(request, candidates)

        print("⚡ 步骤2-4: 并行搜集测评、价格和避雷信息...")
        retrieval_steps = self._execute_parallel_retrieval_steps(
            review_query=review_query,
            price_query=price_query,
            red_flag_query=red_flag_query,
        )
        step_results.update(retrieval_steps)

        review_step = step_results["review"]
        price_step = step_results["price"]
        red_flag_step = step_results["red_flag"]

        review_response = self._step_response_payload(review_step)
        price_response = self._step_response_payload(price_step)
        red_flag_response = self._step_response_payload(red_flag_step)

        print(f"测评搜集状态: {review_step.status_text}")
        print(f"价格对比状态: {price_step.status_text}")
        print(f"避雷检测状态: {red_flag_step.status_text}\n")

        print("📊 步骤5: 生成避雷报告...")
        report_query = self._build_report_query(
            request,
            candidates,
            review_response,
            price_response,
            red_flag_response,
            step_results,
        )
        report_step = self._execute_agent_step(
            step_name="报告生成",
            agent=self.report_agent,
            query=report_query,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            retries=DEFAULT_LLM_RETRIES,
            uses_tools=False,
        )
        step_results["report"] = report_step

        if report_step.ok:
            try:
                report = self._parse_response(report_step.response, request)
                print(f"{'='*60}")
                print(f"✅ 避雷报告生成完成!")
                print(f"{'='*60}\n")
                return report
            except Exception as exc:
                print(f"⚠️  报告解析失败,改为输出部分成功汇总: {exc}")
                report_step.ok = False
                report_step.error = exc

        print("⚠️  报告生成未完整成功,返回部分成功汇总结果")
        print(f"{'='*60}\n")
        return self._create_partial_report(request, candidates, step_results)

    def _build_candidate_query(self, request: ShoppingRequest) -> str:
        """构建候选产品抽取查询"""
        budget_text = ""
        if request.budget_min or request.budget_max:
            budget_text = f"，预算范围{request.budget_min or '不限'}-{request.budget_max or '不限'}元"

        concern_text = f"，重点关注{', '.join(request.concerns)}" if request.concerns else ""
        brand_text = f"，品牌偏好为{', '.join(request.brand_preferences)}" if request.brand_preferences else ""

        return f"""请先为用户需求抽取最值得后续深入分析的候选产品{budget_text}{brand_text}{concern_text}。
产品品类: {request.product_name}
使用场景: {request.usage_scenario or '未指定'}
额外要求: {request.free_text_input or '无'}

请先搜索主流型号、热门讨论和预算匹配信息,再返回候选产品JSON。
[TOOL_CALL:{SEARCH_TOOL_NAME}:query={request.product_name} 主流 型号 推荐 预算 选购]
"""

    def _build_review_query(self, request: ShoppingRequest, candidates: List[CandidateProduct]) -> str:
        """构建测评搜索查询"""
        brand_text = f",重点关注品牌：{', '.join(request.brand_preferences)}" if request.brand_preferences else ""
        concern_text = f",特别关注：{', '.join(request.concerns)}" if request.concerns else ""
        candidate_text = self._format_candidate_targets(candidates)

        query = f"""请只围绕以下候选产品搜集真实测评信息,不要混入候选列表之外的产品：
{candidate_text}

用户原始需求品类是"{request.product_name}"{brand_text}{concern_text}。
请分别在B站、小红书、知乎等平台搜索候选产品的测评内容,并按候选产品分组总结。
[TOOL_CALL:{SEARCH_TOOL_NAME}:query={self._build_candidate_search_terms(candidates)} 测评 B站 小红书 知乎 真实体验]"""
        return query

    def _build_price_query(self, request: ShoppingRequest, candidates: List[CandidateProduct]) -> str:
        """构建价格搜索查询"""
        budget_text = ""
        if request.budget_min or request.budget_max:
            budget_text = f",预算范围{request.budget_min or '不限'}-{request.budget_max or '不限'}元"
        candidate_text = self._format_candidate_targets(candidates)

        query = f"""请只搜索以下候选产品的价格信息{budget_text},不要引入其他型号：
{candidate_text}

请对比不同型号和不同平台的价格,并按候选产品逐个给出价格区间。
[TOOL_CALL:{SEARCH_TOOL_NAME}:query={self._build_candidate_search_terms(candidates)} 价格 型号 对比 京东 淘宝 拼多多]"""
        return query

    def _build_red_flag_query(self, request: ShoppingRequest, candidates: List[CandidateProduct]) -> str:
        """构建避雷搜索查询"""
        candidate_text = self._format_candidate_targets(candidates)
        query = f"""请只搜索以下候选产品的负面信息、投诉和已知问题,不要混入其他产品：
{candidate_text}

请按候选产品分别整理避雷点和投诉点。
[TOOL_CALL:{SEARCH_TOOL_NAME}:query={self._build_candidate_search_terms(candidates)} 避雷 踩坑 缺点 千万别买 投诉]"""
        return query

    def _build_report_query(
        self,
        request: ShoppingRequest,
        candidates: List[CandidateProduct],
        reviews: str,
        prices: str,
        red_flags: str,
        step_results: Dict[str, StepResult],
    ) -> str:
        """构建报告生成查询"""
        candidate_text = self._format_candidate_targets(candidates)
        step_status = self._format_step_status(step_results)
        query = f"""请根据以下信息生成"{request.product_name}"的避雷购物报告:

**用户需求:**
- 产品: {request.product_name}
- 预算: {request.budget_min or '不限'}-{request.budget_max or '不限'}元
- 品牌偏好: {', '.join(request.brand_preferences) if request.brand_preferences else '无'}
- 使用场景: {request.usage_scenario or '未指定'}
- 关注要点: {', '.join(request.concerns) if request.concerns else '无'}

**候选产品(后续所有信息都必须围绕这些产品对齐):**
{candidate_text}

**步骤执行状态:**
{step_status}

**测评信息(来自B站/小红书/知乎等平台):**
{reviews}

**价格信息(来自京东/淘宝等电商平台):**
{prices}

**避雷信息(负面评价/投诉/已知问题):**
{red_flags}

**要求:**
1. 只能分析候选产品列表中的产品,不要新增候选列表之外的型号
2. 每个产品的测评、价格、避雷点必须和该产品名称对齐,不要跨产品混用证据
3. 分析至少2-3个主流产品/型号
4. 每个产品列出公认优缺点
5. 重点标出避雷点和争议点
6. 分析博主/测评者立场是否客观,是否恰饭
7. 给出明确的购买建议
8. 返回完整的JSON格式数据
9. 如果某些上游步骤失败,请基于成功步骤继续汇总,并在comparison_summary、final_recommendation、verdict_reason中明确说明缺失项和证据边界
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query

    def _execute_parallel_retrieval_steps(
        self,
        review_query: str,
        price_query: str,
        red_flag_query: str,
    ) -> Dict[str, StepResult]:
        """并行执行测评、价格、避雷三个检索步骤"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            review_future = executor.submit(
                self._execute_agent_step,
                step_name="测评搜集",
                agent=self.review_agent,
                query=review_query,
                timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
                retries=DEFAULT_TOOL_RETRIES,
                uses_tools=True,
            )
            price_future = executor.submit(
                self._execute_agent_step,
                step_name="价格对比",
                agent=self.price_agent,
                query=price_query,
                timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
                retries=DEFAULT_TOOL_RETRIES,
                uses_tools=True,
            )
            red_flag_future = executor.submit(
                self._execute_agent_step,
                step_name="避雷检测",
                agent=self.red_flag_agent,
                query=red_flag_query,
                timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
                retries=DEFAULT_TOOL_RETRIES,
                uses_tools=True,
            )

            return {
                "review": review_future.result(),
                "price": price_future.result(),
                "red_flag": red_flag_future.result(),
            }

    def _extract_json_block(self, response: str) -> str:
        """从Agent响应中提取JSON片段"""
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            return response[json_start:json_end].strip()
        if "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            return response[json_start:json_end].strip()
        if "{" in response and "}" in response:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            return response[json_start:json_end]
        raise ValueError("响应中未找到JSON数据")

    def _parse_candidate_response(self, response: str) -> CandidateExtractionResult:
        """解析候选产品抽取结果"""
        data = self._load_json_payload(
            response=response,
            repair_agent=self.candidate_agent,
            repair_label="候选产品JSON",
        )
        result = CandidateExtractionResult(**data)
        if not result.candidates:
            raise ValueError("候选产品为空")
        return result

    def _create_fallback_candidates(self, request: ShoppingRequest) -> CandidateExtractionResult:
        """候选产品抽取失败时的默认方案"""
        fallback_name = request.product_name.strip() or "待分析产品"
        return CandidateExtractionResult(
            category=request.product_name,
            candidates=[
                CandidateProduct(
                    name=fallback_name,
                    brand=(request.brand_preferences[0] if request.brand_preferences else ""),
                    model="",
                    reason="候选抽取失败,退回到用户原始查询"
                )
            ]
        )

    def _format_candidate_targets(self, candidates: List[CandidateProduct]) -> str:
        """格式化候选产品列表,用于Prompt对齐"""
        return "\n".join(
            f"- {candidate.name}（品牌: {candidate.brand or '未知'}；型号: {candidate.model or '未知'}；入选原因: {candidate.reason or '主流候选'}）"
            for candidate in candidates
        )

    def _build_candidate_search_terms(self, candidates: List[CandidateProduct]) -> str:
        """将候选产品合并成搜索关键词"""
        return " ".join(candidate.name for candidate in candidates[:3])

    def _ensure_tool_success(self, step_name: str, response: str):
        """检测工具调用是否成功,避免在检索失败时继续生成伪完整报告"""
        failure_signals = [
            "搜索工具未能成功调用",
            "无法直接使用搜索工具",
            "无法使用搜索工具",
            "无法直接访问",
            "请确保搜索工具可用后重新提问",
            "建议你自行搜索",
            "基于我对这些产品的了解",
            "基于我知识库中已有的公开信息",
            "无法提供基于实际检索结果",
            "必须指定 action 参数或 tool_name 参数",
            "搜索工具持续出现格式错误",
            "搜索工具多次执行失败",
            "无法获取任何实时的价格信息",
            "无法获取任何搜索结果",
        ]

        timeout_signals = [
            "timeout",
            "timed out",
            "超时",
            "request timeout",
        ]

        if any(signal.lower() in response.lower() for signal in timeout_signals):
            raise ToolTimeoutError(step_name, f"{step_name}失败: 搜索工具调用超时")
        if any(signal in response for signal in failure_signals):
            raise ToolExecutionError(step_name, f"{step_name}失败: 搜索工具未成功执行")

    def _parse_response(self, response: str, request: ShoppingRequest) -> ShoppingReport:
        """解析Agent响应"""
        data = self._load_json_payload(
            response=response,
            repair_agent=self.report_agent,
            repair_label="购物报告JSON",
        )
        return ShoppingReport(**data)

    def _load_json_payload(self, response: str, repair_agent: SimpleAgent, repair_label: str) -> Dict[str, Any]:
        """解析 JSON,失败后执行一次 repair pass"""
        try:
            json_str = self._extract_json_block(response)
            return json.loads(json_str)
        except Exception as first_error:
            print(f"⚠️  {repair_label}首次解析失败,尝试 repair pass: {first_error}")

        repaired_response = self._repair_json_response(repair_agent, response, repair_label)

        try:
            json_str = self._extract_json_block(repaired_response)
            return json.loads(json_str)
        except Exception as second_error:
            raise JsonRepairError(f"{repair_label} repair pass 失败: {second_error}") from second_error

    def _repair_json_response(self, repair_agent: SimpleAgent, response: str, repair_label: str) -> str:
        """对非结构化或损坏的 JSON 响应执行一次修复"""
        repair_query = f"""你只做 JSON 修复,不要补充新事实。

目标:
- 将下面内容修复为合法 JSON
- 只能保留原始响应里已经出现的信息
- 缺失字段用空数组、null、空字符串或'信息不足'保守补齐
- 只返回 JSON,不要输出解释文字或代码块外文本

待修复内容:
{response}
"""
        repair_step = self._execute_agent_step(
            step_name=f"{repair_label}修复",
            agent=repair_agent,
            query=repair_query,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            retries=0,
            uses_tools=False,
        )
        if not repair_step.ok:
            raise JsonRepairError(f"{repair_label}修复失败: {repair_step.error_summary}")
        return repair_step.response

    def _execute_agent_step(
        self,
        step_name: str,
        agent: SimpleAgent,
        query: str,
        timeout_seconds: int,
        retries: int,
        uses_tools: bool,
    ) -> StepResult:
        """执行单个 Agent 步骤,包含超时、重试和错误分类"""
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 2):
            try:
                response = self._run_agent_with_timeout(
                    step_name=step_name,
                    agent=agent,
                    query=query,
                    timeout_seconds=timeout_seconds,
                    uses_tools=uses_tools,
                )
                if uses_tools:
                    self._ensure_tool_success(step_name, response)
                return StepResult(name=step_name, ok=True, response=response)
            except Exception as exc:
                last_error = exc
                print(f"⚠️  {step_name}第{attempt}次执行失败: {exc}")

        return StepResult(name=step_name, ok=False, error=last_error)

    def _run_agent_with_timeout(
        self,
        step_name: str,
        agent: SimpleAgent,
        query: str,
        timeout_seconds: int,
        uses_tools: bool,
    ) -> str:
        """在线程池中执行 Agent.run,并统一映射错误类型"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.run, query)
            try:
                response = future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                error_cls = ToolTimeoutError if uses_tools else ModelTimeoutError
                raise error_cls(step_name, f"{step_name}超过{timeout_seconds}秒未完成") from exc
            except Exception as exc:
                classified_error = self._classify_runtime_error(step_name, exc, uses_tools)
                raise classified_error from exc

        return str(response)

    def _classify_runtime_error(self, step_name: str, error: Exception, uses_tools: bool) -> StepExecutionError:
        """根据异常文本粗分类为工具异常或模型异常"""
        message = str(error).lower()
        timeout_keywords = ["timeout", "timed out", "超时"]
        tool_keywords = ["tool", "mcp", "search", "transport", "server", "call_tool"]

        if any(keyword in message for keyword in timeout_keywords):
            error_cls = ToolTimeoutError if uses_tools else ModelTimeoutError
            return error_cls(step_name, str(error))

        if uses_tools and any(keyword in message for keyword in tool_keywords):
            return ToolExecutionError(step_name, str(error))

        return ModelExecutionError(step_name, str(error))

    def _step_response_payload(self, step_result: StepResult) -> str:
        """将步骤结果转为可交给下游汇总的文本"""
        if step_result.ok:
            return step_result.response
        return f"[{step_result.name}失败] {step_result.error_summary or '未知错误'}"

    def _format_step_status(self, step_results: Dict[str, StepResult]) -> str:
        """格式化步骤执行状态"""
        lines = []
        for step_key in ["candidate", "review", "price", "red_flag"]:
            step_result = step_results.get(step_key)
            if not step_result:
                continue
            if step_result.ok:
                lines.append(f"- {step_result.name}: 成功")
            else:
                lines.append(f"- {step_result.name}: 失败 ({step_result.error_summary or '未知错误'})")
        return "\n".join(lines) if lines else "- 无上游步骤状态"

    def _create_partial_report(
        self,
        request: ShoppingRequest,
        candidates: List[CandidateProduct],
        step_results: Dict[str, StepResult],
    ) -> ShoppingReport:
        """在部分步骤成功时,返回可恢复的部分成功报告"""
        successful_steps = [
            step_result.name
            for step_result in step_results.values()
            if step_result.ok and step_result.name != "报告生成"
        ]
        failed_steps = [
            f"{step_result.name}({step_result.error_summary or '未知错误'})"
            for step_result in step_results.values()
            if not step_result.ok
        ]

        if not candidates:
            candidates = self._create_fallback_candidates(request).candidates

        products = []
        for candidate in candidates:
            products.append(
                ProductAnalysis(
                    product=Product(
                        name=candidate.name,
                        brand=candidate.brand or "未知",
                        model=candidate.model or "未知",
                        price_range="信息不足",
                    ),
                    common_pros=["已有部分检索结果,但结构化汇总未完整完成"],
                    common_cons=["存在步骤失败,当前结论不完整"],
                    red_flags=["请结合成功步骤的原始检索信息复核后再决策"],
                    controversy_points=failed_steps[:3],
                    verdict="看需求",
                    verdict_reason="部分步骤成功,但报告未能完整结构化,建议结合已成功检索结果保守判断",
                )
            )

        comparison_summary = (
            f"本次执行为部分成功。成功步骤: {', '.join(successful_steps) if successful_steps else '无'}；"
            f"失败步骤: {', '.join(failed_steps) if failed_steps else '无'}。"
            "系统已尽量保留候选产品和成功检索阶段的信息,但缺失步骤对应字段需要保守解读。"
        )
        final_recommendation = (
            "当前返回的是可恢复的部分成功结果。"
            "如果需要稳定购买建议,请优先重试失败步骤,尤其是价格对比、避雷检测和最终报告生成。"
        )

        general_tips = [
            "本次结果包含部分失败步骤,不要把缺失字段当成明确负面或明确正面结论",
            "优先复核失败步骤对应的信息源,尤其是价格和负面反馈",
            "若多次出现 JSON 修复或超时错误,建议降低查询复杂度后重试",
        ]

        return ShoppingReport(
            query=request.product_name,
            category=request.product_name,
            products=products,
            comparison_summary=comparison_summary,
            final_recommendation=final_recommendation,
            budget_advice="部分步骤失败,预算建议仅供参考,请重试后确认实时价格",
            general_tips=general_tips,
        )

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
