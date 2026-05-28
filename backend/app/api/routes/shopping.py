"""购物避雷API路由"""

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from ...models.schemas import (
    ShoppingRequest,
    ShoppingReportResponse,
    ShoppingAnalysisTaskStatus,
    ShoppingTaskCreateResponse,
    ShoppingTaskTraceResponse,
    ErrorResponse
)
from ...agents.shopping_advisor_agent import get_shopping_advisor
from ...services.task_manager import TaskTracer, task_store

router = APIRouter(prefix="/shopping", tags=["购物避雷"])
task_executor = ThreadPoolExecutor(max_workers=4)


def _run_shopping_analysis_task(task_id: str, request: ShoppingRequest):
    """后台执行购物分析任务,并持续写入任务状态与 trace。"""
    try:
        task_store.mark_running(task_id, "正在初始化多 Agent 工作流")
        tracer = TaskTracer(task_id=task_id, store=task_store)
        advisor = get_shopping_advisor()
        report = advisor.analyze_product(request, tracer=tracer)
        task_store.complete_task(task_id, report, status=tracer.final_status())
    except Exception as e:
        print(f"❌ 后台购物分析任务失败: {task_id} - {str(e)}")
        import traceback
        traceback.print_exc()
        task_store.fail_task(task_id, str(e))


@router.post(
    "/analyze",
    response_model=ShoppingReportResponse,
    summary="生成避雷购物报告",
    description="根据用户输入的产品需求,使用多Agent协作分析B站/小红书/知乎真实测评,生成深度避雷购物报告"
)
async def analyze_product(request: ShoppingRequest):
    """
    生成避雷购物报告

    Args:
        request: 购物避雷请求参数

    Returns:
        购物避雷报告响应
    """
    try:
        print(f"\n{'='*60}")
        print(f"📥 收到购物避雷请求:")
        print(f"   产品: {request.product_name}")
        if request.budget_min or request.budget_max:
            print(f"   预算: {request.budget_min or '不限'}-{request.budget_max or '不限'}元")
        print(f"   关注: {', '.join(request.concerns) if request.concerns else '无'}")
        print(f"{'='*60}\n")

        # 获取Agent实例
        print("🔄 获取多智能体系统实例...")
        advisor = get_shopping_advisor()

        # 生成避雷报告
        print("🛡️ 开始生成避雷报告...")
        report = advisor.analyze_product(request)

        print("✅ 避雷报告生成成功,准备返回响应\n")

        return ShoppingReportResponse(
            success=True,
            message="避雷报告生成成功",
            data=report
        )

    except Exception as e:
        print(f"❌ 生成避雷报告失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"生成避雷报告失败: {str(e)}"
        )


@router.post(
    "/tasks",
    response_model=ShoppingTaskCreateResponse,
    summary="创建购物分析任务",
    description="创建后台购物分析任务,返回 task_id 用于轮询任务状态和节点级 trace"
)
async def create_analysis_task(request: ShoppingRequest):
    """创建后台购物分析任务"""
    try:
        task_id = task_store.create_task()
        task_executor.submit(_run_shopping_analysis_task, task_id, request)
        return ShoppingTaskCreateResponse(
            success=True,
            message="购物分析任务已创建",
            task_id=task_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"创建购物分析任务失败: {str(e)}"
        )


@router.get(
    "/tasks/{task_id}",
    response_model=ShoppingAnalysisTaskStatus,
    summary="查询购物分析任务状态",
    description="根据 task_id 查询任务进度、结果和节点级 trace"
)
async def get_analysis_task(task_id: str):
    """查询任务状态"""
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.get(
    "/tasks/{task_id}/trace",
    response_model=ShoppingTaskTraceResponse,
    summary="查询购物分析任务Trace",
    description="返回每个 LangGraph 节点的开始时间、结束时间、耗时、状态和错误信息"
)
async def get_analysis_task_trace(task_id: str):
    """查询任务 trace"""
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return ShoppingTaskTraceResponse(
        success=True,
        task_id=task_id,
        trace=task.trace,
    )


@router.get(
    "/health",
    summary="健康检查",
    description="检查购物避雷服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        advisor = get_shopping_advisor()

        return {
            "status": "healthy",
            "service": "shopping-advisor",
            "agents": [
                "候选产品抽取专家",
                "测评搜集专家",
                "价格对比专家",
                "避雷检测专家",
                "报告生成专家"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )
