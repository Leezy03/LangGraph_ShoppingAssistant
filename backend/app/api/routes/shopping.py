"""购物避雷API路由"""

from fastapi import APIRouter, HTTPException
from ...models.schemas import (
    ShoppingRequest,
    ShoppingReportResponse,
    ErrorResponse
)
from ...agents.shopping_advisor_agent import get_shopping_advisor

router = APIRouter(prefix="/shopping", tags=["购物避雷"])


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
