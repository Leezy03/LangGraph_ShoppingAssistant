"""数据模型定义 - 避雷购物助手"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============ 请求模型 ============

class ShoppingRequest(BaseModel):
    """购物避雷请求"""
    product_name: str = Field(..., description="要查询的产品名称", example="洗地机")
    budget_min: Optional[int] = Field(default=None, description="预算下限(元)", example=1000)
    budget_max: Optional[int] = Field(default=None, description="预算上限(元)", example=3000)
    brand_preferences: List[str] = Field(default=[], description="品牌偏好", example=["追觅", "石头"])
    usage_scenario: str = Field(default="", description="使用场景", example="家用，120平米")
    concerns: List[str] = Field(default=[], description="关注要点", example=["续航", "噪音", "售后"])
    free_text_input: Optional[str] = Field(default="", description="额外要求", example="家里有宠物，毛发多")

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "洗地机",
                "budget_min": 1000,
                "budget_max": 3000,
                "brand_preferences": ["追觅", "石头"],
                "usage_scenario": "家用，120平米",
                "concerns": ["续航", "噪音", "售后"],
                "free_text_input": "家里有宠物，毛发多"
            }
        }


class ProductSearchRequest(BaseModel):
    """产品搜索请求"""
    keywords: str = Field(..., description="搜索关键词", example="洗地机")
    platform: str = Field(default="all", description="搜索平台: all/bilibili/xiaohongshu/zhihu")


# ============ 响应模型 ============

class ReviewSource(BaseModel):
    """测评来源"""
    platform: str = Field(..., description="平台: B站/小红书/知乎")
    author: str = Field(..., description="作者/博主名")
    title: str = Field(..., description="测评标题")
    url: Optional[str] = Field(default=None, description="原文链接")
    stance: str = Field(..., description="立场: 推荐/不推荐/中立")
    is_sponsored: bool = Field(default=False, description="是否疑似恰饭(广告)")
    key_points: List[str] = Field(default=[], description="核心观点")
    credibility_score: float = Field(default=5.0, description="可信度评分(1-10)")


class Product(BaseModel):
    """产品信息"""
    name: str = Field(..., description="产品名称")
    brand: str = Field(default="", description="品牌")
    model: str = Field(default="", description="型号")
    price_range: str = Field(default="", description="价格区间")
    rating: Optional[float] = Field(default=None, description="综合评分")
    image_url: Optional[str] = Field(default=None, description="产品图片URL")
    specs: Optional[Dict[str, Any]] = Field(default=None, description="关键参数")


class CandidateProduct(BaseModel):
    """候选产品,用于统一后续证据归档目标"""
    name: str = Field(..., description="候选产品标准名称")
    brand: str = Field(default="", description="品牌")
    model: str = Field(default="", description="型号")
    reason: str = Field(default="", description="入选候选列表的原因")


class CandidateExtractionResult(BaseModel):
    """候选产品抽取结果"""
    category: str = Field(default="", description="产品品类")
    candidates: List[CandidateProduct] = Field(default=[], description="候选产品列表")


class ProductAnalysis(BaseModel):
    """单个产品的深度分析"""
    product: Product = Field(..., description="产品信息")
    reviews: List[ReviewSource] = Field(default=[], description="测评来源列表")
    common_pros: List[str] = Field(default=[], description="公认优点")
    common_cons: List[str] = Field(default=[], description="公认缺点")
    red_flags: List[str] = Field(default=[], description="避雷点")
    controversy_points: List[str] = Field(default=[], description="争议点(博主意见不一致)")
    verdict: str = Field(default="待定", description="结论: 推荐/不推荐/看需求")
    verdict_reason: str = Field(default="", description="结论理由")


class ShoppingReport(BaseModel):
    """购物避雷报告"""
    query: str = Field(..., description="用户查询")
    category: str = Field(default="", description="产品品类")
    products: List[ProductAnalysis] = Field(default=[], description="产品分析列表")
    comparison_summary: str = Field(default="", description="横向对比总结")
    final_recommendation: str = Field(default="", description="最终购买建议")
    budget_advice: Optional[str] = Field(default=None, description="预算建议")
    general_tips: List[str] = Field(default=[], description="品类选购通用建议")


class ShoppingReportResponse(BaseModel):
    """购物报告响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[ShoppingReport] = Field(default=None, description="购物报告数据")


# ============ 任务状态与 Trace ============

class TaskTraceEvent(BaseModel):
    """单个工作流节点的 trace 事件"""
    event_id: str = Field(..., description="Trace事件ID")
    step_key: str = Field(..., description="步骤Key")
    step_name: str = Field(..., description="步骤名称")
    status: str = Field(..., description="状态: running/success/failed/partial")
    message: str = Field(default="", description="事件说明")
    started_at: datetime = Field(..., description="开始时间")
    ended_at: Optional[datetime] = Field(default=None, description="结束时间")
    duration_ms: Optional[int] = Field(default=None, description="耗时毫秒")
    error_type: Optional[str] = Field(default=None, description="错误类型")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class ShoppingAnalysisTaskStatus(BaseModel):
    """购物分析任务状态"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态: pending/running/succeeded/partial/failed")
    current_step: Optional[str] = Field(default=None, description="当前步骤")
    progress: int = Field(default=0, description="进度百分比")
    message: str = Field(default="", description="状态说明")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    report: Optional[ShoppingReport] = Field(default=None, description="分析报告")
    error: Optional[str] = Field(default=None, description="失败原因")
    trace: List[TaskTraceEvent] = Field(default_factory=list, description="节点级Trace事件")


class ShoppingTaskCreateResponse(BaseModel):
    """创建购物分析任务响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    task_id: str = Field(..., description="任务ID")


class ShoppingTaskTraceResponse(BaseModel):
    """购物分析任务Trace响应"""
    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务ID")
    trace: List[TaskTraceEvent] = Field(default_factory=list, description="Trace事件列表")


# ============ 错误响应 ============

class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")
