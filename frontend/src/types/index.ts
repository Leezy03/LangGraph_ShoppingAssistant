// 类型定义 - 避雷购物助手

export interface ReviewSource {
  platform: string
  author: string
  title: string
  url?: string
  stance: string
  is_sponsored: boolean
  key_points: string[]
  credibility_score: number
}

export interface Product {
  name: string
  brand: string
  model: string
  price_range: string
  rating?: number
  image_url?: string
  specs?: Record<string, any>
}

export interface ProductAnalysis {
  product: Product
  reviews: ReviewSource[]
  common_pros: string[]
  common_cons: string[]
  red_flags: string[]
  controversy_points: string[]
  verdict: string
  verdict_reason: string
}

export interface ShoppingReport {
  query: string
  category: string
  products: ProductAnalysis[]
  comparison_summary: string
  final_recommendation: string
  budget_advice?: string
  general_tips: string[]
}

export interface ShoppingFormData {
  product_name: string
  budget_min?: number
  budget_max?: number
  brand_preferences: string[]
  usage_scenario: string
  concerns: string[]
  free_text_input: string
}

export interface ShoppingReportResponse {
  success: boolean
  message: string
  data?: ShoppingReport
}

