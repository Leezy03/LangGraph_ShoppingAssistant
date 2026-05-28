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

export interface TaskTraceEvent {
  event_id: string
  step_key: string
  step_name: string
  status: 'running' | 'success' | 'failed' | 'partial' | string
  message: string
  started_at: string
  ended_at?: string
  duration_ms?: number
  error_type?: string
  error_message?: string
}

export interface ShoppingAnalysisTaskStatus {
  task_id: string
  status: 'pending' | 'running' | 'succeeded' | 'partial' | 'failed' | string
  current_step?: string
  progress: number
  message: string
  created_at: string
  updated_at: string
  completed_at?: string
  report?: ShoppingReport
  error?: string
  trace: TaskTraceEvent[]
}

export interface ShoppingTaskCreateResponse {
  success: boolean
  message: string
  task_id: string
}

export interface ShoppingTaskTraceResponse {
  success: boolean
  task_id: string
  trace: TaskTraceEvent[]
}
