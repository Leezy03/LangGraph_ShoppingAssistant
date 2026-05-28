import axios from 'axios'
import type {
  ShoppingAnalysisTaskStatus,
  ShoppingFormData,
  ShoppingReportResponse,
  ShoppingTaskCreateResponse,
  ShoppingTaskTraceResponse
} from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 360000, // 6分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成避雷购物报告
 */
export async function analyzeProduct(formData: ShoppingFormData): Promise<ShoppingReportResponse> {
  try {
    const response = await apiClient.post<ShoppingReportResponse>('/api/shopping/analyze', formData)
    return response.data
  } catch (error: any) {
    console.error('生成避雷报告失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成避雷报告失败')
  }
}

/**
 * 创建后台购物分析任务
 */
export async function createShoppingTask(formData: ShoppingFormData): Promise<ShoppingTaskCreateResponse> {
  try {
    const response = await apiClient.post<ShoppingTaskCreateResponse>('/api/shopping/tasks', formData)
    return response.data
  } catch (error: any) {
    console.error('创建购物分析任务失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '创建购物分析任务失败')
  }
}

/**
 * 查询购物分析任务状态
 */
export async function getShoppingTask(taskId: string): Promise<ShoppingAnalysisTaskStatus> {
  try {
    const response = await apiClient.get<ShoppingAnalysisTaskStatus>(`/api/shopping/tasks/${taskId}`)
    return response.data
  } catch (error: any) {
    console.error('查询购物分析任务失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '查询购物分析任务失败')
  }
}

/**
 * 查询购物分析任务Trace
 */
export async function getShoppingTaskTrace(taskId: string): Promise<ShoppingTaskTraceResponse> {
  try {
    const response = await apiClient.get<ShoppingTaskTraceResponse>(`/api/shopping/tasks/${taskId}/trace`)
    return response.data
  } catch (error: any) {
    console.error('查询购物分析任务Trace失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '查询购物分析任务Trace失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/api/shopping/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient
