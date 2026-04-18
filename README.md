# AI避雷购物助手

基于 HelloAgents + FastAPI + Vue3 构建的多智能体购物决策系统。

项目核心目标是帮助用户做「避雷式」选购：
- 聚合多平台测评与口碑线索
- 识别可能存在的广告倾向与信息偏差
- 输出结构化结论（推荐/不推荐/看需求）与预算建议

## 功能概览

- 多智能体协作分析：测评搜集、价格对比、避雷检测、报告生成
- 结构化报告输出：产品对比、优缺点、红旗风险、争议点
- 前后端分离：FastAPI API + Vue3 可视化结果页
- 支持本地开发与云端部署（前端可上 Netlify，后端可上 Render/Railway）

## 技术栈

- 后端：Python, FastAPI, Pydantic, HelloAgents
- 智能体工具：MCPTool + Brave Search MCP Server
- 前端：Vue 3, TypeScript, Vite, Ant Design Vue

## 项目结构

```text
helloagents-trip-planner/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   └── shopping_advisor_agent.py
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   └── routes/
│   │   │       └── shopping.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── services/
│   │   │   └── llm_service.py
│   │   └── config.py
│   ├── requirements.txt
│   ├── run.py
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── views/
│   │   │   ├── Home.vue
│   │   │   └── Result.vue
│   │   ├── types/
│   │   └── main.ts
│   ├── package.json
│   └── .env.example
└── README.md
```

## 快速开始

### 1) 后端启动

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
python run.py
```

后端默认地址：
- http://localhost:8000
- 文档：http://localhost:8000/docs

### 2) 前端启动

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

前端默认地址：
- http://localhost:5173

## 环境变量

后端变量（见 backend/.env.example）：
- SEARCH_API_KEY：Brave Search API Key（必填）
- LLM_API_KEY：大模型 API Key（必填）
- LLM_BASE_URL：模型服务地址（必填）
- LLM_MODEL_ID：模型名称（必填）
- HOST、PORT、CORS_ORIGINS、LOG_LEVEL：服务配置

前端变量（见 frontend/.env.example）：
- VITE_API_BASE_URL：后端 API 地址

## API 端点

- POST /api/shopping/analyze
  - 输入：ShoppingRequest
  - 输出：ShoppingReportResponse
- GET /api/shopping/health
  - 服务健康检查

## 致谢与来源说明

本项目基于开源项目进行二次开发，核心框架与工程结构来源于：
- Hello-Agents 教程项目：https://github.com/datawhalechina/Hello-Agents
- HelloAgents 框架项目：https://github.com/jjyaoao/HelloAgents

本仓库主要改造内容：
- 将业务域从“旅行规划”重构为“避雷购物助手”
- 重写后端路由、数据模型和智能体提示词，聚焦购物测评分析
- 重写前端输入与结果展示页面，输出结构化购物决策报告

若本项目中包含原项目受许可证约束的内容，请以原项目许可证要求为准，并保留必要的版权与署名信息。

## 安全说明

- 不要提交真实 .env 到仓库
- 建议仅提交 .env.example
- 若密钥曾暴露，请及时在服务商后台轮换


