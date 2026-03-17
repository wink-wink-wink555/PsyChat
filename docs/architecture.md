# 心理咨询伴侣 - 系统架构

## 项目类型
**RAG-Augmented Conversational Agent** (RAG增强型对话智能体)

## 核心Pipeline流程图

```mermaid
graph TD
    A[用户输入] --> B[Psychology Agent 决策层]

    B --> C{RAG判断 + 主题分类\n单次LLM调用}

    C -->|不需要检索| D[对话持续监控\n连续N轮无RAG?]
    D -->|未超限| E[直接回答\nREBT提示词]
    D -->|已超限| F[🚨 强制触发RAG]

    C -->|需要检索| G[ReAct 查询优化循环\n最多3轮]
    F --> G

    G --> G1[Observation: 轻检索获取锚点 top3]
    G1 --> G2[Thought+Action: 分析锚点质量]
    G2 -->|FINISH| H[使用当前查询词]
    G2 -->|REWRITE| G1

    H --> I[多查询词向量检索\nChromaDB]

    I --> J{找到相关文档?}
    J -->|未找到| E
    J -->|找到| K[上下文扩展\n纯I/O 无LLM]

    K --> K1[通过chunk锚点\n回溯原始完整对话]
    K1 --> L[咨询师话术分析\n带主题缓存]

    L --> L1{缓存命中?}
    L1 -->|命中| M[复用话术分析结果\n0次LLM]
    L1 -->|未命中| L2[LLM分析4维度话术共性\n1次LLM]
    L2 --> M

    M --> N[构建增强提示词\n完整案例+片段+话术+策略]
    E --> O
    N --> O[LLM生成最终回答]

    O --> P[更新对话历史]
    P --> Q[语音合成TTS]
    Q --> R[返回结果]
```


## 系统架构层次

```mermaid
graph LR
    subgraph "用户交互层"
        A1[Web界面<br/>FastAPI]
        A2[CLI界面]
    end
    
    subgraph "核心业务层"
        B1[RAG System<br/>主控制器]
        B2[Psychology Agent<br/>智能决策]
    end
    
    subgraph "检索增强层"
        C1[Vector Store<br/>ChromaDB]
        C2[Data Processor<br/>文档处理]
    end
    
    subgraph "AI服务层"
        D1[DeepSeek LLM<br/>对话生成]
        D2[Alibaba Embedding<br/>向量化]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> C1
    B1 --> C1
    C1 --> D2
    B1 --> D1
    C2 --> C1
    
    style B1 fill:#667eea,color:#fff
    style B2 fill:#764ba2,color:#fff
    style C1 fill:#ff9a9e,color:#fff
    style D1 fill:#fecfef,color:#333
```

## 技术栈
- **LLM**: DeepSeek Chat
- **Embedding**: Alibaba Text-Embedding-v4
- **Vector DB**: ChromaDB
- **Web框架**: FastAPI
- **AI理论**: 理情行为疗法 (REBT)

