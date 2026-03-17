#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理咨询伴侣智能AGENT
能够自主决定是否进行RAG检索、主题分类和查询生成
采用ReAct模式进行查询优化：Observation(轻检索) → Thought(分析) → Action(改写/完成)
"""

import requests
from typing import List, Dict, Any, Optional, Tuple
from config import *

class PsychologyAgent:
    MAX_REACT_STEPS = 3

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.llm_url = f"{DEEPSEEK_BASE_URL}/chat/completions"
        print("心理咨询AGENT初始化完成")

    def _build_conversation_context(self, conversation_history: List[Dict] = None) -> str:
        """构建对话历史上下文"""
        if not conversation_history:
            return "无"
        lines = []
        for msg in conversation_history:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                lines.append(f"用户: {content}")
            elif role == 'assistant':
                lines.append(f"助手: {content}")
        return '\n'.join(lines) if lines else "无"

    # ==================== 合并后：RAG判断 + 主题分类（单次LLM调用） ====================

    def judge_rag_and_classify(self, user_message: str, conversation_history: List[Dict] = None) -> Tuple[bool, List[str]]:
        """一次LLM调用完成：判断是否需要RAG + 主题分类

        Returns:
            Tuple[bool, List[str]]: (是否需要RAG, 主题列表)
        """
        context = self._build_conversation_context(conversation_history)

        prompt = f"""你是心理咨询助手的智能决策系统。请完成两个任务：
1. 判断用户问题是否需要从心理学知识库中检索
2. 如果需要，识别属于哪些心理学主题（1-3个）

对话历史:
{context}

用户问题: {user_message}

RAG判断标准:
- 简单问候、感谢、闲聊 → 不需要检索
- 具体心理问题、情感困扰、人际关系等需专业建议 → 需要检索
- 询问心理学知识、理论、方法 → 需要检索
- 需要情感支持和专业指导 → 需要检索

主题（只从以下12个选择）：
1. 情绪 - 焦虑、抑郁、愤怒、恐惧、情绪管理、情感压抑、情绪波动
2. 人际 - 朋友关系、社交恐惧、人际冲突、被排斥孤立、沟通困难
3. 婚恋 - 恋爱关系、伴侣沟通、情感矛盾、分手挽回、亲密关系困扰
4. 家庭 - 父母关系、家庭暴力、原生家庭影响、亲子关系、家庭责任
5. 性心理 - 性取向、性欲、性困惑、婚外情、性别认同
6. 成长 - 青少年发展、学业压力、自我突破、人生规划、考试压力
7. 治疗 - 心理疾病、躯体化障碍、心理治疗方法、专业干预
8. 社会 - 社会现象、心理健康科普、社会议题
9. 职场 - 职业选择、工作压力、失业困境、工作倦怠、职场人际
10. 自我 - 自我认同、自我价值、人生迷茫、自信心
11. 行为 - 强迫行为、习惯问题、行为模式、反复确认
12. 心理学知识 - 心理学理论、人格特质、心理学概念

返回格式:
- 不需要检索: NO
- 需要检索: YES,主题1,主题2
- 需要检索但无法分类: YES,通用

示例:
- "你好啊" → NO
- "我和女朋友总是吵架" → YES,婚恋,人际
- "感觉自己做什么都不行" → YES,自我,情绪
- "总是反复确认门锁" → YES,行为,情绪
- 对话提到工作，用户说"我很焦虑" → YES,职场,情绪

请结合对话历史理解用户的完整含义和真实意图。严格按格式回答，不要解释。"""

        try:
            response = self._call_llm(prompt, max_tokens=30)
            parts = [p.strip() for p in response.strip().split(',')]

            if not parts or parts[0].upper() == 'NO':
                print("📊 RAG判断: NO")
                return False, []

            if parts[0].upper() == 'YES':
                topics = [t for t in parts[1:] if t and t != '通用']
                print(f"📊 RAG判断: YES | 主题: {topics if topics else '通用（全主题检索）'}")
                return True, topics

            return False, []
        except:
            return True, []

    # ==================== ReAct 模式：轻检索 → 思考 → 行动 ====================

    def generate_search_queries_with_pre_retrieval(self, user_message: str, topics: List[str] = None,
                                                     conversation_history: List[Dict] = None,
                                                     vector_store=None) -> List[str]:
        """ReAct模式的查询优化循环：
        Observation(轻检索锚点) → Thought(分析锚点质量) → Action(FINISH/REWRITE)
        最多迭代 MAX_REACT_STEPS 轮，防止死循环。
        """
        if not vector_store:
            return self._fallback_query_generation(user_message, topics, conversation_history)

        context = self._build_conversation_context(conversation_history)
        current_queries = [user_message]
        prev_rewrite = None

        for step in range(self.MAX_REACT_STEPS):
            print(f"\n🔄 ReAct 第{step + 1}/{self.MAX_REACT_STEPS}轮")

            # === Observation: 轻检索获取知识库锚点 ===
            retrieve_query = current_queries[0] if step == 0 else current_queries[-1]
            print(f"🔍 Observation - 轻检索: {retrieve_query}")
            anchor_docs = self._light_retrieval(retrieve_query, vector_store, top_k=3)

            if not anchor_docs:
                print("⚠️ 轻检索无结果")
                if step == 0:
                    return self._fallback_query_generation(user_message, topics, conversation_history)
                break

            anchor_info = self._extract_anchor_info(anchor_docs)
            anchor_topics = ', '.join(anchor_info['topics'][:3]) if anchor_info['topics'] else '无'
            anchor_expressions = '\n'.join([f"- {expr}" for expr in anchor_info['expressions'][:3]])

            # === Thought + Action: LLM 分析锚点并决策 ===
            query_desc = f"原始查询: {user_message}"
            if step > 0 and len(current_queries) > 1:
                query_desc += f"\n上轮改写: {', '.join(current_queries[1:])}"

            result = self._react_think_and_act(
                query_desc, anchor_topics, anchor_expressions, context, is_first_round=(step == 0)
            )

            if result['action'] == 'FINISH':
                print(f"✅ Thought→Action: 查询已足够精准，FINISH")
                break

            if result['action'] == 'REWRITE':
                new_queries = result.get('queries', [])
                if not new_queries:
                    print("⚠️ 改写结果为空，结束ReAct")
                    break

                # 死循环保护：检测改写结果是否与上轮相同
                if prev_rewrite and set(new_queries) == set(prev_rewrite):
                    print("⚠️ 改写结果与上轮相同，终止循环")
                    break

                prev_rewrite = new_queries
                current_queries = [user_message] + new_queries
                print(f"📝 Thought→Action: REWRITE → {new_queries}")

                if step >= self.MAX_REACT_STEPS - 1:
                    print(f"⚠️ 达到最大迭代次数({self.MAX_REACT_STEPS})，强制结束")
                    break

        print(f"✅ 最终检索查询: {current_queries}")
        return current_queries

    def _react_think_and_act(self, query_desc: str, anchor_topics: str, anchor_expressions: str,
                               context: str, is_first_round: bool) -> Dict[str, Any]:
        """ReAct单步：Thought(分析锚点质量) + Action(FINISH 或 REWRITE并输出改写查询词)
        合并了原 _should_rewrite_query 和 _guided_query_rewrite 的功能。
        """
        round_hint = ("这是首轮分析，请仔细判断原始查询与知识库的匹配度。"
                       if is_first_round else
                       "这是验证轮，改写后若检索锚点已明显改善，请选择FINISH。")

        prompt = f"""你是心理咨询知识库的检索优化专家。请分析轻检索锚点，决定下一步操作。
{round_hint}

对话历史:
{context}

{query_desc}

知识库轻检索锚点:
主题: {anchor_topics}
表达样例:
{anchor_expressions}

**Thought - 从以下维度分析：**
1. 查询是否过于口语化，需要转换为更专业的心理学表达？
2. 查询是否过于宽泛，需要细化为具体的心理学概念？
3. 查询与知识库表达差异是否较大，需要对齐表达方式？
4. 查询是否缺少关键信息，需要补充心理学术语？
5. 查询是否含指代词（"这样""那个"等），或结合对话历史发现真实含义与锚点不匹配？

**Action - 二选一：**
- 查询已清晰且与知识库表达接近 → FINISH
- 需要改写 → REWRITE，并生成3-5个精准查询词

改写要求（仅REWRITE时）：
- 结合对话历史理解用户真实意图，处理指代词和语境
- 参考知识库表达方式，从不同角度描述（症状/原因/解决方案）
- 每个查询词≤15字

返回格式（严格遵守，不要解释）：
FINISH
或
REWRITE
查询词1
查询词2
查询词3

示例:
查询"上班好累不想干了"，锚点偏专业 →
REWRITE
工作倦怠
职业压力
职场疲劳
工作动力缺失

对话中助手问"迷茫从什么时候开始"，用户答"一直都这样" →
REWRITE
长期迷茫感
缺乏生活方向感
存在价值迷失

查询"焦虑症怎么缓解"，锚点匹配良好 →
FINISH"""

        try:
            response = self._call_llm(prompt, max_tokens=120)
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

            if not lines:
                return {'action': 'FINISH'}

            first_line = lines[0].upper().strip()

            if 'FINISH' in first_line:
                return {'action': 'FINISH'}

            if 'REWRITE' in first_line:
                queries = []
                for line in lines[1:]:
                    cleaned = line.lstrip('- •123456789.').strip()
                    if cleaned and len(cleaned) > 2 and 'REWRITE' not in cleaned.upper() and 'FINISH' not in cleaned.upper():
                        queries.append(cleaned)
                return {'action': 'REWRITE', 'queries': queries[:5]}

            return {'action': 'FINISH'}
        except Exception as e:
            print(f"⚠️ ReAct决策出错: {e}")
            return {'action': 'FINISH'}

    # ==================== 辅助方法 ====================

    def _light_retrieval(self, query: str, vector_store, top_k: int = 3) -> List[Dict]:
        """轻检索：快速获取知识库锚点"""
        try:
            results = vector_store.search(query, top_k=top_k, threshold=0.05, topics=None)
            return results[:top_k] if results else []
        except Exception as e:
            print(f"轻检索失败: {e}")
            return []

    def _extract_anchor_info(self, anchor_docs: List[Dict]) -> Dict[str, Any]:
        """从锚点文档中提取关键信息"""
        anchor_info = {
            'keywords': [],
            'concepts': [],
            'expressions': [],
            'topics': []
        }

        for doc in anchor_docs:
            content = doc['content']
            metadata = doc['metadata']

            if 'topic' in metadata:
                topic = metadata['topic']
                if topic not in anchor_info['topics']:
                    anchor_info['topics'].append(topic)

            if len(content) > 20:
                expression = content[:50].replace('\n', ' ').strip()
                if expression not in anchor_info['expressions']:
                    anchor_info['expressions'].append(expression)

        return anchor_info

    def _fallback_query_generation(self, user_message: str, topics: List[str] = None,
                                     conversation_history: List[Dict] = None) -> List[str]:
        """降级方案：无vector_store时的查询生成"""
        context = self._build_conversation_context(conversation_history)
        topics_str = ', '.join(topics) if topics else '未分类'

        prompt = f"""你是心理咨询知识库检索专家。根据用户问题和对话历史，生成3-5个不同角度的检索查询词。

用户问题: {user_message}
主题分类: {topics_str}
对话历史: {context}

要求: 使用心理学专业术语，从不同角度描述（症状/原因/解决方案），每个查询词≤15字。

示例:
问题: "最近失业了，感觉被生活抛弃了，不知道怎么办"
查询词:
失业压力
工作焦虑
职业迷茫
生活困境
重新规划

请只返回查询词，每行一个，不要编号。"""

        try:
            response = self._call_llm(prompt, max_tokens=100)
            generated_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                line = line.lstrip('- •123456789.').strip()
                if line and len(line) > 2:
                    generated_queries.append(line)

            return [user_message] + generated_queries[:5]

        except Exception as e:
            print(f"降级查询生成失败: {e}")
            return [user_message]

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """调用LLM API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.8
            }

            response = requests.post(self.llm_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return ""

        except Exception as e:
            print(f"调用LLM时出错: {e}")
            return ""

    # ==================== 入口方法 ====================

    def analyze_user_input(self, user_message: str, conversation_history: List[Dict] = None,
                           vector_store=None, force_retrieval: bool = False) -> Dict[str, Any]:
        """综合分析用户输入，返回决策结果

        Args:
            user_message: 用户消息
            conversation_history: 对话历史
            vector_store: 向量存储对象
            force_retrieval: 是否强制检索（由对话持续监控Agent触发）

        Returns:
            分析结果字典
        """
        # 1. 合并后的 RAG判断 + 主题分类（单次LLM调用）
        if force_retrieval:
            need_rag = True
            print("🚨 强制检索模式：跳过RAG判断")
            _, topics = self.judge_rag_and_classify(user_message, conversation_history)
        else:
            need_rag, topics = self.judge_rag_and_classify(user_message, conversation_history)

        # 2. ReAct模式的查询优化
        search_queries = []
        if need_rag:
            search_queries = self.generate_search_queries_with_pre_retrieval(
                user_message, topics, conversation_history, vector_store
            )

        return {
            'need_rag': need_rag,
            'topics': topics,
            'topic': topics[0] if topics else None,
            'search_queries': search_queries if need_rag else [],
            'search_query': search_queries[0] if need_rag and search_queries else None,
            'original_message': user_message,
            'forced': force_retrieval
        }
