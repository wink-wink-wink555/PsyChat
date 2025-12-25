#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理咨询伴侣 Web界面
基于FastAPI提供美观的Web界面，支持RAG知识库查询
"""

import sys
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
from typing import Optional
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_system import RAGSystem
from src.config import *

# 创建FastAPI应用
app = FastAPI(title="心理咨询伴侣")

# 创建全局RAG系统实例
rag_system = RAGSystem()


def get_web_interface():
    """生成RAG系统Web界面HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>心理咨询伴侣</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }

            body {
                font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
                background: linear-gradient(135deg, #ffeef8 0%, #e8f3ff 50%, #fff5f7 100%);
                min-height: 100vh;
                line-height: 1.8;
                font-weight: 300;
                letter-spacing: 0.3px;
            }

            .container {
                max-width: 950px;
                margin: 0 auto;
                padding: 25px;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            .header {
                background: rgba(255, 255, 255, 0.75);
                backdrop-filter: blur(20px);
                padding: 20px 30px;
                border-radius: 24px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 2px 30px rgba(255, 192, 203, 0.15), 0 1px 8px rgba(0, 0, 0, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.6);
            }

            .header h1 {
                color: #6b5b7f;
                font-size: 1.75em;
                margin: 0;
                font-weight: 400;
                letter-spacing: 1.5px;
                background: linear-gradient(135deg, #d4a5d6, #f2c2d0, #b8a6d9);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .header p {
                color: #8b7f95;
                margin: 8px 0 0 0;
                font-size: 0.85em;
                font-weight: 300;
                letter-spacing: 0.5px;
            }

            .chat-container {
                background: rgba(255, 255, 255, 0.65);
                backdrop-filter: blur(20px);
                border-radius: 28px;
                padding: 25px;
                flex: 1;
                display: flex;
                flex-direction: column;
                box-shadow: 0 4px 40px rgba(212, 165, 214, 0.1), 0 2px 12px rgba(0, 0, 0, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }

            .messages {
                flex: 1;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 20px;
                margin-bottom: 20px;
                background: rgba(255, 252, 253, 0.4);
                border-radius: 20px;
                border: 1px solid rgba(240, 230, 245, 0.4);
                height: calc(100vh - 300px);
                min-height: 400px;
                max-height: calc(100vh - 300px);
                scroll-behavior: smooth;
            }

            .message {
                margin-bottom: 18px;
                padding: 16px 24px;
                border-radius: 20px;
                max-width: 82%;
                word-wrap: break-word;
                position: relative;
                animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }

            @keyframes messageSlide {
                from {
                    opacity: 0;
                    transform: translateY(15px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .user-message {
                background: linear-gradient(135deg, #e8d5f2, #f0daf5);
                color: #6b5b7f;
                margin-left: auto;
                box-shadow: 0 3px 20px rgba(212, 165, 214, 0.2), 0 1px 6px rgba(0, 0, 0, 0.05);
                border-bottom-right-radius: 6px;
                font-weight: 400;
            }

            .assistant-message {
                background: linear-gradient(135deg, #ffffff, #fefafc);
                color: #5a4d6a;
                margin-right: auto;
                border-left: 3px solid #d4a5d6;
                box-shadow: 0 3px 20px rgba(184, 166, 217, 0.12), 0 1px 6px rgba(0, 0, 0, 0.04);
                border-bottom-left-radius: 6px;
                text-align: left;
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: break-word;
                font-weight: 300;
            }

            .sources-info {
                background: rgba(232, 213, 242, 0.15);
                margin-top: 12px;
                border-radius: 14px;
                font-size: 0.88em;
                border: 1px solid rgba(212, 165, 214, 0.2);
                overflow: hidden;
            }

            .sources-header {
                background: rgba(240, 230, 245, 0.25);
                padding: 11px 14px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-weight: 400;
                color: #9b7eab;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .sources-header:hover {
                background: rgba(232, 213, 242, 0.3);
            }

            .sources-toggle {
                font-size: 0.85em;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-weight: 500;
            }

            .sources-content {
                padding: 14px;
                display: none;
                border-top: 1px solid rgba(240, 230, 245, 0.3);
            }

            .sources-content.show {
                display: block;
            }

            .input-form {
                display: flex;
                gap: 14px;
                align-items: flex-end;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(254, 250, 252, 0.7));
                padding: 18px;
                border-radius: 20px;
                border: 1px solid rgba(240, 230, 245, 0.4);
                box-shadow: 0 3px 25px rgba(212, 165, 214, 0.12), 0 1px 8px rgba(0, 0, 0, 0.03);
                backdrop-filter: blur(15px);
            }

            .message-input {
                flex: 1;
                padding: 13px 18px;
                border: 1.5px solid rgba(232, 213, 242, 0.3);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.9);
                font-size: 0.92em;
                resize: none;
                min-height: 46px;
                max-height: 120px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 2px 12px rgba(212, 165, 214, 0.08);
                font-family: inherit;
                line-height: 1.6;
                font-weight: 300;
                color: #5a4d6a;
            }

            .message-input:focus {
                outline: none;
                border-color: #d4a5d6;
                box-shadow: 0 0 0 3px rgba(212, 165, 214, 0.12), 0 4px 18px rgba(212, 165, 214, 0.15);
                transform: translateY(-1px);
            }

            .message-input::placeholder {
                color: #b8a6c8;
                font-style: normal;
                font-weight: 300;
            }

            .send-button {
                width: 46px;
                height: 46px;
                background: linear-gradient(135deg, #e8d5f2, #d4a5d6);
                border: none;
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 3px 18px rgba(212, 165, 214, 0.25), 0 1px 6px rgba(0, 0, 0, 0.05);
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                position: relative;
            }

            .send-button i {
                color: #6b5b7f;
                font-size: 15px;
            }

            .send-button:hover:not(:disabled) {
                transform: translateY(-2px) scale(1.03);
                box-shadow: 0 6px 28px rgba(212, 165, 214, 0.35), 0 2px 10px rgba(0, 0, 0, 0.08);
                background: linear-gradient(135deg, #d4a5d6, #c295c8);
            }

            .send-button:active:not(:disabled) {
                transform: translateY(0px) scale(1);
                box-shadow: 0 2px 12px rgba(212, 165, 214, 0.25);
            }

            .send-button:disabled {
                opacity: 0.45;
                cursor: not-allowed;
                transform: none;
                box-shadow: 0 2px 10px rgba(212, 165, 214, 0.15);
                background: linear-gradient(135deg, #e6dce9, #d5cad8);
            }

            .loading {
                display: none;
                text-align: center;
                padding: 28px;
                margin: 18px 0;
                background: linear-gradient(135deg, rgba(232, 213, 242, 0.15), rgba(240, 230, 245, 0.15));
                border-radius: 18px;
                border: 1px solid rgba(212, 165, 214, 0.2);
            }

            .loading.show { 
                display: block; 
            }

            .loading-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 16px;
            }

            .loading-text {
                color: #9b7eab;
                font-weight: 400;
                font-size: 1.1em;
                display: flex;
                align-items: center;
                gap: 14px;
                letter-spacing: 0.5px;
            }

            .loading-spinner {
                width: 22px;
                height: 22px;
                border: 2.5px solid rgba(212, 165, 214, 0.25);
                border-top: 2.5px solid #d4a5d6;
                border-radius: 50%;
                animation: spin 1s cubic-bezier(0.4, 0, 0.2, 1) infinite;
            }

            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .example-questions {
                background: linear-gradient(135deg, rgba(255, 250, 253, 0.7), rgba(250, 245, 255, 0.7));
                border-radius: 18px;
                padding: 24px;
                margin-bottom: 18px;
                border: 1px solid rgba(240, 230, 245, 0.4);
                backdrop-filter: blur(8px);
            }

            .welcome-message {
                color: #6b5b7f;
                margin-bottom: 18px;
                font-size: 0.95em;
                line-height: 1.8;
                text-align: center;
                padding: 18px;
                background: rgba(255, 255, 255, 0.65);
                border-radius: 15px;
                border-left: 3px solid #d4a5d6;
                font-weight: 300;
                letter-spacing: 0.3px;
            }

            .example-questions h3 {
                color: #6b5b7f;
                margin-bottom: 16px;
                font-size: 0.95em;
                text-align: center;
                font-weight: 400;
                letter-spacing: 1px;
            }

            .examples-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 13px;
            }

            .example-item {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.85), rgba(254, 250, 252, 0.85));
                border-radius: 14px;
                padding: 13px 18px;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border-left: 2.5px solid #d4a5d6;
                font-size: 0.88em;
                box-shadow: 0 2px 10px rgba(212, 165, 214, 0.1), 0 1px 4px rgba(0, 0, 0, 0.03);
                border: 1px solid rgba(240, 230, 245, 0.3);
                text-align: center;
                font-weight: 300;
                color: #6b5b7f;
            }

            .example-item:hover {
                background: linear-gradient(135deg, #e8d5f2, #f0daf5);
                color: #6b5b7f;
                transform: translateY(-2px) scale(1.02);
                box-shadow: 0 4px 20px rgba(212, 165, 214, 0.25), 0 2px 8px rgba(0, 0, 0, 0.05);
            }

                        .assistant-message h1 {
                font-size: 1.35em;
                color: #5a4d6a;
                margin: 16px 0 12px 0;
                font-weight: 500;
            }
            
            .assistant-message h2 {
                font-size: 1.18em;
                color: #5a4d6a;
                margin: 14px 0 10px 0;
                font-weight: 500;
            }
            
            .assistant-message h3 {
                font-size: 1.08em;
                color: #6b5b7f;
                margin: 12px 0 8px 0;
                font-weight: 400;
            }
            
            .assistant-message h4 {
                font-size: 1.02em;
                color: #6b5b7f;
                margin: 10px 0 6px 0;
                font-weight: 400;
            }
            
            .assistant-message p {
                margin: 10px 0;
                line-height: 1.8;
                text-align: left;
                padding: 0;
                text-indent: 0;
            }
            
            .assistant-message strong {
                color: #5a4d6a;
                font-weight: 500;
            }
            
            /* 自定义列表样式 */
            .custom-list {
                list-style: none;
                padding-left: 0;
                margin: 16px 0;
            }
            
            .custom-list li {
                margin: 9px 0;
                position: relative;
                line-height: 1.8;
                display: block;
            }
            
            .main-item {
                font-weight: 400;
                color: #5a4d6a;
                font-size: 1.02em;
                margin: 13px 0;
                padding-left: 22px;
            }
            
            .sub-item {
                font-weight: 300;
                color: #6b5b7f;
                font-size: 0.94em;
                margin: 7px 0;
                padding-left: 38px;
            }
            
            .main-item::before {
                content: '';
                position: absolute;
                left: 6px;
                top: 0.65em;
                width: 5px;
                height: 5px;
                background: #d4a5d6;
                border-radius: 50%;
            }
            
            .sub-item::before {
                content: '';
                position: absolute;
                left: 22px;
                top: 0.65em;
                width: 3.5px;
                height: 3.5px;
                background: #c5b3d2;
                border-radius: 50%;
            }
            
            .assistant-message a {
                color: #9b7eab;
                text-decoration: none;
                font-weight: 400;
                border-bottom: 1px solid rgba(155, 126, 171, 0.3);
                transition: all 0.2s ease;
            }
            
            .assistant-message a:hover {
                color: #8a6d9a;
                border-bottom-color: #8a6d9a;
            }
            
            .assistant-message p {
                margin: 9px 0;
                line-height: 1.8;
            }
            
            .assistant-message ul, .assistant-message ol {
                margin: 9px 0;
                padding-left: 22px;
            }
            
            .assistant-message li {
                margin: 5px 0;
                line-height: 1.7;
            }
            
            .assistant-message blockquote {
                border-left: 3px solid #d4a5d6;
                margin: 12px 0;
                padding: 12px 16px;
                background: rgba(232, 213, 242, 0.1);
                border-radius: 0 10px 10px 0;
            }
            
            .assistant-message code {
                background: rgba(212, 165, 214, 0.12);
                padding: 2px 7px;
                border-radius: 5px;
                font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
                font-size: 0.88em;
            }
            
            .code-block {
                background: rgba(90, 77, 106, 0.04);
                border-radius: 12px;
                margin: 12px 0;
                border: 1px solid rgba(240, 230, 245, 0.4);
                position: relative;
            }
            
            .code-header {
                background: rgba(232, 213, 242, 0.15);
                padding: 9px 14px;
                border-bottom: 1px solid rgba(240, 230, 245, 0.3);
                border-radius: 12px 12px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .code-language {
                font-weight: 500;
                color: #6b5b7f;
                font-size: 0.86em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .copy-button {
                background: linear-gradient(135deg, #e8d5f2, #d4a5d6);
                color: #6b5b7f;
                border: none;
                padding: 5px 10px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.78em;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                gap: 5px;
                font-weight: 400;
            }
            
            .copy-button:hover {
                background: linear-gradient(135deg, #d4a5d6, #c295c8);
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(212, 165, 214, 0.25);
            }
            
            .copy-button:active {
                transform: translateY(0);
            }
            
            .copy-button.copied {
                background: linear-gradient(135deg, #b8e6c9, #a0d9b5);
                color: #3d6b4a;
            }
            
            .assistant-message pre {
                background: none;
                padding: 14px;
                margin: 0;
                border-radius: 0 0 12px 12px;
                overflow-x: auto;
                line-height: 1.5;
            }
            
            .assistant-message pre code {
                background: none;
                padding: 0;
                border-radius: 0;
                line-height: 1.5;
                white-space: pre;
            }

            /* 响应式设计 */
            @media (max-width: 768px) {
                .container {
                    padding: 12px;
                }

                .header h1 {
                    font-size: 1.5em;
                }

                .message {
                    max-width: 92%;
                    padding: 14px 18px;
                }

                .examples-grid {
                    grid-template-columns: 1fr;
                    gap: 10px;
                }

                .input-form {
                    flex-direction: column;
                    gap: 14px;
                    padding: 14px;
                }

                .message-input {
                    min-height: 44px;
                }

                .send-button {
                    width: 100%;
                    height: 46px;
                }

                .messages {
                    height: calc(100vh - 340px);
                }
            }

            /* 滚动条美化 */
            .messages::-webkit-scrollbar {
                width: 5px;
            }

            .messages::-webkit-scrollbar-track {
                background: rgba(240, 230, 245, 0.25);
                border-radius: 3px;
            }

            .messages::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #e8d5f2, #d4a5d6);
                border-radius: 3px;
            }

            .messages::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #d4a5d6, #c295c8);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✨ 心理咨询伴侣</h1>
            </div>

            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="example-questions">
                        <div class="welcome-message">
                            🌸 欢迎来到心理咨询伴侣，我会温柔地陪伴你，倾听你的心声，为你提供专业的心理支持。
                            <br><br>
                            <em>温馨提示：本服务提供心理支持和建议，如有严重心理问题请及时寻求专业医疗帮助。</em>
                        </div>
                        <h3>💭 你可以这样开始：</h3>
                        <div class="examples-grid">
                            <div class="example-item" onclick="askExample('我最近感到很焦虑，该怎么办？')">
                                😰 我最近感到很焦虑
                            </div>
                            <div class="example-item" onclick="askExample('如何改善人际关系？')">
                                👥 如何改善人际关系
                            </div>
                            <div class="example-item" onclick="askExample('怎样管理工作压力？')">
                                💼 怎样管理工作压力
                            </div>
                            <div class="example-item" onclick="askExample('如何提高自信心？')">
                                💪 如何提高自信心
                            </div>
                        </div>
                    </div>
                </div>

                <div class="loading" id="loading">
                    <div class="loading-content">
                        <div class="loading-text">
                            <div class="loading-spinner"></div>
                            <span>正在为你生成回复...</span>
                        </div>
                    </div>
                </div>

                <form class="input-form" onsubmit="return submitForm(event)">
                    <textarea 
                        id="messageInput" 
                        class="message-input" 
                        placeholder="说说你的感受，我会认真倾听..."
                        rows="2"
                        onkeydown="handleKeyPress(event)"
                    ></textarea>
                    <button type="submit" class="send-button" id="sendButton">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </form>
            </div>
        </div>

<script>
function askExample(text) {
    document.getElementById('messageInput').value = text;
    submitMessage();
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        submitMessage();
    }
}

function submitForm(event) {
    event.preventDefault();
    submitMessage();
    return false;
}

async function submitMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    input.value = '';
    showLoading(true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: 'message=' + encodeURIComponent(message)
        });

        if (response.ok) {
            const result = await response.json();
            addMessage(result.message, 'assistant', result.sources);
        } else {
            addMessage('抱歉，发生了错误，请稍后重试。', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('网络连接错误，请检查网络后重试。', 'assistant');
    } finally {
        showLoading(false);
    }
}

function parseMarkdown(text) {
    // 首先对文本进行彻底的清理
    let result = text;
    
    // 将文本按行分割，逐行处理
    let initialLines = result.split('\\n');
    let cleanedLines = [];
    
    for (let i = 0; i < initialLines.length; i++) {
        let line = initialLines[i];
        
        // 完全清理行首的空格和制表符
        line = line.replace(/^[\\s\\t]+/, '');
        
        // 如果不是空行，就添加到结果中
        if (line.trim().length > 0) {
            cleanedLines.push(line);
        }
    }
    
    // 重新组合文本
    result = cleanedLines.join('\\n');
    
    // 先处理代码块，保护其中的内容不被其他规则处理
    let codeBlocks = [];
    let codeBlockIndex = 0;
    
    // 提取代码块并用占位符替换
    result = result.replace(/```(\\w+)?([\\s\\S]*?)```/g, function(match, language, content) {
        let placeholder = `__CODEBLOCK_${codeBlockIndex}__`;
        let lang = language || 'code';
        let codeId = `code-${Date.now()}-${codeBlockIndex}`;
        
        codeBlocks[codeBlockIndex] = `
            <div class="code-block">
                <div class="code-header">
                    <span class="code-language">${lang}</span>
                    <button class="copy-button" onclick="copyCode('${codeId}')">
                        <i class="fas fa-copy"></i>
                        复制
                    </button>
                </div>
                <pre><code id="${codeId}">${content}</code></pre>
            </div>
        `;
        codeBlockIndex++;
        return placeholder;
    });
    
    // 处理标题（现在代码块已经被保护）
    result = result.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    result = result.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    result = result.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    result = result.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    // 处理粗体
    result = result.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    
    // 处理斜体
    result = result.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    
    // 处理行内代码
    result = result.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // 处理链接
    result = result.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // 处理列表项 - 更精确的方法
    let listLines = result.split('\\n');
    let processedLines = [];
    let inList = false;
    
    for (let j = 0; j < listLines.length; j++) {
        let currentLine = listLines[j];
        
        // 检查是否是有序列表项
        if (/^\\d+\\.\\s*(.+)$/.test(currentLine)) {
            let content = currentLine.replace(/^\\d+\\.\\s*/, '').trim();
            if (!inList) {
                processedLines.push('<ul class="custom-list">');
                inList = true;
            }
            processedLines.push('<li class="main-item">' + content + '</li>');
        }
        // 检查是否是无序列表项
        else if (/^[-*+]\\s*(.+)$/.test(currentLine)) {
            let content = currentLine.replace(/^[-*+]\\s*/, '').trim();
            if (!inList) {
                processedLines.push('<ul class="custom-list">');
                inList = true;
            }
            processedLines.push('<li class="sub-item">' + content + '</li>');
        }
        // 不是列表项
        else {
            if (inList) {
                processedLines.push('</ul>');
                inList = false;
            }
            processedLines.push(currentLine);
        }
    }
    
    // 如果最后还在列表中，关闭列表
    if (inList) {
        processedLines.push('</ul>');
    }
    
    result = processedLines.join('\\n');
    
    // 处理引用
    result = result.replace(/^>\\s*(.+)$/gm, '<blockquote>$1</blockquote>');
    
    // 处理段落（将连续的文本包装在p标签中）
    result = result.replace(/^(?!<[h|u|o|b|p|d])(.+)$/gm, function(match, content) {
        return '<p>' + content.trim() + '</p>';
    });
    
    // 清理多余的p标签
    result = result.replace(/<p><\\/p>/g, '');
    result = result.replace(/<p>(<h[1-6]>.*<\\/h[1-6]>)<\\/p>/g, '$1');
    result = result.replace(/<p>(<ul.*<\\/ul>)<\\/p>/g, '$1');
    result = result.replace(/<p>(<blockquote>.*<\\/blockquote>)<\\/p>/g, '$1');
    result = result.replace(/<p>(<pre>.*<\\/pre>)<\\/p>/g, '$1');
    
    // 最后恢复代码块
    for (let k = 0; k < codeBlocks.length; k++) {
        result = result.replace(`__CODEBLOCK_${k}__`, codeBlocks[k]);
    }
    
    return result;
}

function addMessage(content, sender, sources) {
    const messages = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    let html = '';
    if (sender === 'assistant') {
        html = `<div>${parseMarkdown(content)}</div>`;
    } else {
        html = `<div>${content}</div>`;
    }
    
    if (sources && sources.length > 0) {
        const sourcesId = 'sources-' + Date.now();
        html += `
            <div class="sources-info">
                <div class="sources-header" onclick="toggleSources('${sourcesId}')">
                    <span>📚 参考来源 (${sources.length}个)</span>
                    <span class="sources-toggle" id="toggle-${sourcesId}">▼</span>
                </div>
                <div class="sources-content" id="${sourcesId}">`;
        
        for (let i = 0; i < sources.length; i++) {
            const source = sources[i];
            html += `<div>• <strong>${source.source}</strong> (相似度: ${(source.similarity * 100).toFixed(1)}%)`;
            if (source.header) {
                html += `<br>&nbsp;&nbsp;&nbsp;&nbsp;标题: ${source.header}`;
            }
            html += `</div>`;
        }
        
        html += `
                </div>
            </div>`;
    }
    
    messageDiv.innerHTML = html;
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function toggleSources(sourcesId) {
    const content = document.getElementById(sourcesId);
    const toggle = document.getElementById('toggle-' + sourcesId);

    if (content.classList.contains('show')) {
        content.classList.remove('show');
        toggle.classList.remove('expanded');
        toggle.textContent = '▼';
    } else {
        content.classList.add('show');
        toggle.classList.add('expanded');
        toggle.textContent = '▲';
    }
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    const sendButton = document.getElementById('sendButton');

    if (show) {
        loading.classList.add('show');
        sendButton.disabled = true;
    } else {
        loading.classList.remove('show');
        sendButton.disabled = false;
    }
}

function copyCode(codeId) {
    const codeElement = document.getElementById(codeId);
    const button = event.target.closest('.copy-button');
    
    if (codeElement) {
        const text = codeElement.textContent;
        
        // 使用现代的 Clipboard API
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                showCopySuccess(button);
            }).catch(() => {
                // 降级到传统方法
                fallbackCopyText(text, button);
            });
        } else {
            // 降级到传统方法
            fallbackCopyText(text, button);
        }
    }
}

function fallbackCopyText(text, button) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showCopySuccess(button);
    } catch (err) {
        console.error('复制失败:', err);
    } finally {
        document.body.removeChild(textArea);
    }
}

function showCopySuccess(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i> 已复制';
    button.classList.add('copied');
    
    setTimeout(() => {
        button.innerHTML = originalText;
        button.classList.remove('copied');
    }, 2000);
}
</script>
    </body>
    </html>
    """
    return html_content


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页面 - RAG对话界面"""
    return get_web_interface()


@app.post("/chat")
async def chat(message: str = Form(...)):
    """处理聊天请求 - RAG系统查询"""
    try:
        # 检查知识库状态
        info = rag_system.get_knowledge_base_info()

        # 如果知识库为空，先构建知识库
        if info.get('document_count', 0) == 0:
            print("知识库为空，开始构建...")
            success = rag_system.build_knowledge_base()
            if not success:
                return {
                    "success": False,
                    "message": "知识库构建失败，请检查配置和依赖。",
                    "sources": []
                }

        # 生成回答
        result = rag_system.generate_response(message)

        if result['success']:
            return {
                "success": True,
                "message": result['response'],
                "sources": result['sources']
            }
        else:
            return {
                "success": False,
                "message": result.get('response', '抱歉，无法生成回答。'),
                "sources": []
            }

    except Exception as e:
        print(f"❌ RAG聊天处理失败: {str(e)}")
        return {
            "success": False,
            "message": f"抱歉，处理您的请求时出现错误: {str(e)}",
            "sources": []
        }


@app.get("/info")
async def get_info():
    """获取知识库信息"""
    try:
        info = rag_system.get_knowledge_base_info()
        return {
            "success": True,
            "info": info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """主函数 - 启动Web界面"""
    print("🚀 启动心理咨询伴侣Web界面...")
    print("🧠 基于阿里云百炼Qwen3-Embedding + DeepSeek")
    print("📚 知识库: 心理咨询相关文档")
    print("🌐 访问地址: http://localhost:8000")
    print("💡 功能: 智能问答、向量检索、知识库管理")
    print()

    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()

