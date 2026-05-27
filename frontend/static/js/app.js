// MediAgent 智慧医疗助手前端应用主逻辑
class MediAgentApp {
    constructor() {
        this.messages = [];
        this.chatHistory = [];
        this.currentChatId = null;
        this.agentMode = 'react'; // 默认 ReAct 模式（与 UI 一致）
        this.streamingEnabled = true; // 默认开启流式
        this.isLoading = false;
        this.sidebarCollapsed = false;

        // SSE 相关状态
        this.currentAbortController = null;
        this.currentReasoningSteps = [];
        this.currentRequestId = null;

        // 定时器 ID（用于清理，防止内存泄漏）
        this._loadingTimer = null;
        this._loadingDotTimer = null;
        this._typingTimer = null;

        this.initElements();
        this.initEventListeners();
        this.restoreSidebarState();
        this.loadChatHistory();
        this.fetchUsageStats();
    }

    // 初始化 DOM 元素
    initElements() {
        this.sidebar = document.getElementById('sidebar');
        this.menuToggle = document.getElementById('menuToggle');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.chatHistoryEl = document.getElementById('chatHistory');
        this.chatTitle = document.getElementById('chatTitle');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');

        // Agent 模式选择器
        this.agentModeSelector = document.getElementById('agentModeSelector');
        this.agentModeBtns = document.querySelectorAll('.agent-mode-btn');

        // 流式开关
        this.streamingCheckbox = document.getElementById('streamingCheckbox');

        // 思维面板
        this.thinkingPanel = document.getElementById('thinkingPanel');
        this.thinkingPanelBody = document.getElementById('thinkingPanelBody');
        this.thinkingPanelHeader = document.getElementById('thinkingPanelHeader');
        this.thinkingPanelToggle = document.getElementById('thinkingPanelToggle');
        this.thinkingPanelClose = document.getElementById('thinkingPanelClose');
        this.thinkingSteps = document.getElementById('thinkingSteps');
        this.thinkingPanelEmpty = document.getElementById('thinkingPanelEmpty');
        this.thinkingStepCount = document.getElementById('thinkingStepCount');

        // 余额显示元素
        this.totalBalance = document.getElementById('totalBalance');
        this.toppedUpBalance = document.getElementById('toppedUpBalance');
        this.grantedBalance = document.getElementById('grantedBalance');
        this.balanceStatus = document.getElementById('balanceStatus');

        // 统计元素
        this.usageTokens = document.getElementById('usageTokens');
        this.usageCalls = document.getElementById('usageCalls');
        this.lastUpdated = document.getElementById('lastUpdated');
        this.refreshUsageBtn = document.getElementById('refreshUsageBtn');

        // 侧边栏折叠元素
        this.sidebarTouchZone = document.getElementById('sidebarTouchZone');
        this.expandSidebarBtn = document.getElementById('expandSidebarBtn');
    }

    // 初始化事件监听
    initEventListeners() {
        // 菜单切换
        this.menuToggle.addEventListener('click', () => {
            this.sidebar.classList.toggle('open');
        });

        // 新建对话
        this.newChatBtn.addEventListener('click', () => {
            this.startNewChat();
        });

        // 发送消息
        this.sendBtn.addEventListener('click', () => {
            this.sendMessage();
        });

        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Agent 模式选择器
        this.agentModeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.agentModeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.agentMode = btn.dataset.mode;
            });
        });

        // 流式开关
        this.streamingCheckbox.addEventListener('change', () => {
            this.streamingEnabled = this.streamingCheckbox.checked;
        });

        // 思维面板折叠/展开
        if (this.thinkingPanelToggle) {
            this.thinkingPanelToggle.addEventListener('click', () => {
                this.thinkingPanelBody.classList.toggle('collapsed');
                const icon = this.thinkingPanelToggle.querySelector('svg');
                if (this.thinkingPanelBody.classList.contains('collapsed')) {
                    icon.style.transform = 'rotate(180deg)';
                } else {
                    icon.style.transform = 'rotate(0deg)';
                }
            });
        }

        // 思维面板关闭
        if (this.thinkingPanelClose) {
            this.thinkingPanelClose.addEventListener('click', () => {
                this.thinkingPanel.classList.remove('visible');
            });
        }

        // 自动调整输入框高度
        this.messageInput.addEventListener('input', () => {
            this.autoResizeInput();
        });

        // 快捷操作点击
        document.querySelectorAll('.quick-action').forEach(action => {
            action.addEventListener('click', () => {
                const prompt = action.dataset.prompt;
                this.messageInput.value = prompt;
                this.autoResizeInput();
                this.messageInput.focus();
            });
        });

        // 刷新余额
        this.refreshUsageBtn.addEventListener('click', () => {
            this.fetchUsageStats();
        });

        // 侧边栏折叠/展开（使用透明触摸区域）
        if (this.sidebarTouchZone) {
            this.sidebarTouchZone.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleSidebar();
            });
        }

        if (this.expandSidebarBtn) {
            this.expandSidebarBtn.addEventListener('click', () => {
                this.toggleSidebar(true);
            });
        }

        // 点击外部关闭侧边栏
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 &&
                !this.sidebar.contains(e.target) &&
                !this.menuToggle.contains(e.target)) {
                this.sidebar.classList.remove('open');
            }
        });
    }

    // 自动调整输入框高度
    autoResizeInput() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    // ==================== 加载状态管理 ====================

    showLoading(show = true) {
        this.isLoading = show;
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
        this.sendBtn.disabled = show;
        this.messageInput.disabled = show;

        if (show) {
            this._startLoadingAnimation();
        } else {
            this._clearLoadingAnimation();
        }
    }

    // Loading 动画 - 修复：正确存储和清理所有定时器
    _startLoadingAnimation() {
        this._clearLoadingAnimation(); // 先清理旧的定时器

        const messages = [
            'MediAgent 正在分析...',
            'MediAgent 正在查询医疗知识库...',
            'MediAgent 正在检索药品信息...',
            'MediAgent 正在生成健康建议...',
            '复杂医疗分析可能需要较长时间...',
        ];

        let index = 0;
        let dotCount = 0;

        // 主提示循环（每3秒切换）
        this._loadingTimer = setInterval(() => {
            index = (index + 1) % messages.length;
            if (this.loadingText) {
                this.loadingText.textContent = messages[index];
            }
        }, 3000);

        // 点号动画（每500ms）- 修复：存储引用以便清理
        this._loadingDotTimer = setInterval(() => {
            dotCount = (dotCount + 1) % 4;
            if (this.loadingText) {
                this.loadingText.textContent = messages[index % messages.length] + '.'.repeat(dotCount);
            }
        }, 500);
    }

    // 清理所有加载动画定时器 - 修复内存泄漏
    _clearLoadingAnimation() {
        if (this._loadingTimer) {
            clearInterval(this._loadingTimer);
            this._loadingTimer = null;
        }
        if (this._loadingDotTimer) {
            clearInterval(this._loadingDotTimer);
            this._loadingDotTimer = null;
        }
    }

    // ==================== 打字指示器 ====================

    showTypingIndicator() {
        this._clearTypingIndicator();

        const indicator = document.createElement('div');
        indicator.className = 'message assistant typing-message';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="white" opacity="0.3"/>
                    <path d="M12 4v16M4 12h16" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-role">MediAgent</span>
                    <span class="message-time typing-time"></span>
                </div>
                <div class="message-body">
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                    <span class="typing-text">MediAgent 正在思考</span>
                </div>
            </div>
        `;

        this.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    _clearTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // ==================== 思维面板管理 ====================

    showThinkingPanel() {
        this.thinkingPanel.classList.add('visible');
        this.thinkingPanelBody.classList.remove('collapsed');
        this.thinkingSteps.innerHTML = '';
        this.thinkingPanelEmpty.style.display = 'flex';
        this.thinkingStepCount.textContent = '';
        this.currentReasoningSteps = [];
    }

    hideThinkingPanel() {
        // 推理完成后保持面板可见，让用户可以查看
        // 用户可以手动关闭
    }

    clearThinkingPanel() {
        this.thinkingSteps.innerHTML = '';
        this.thinkingPanelEmpty.style.display = 'flex';
        this.thinkingStepCount.textContent = '';
        this.currentReasoningSteps = [];
    }

    // 添加推理步骤到思维面板
    addThinkingStep(type, content) {
        this.thinkingPanelEmpty.style.display = 'none';

        const step = { type, content, timestamp: new Date().toISOString() };
        this.currentReasoningSteps.push(step);

        const stepEl = document.createElement('div');
        stepEl.className = `thinking-step thinking-step-${type}`;
        stepEl.style.animation = 'thinkingStepIn 0.3s ease-out';

        const typeLabels = {
            thought: '思考',
            action: '行动',
            observation: '观察',
            plan: '计划',
            error: '错误'
        };

        const typeIcons = {
            thought: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
            action: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>',
            observation: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>',
            plan: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h10M4 18h14"/></svg>',
            error: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>'
        };

        stepEl.innerHTML = `
            <div class="thinking-step-header">
                <span class="thinking-step-icon">${typeIcons[type] || typeIcons.thought}</span>
                <span class="thinking-step-label">${typeLabels[type] || type}</span>
                <span class="thinking-step-number">#${this.currentReasoningSteps.length}</span>
            </div>
            <div class="thinking-step-content">${this.formatMessage(content)}</div>
        `;

        this.thinkingSteps.appendChild(stepEl);
        this.thinkingStepCount.textContent = `(${this.currentReasoningSteps.length} 步)`;

        // 自动滚动思维面板到底部
        this.thinkingPanelBody.scrollTop = this.thinkingPanelBody.scrollHeight;
    }

    // ==================== 对话管理 ====================

    startNewChat() {
        this.messages = [];
        this.currentChatId = 'chat_' + Date.now();
        this.welcomeScreen.style.display = 'flex';
        this.messagesContainer.classList.remove('active');
        this.messagesContainer.innerHTML = '';
        this.chatTitle.textContent = '新对话';
        this.messageInput.value = '';
        this.autoResizeInput();
        this.clearThinkingPanel();
        this.thinkingPanel.classList.remove('visible');

        // 取消进行中的请求
        this._abortCurrentRequest();

        // 更新历史记录高亮
        this.updateHistoryHighlight();

        // 移动端关闭侧边栏
        if (window.innerWidth <= 768) {
            this.sidebar.classList.remove('open');
        }
    }

    // 加载对话历史
    loadChatHistory() {
        const saved = localStorage.getItem('mediagent_history');
        if (saved) {
            this.chatHistory = JSON.parse(saved);
            this.renderChatHistory();
        }
    }

    // 保存对话历史
    saveChatHistory() {
        if (this.messages.length > 0) {
            const chatItem = {
                id: this.currentChatId,
                title: this.getChatTitle(),
                timestamp: new Date().toISOString(),
                messages: this.messages
            };

            // 查找是否已存在
            const index = this.chatHistory.findIndex(c => c.id === this.currentChatId);
            if (index >= 0) {
                this.chatHistory[index] = chatItem;
            } else {
                this.chatHistory.unshift(chatItem);
                // 只保留最近 50 个对话
                if (this.chatHistory.length > 50) {
                    this.chatHistory = this.chatHistory.slice(0, 50);
                }
            }

            localStorage.setItem('mediagent_history', JSON.stringify(this.chatHistory));
            this.renderChatHistory();
        }
    }

    // 获取对话标题
    getChatTitle() {
        const firstMessage = this.messages.find(m => m.role === 'user');
        if (firstMessage) {
            const title = firstMessage.content.slice(0, 30);
            return title.length < firstMessage.content.length ? title + '...' : title;
        }
        return '新对话';
    }

    // 渲染对话历史
    renderChatHistory() {
        this.chatHistoryEl.innerHTML = '';

        this.chatHistory.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'history-item' + (chat.id === this.currentChatId ? ' active' : '');
            item.textContent = chat.title || '新对话';
            item.addEventListener('click', () => {
                this.loadChat(chat.id);
            });
            this.chatHistoryEl.appendChild(item);
        });
    }

    // 更新历史记录高亮
    updateHistoryHighlight() {
        document.querySelectorAll('.history-item').forEach(item => {
            item.classList.remove('active');
        });
    }

    // 加载历史对话
    loadChat(chatId) {
        const chat = this.chatHistory.find(c => c.id === chatId);
        if (chat) {
            this.currentChatId = chatId;
            this.messages = chat.messages;
            this.renderMessages();
            this.chatTitle.textContent = chat.title || '新对话';
            this.welcomeScreen.style.display = 'none';
            this.messagesContainer.classList.add('active');

            // 移动端关闭侧边栏
            if (window.innerWidth <= 768) {
                this.sidebar.classList.remove('open');
            }
        }
    }

    // ==================== 消息发送与接收 ====================

    async sendMessage() {
        const content = this.messageInput.value.trim();
        if (!content || this.isLoading) return;

        // 隐藏欢迎界面
        this.welcomeScreen.style.display = 'none';
        this.messagesContainer.classList.add('active');

        // 添加用户消息
        this.addMessage('user', content);
        this.messageInput.value = '';
        this.autoResizeInput();

        // 根据是否启用流式选择不同的发送方式
        if (this.streamingEnabled) {
            await this.sendToAPIStreaming(content);
        } else {
            await this.sendToAPI(content);
        }
    }

    // 添加消息到界面
    addMessage(role, content, thinking = null, requestId = null, isError = false) {
        const message = {
            role,
            content,
            timestamp: new Date().toISOString(),
            thinking,
            requestId,
            isError
        };
        this.messages.push(message);

        // 如果是新对话，生成 ID
        if (!this.currentChatId) {
            this.currentChatId = 'chat_' + Date.now();
        }

        // 更新标题
        if (role === 'user' && this.messages.filter(m => m.role === 'user').length === 1) {
            this.chatTitle.textContent = this.getChatTitle();
        }

        this.renderMessages();
        this.saveChatHistory();
    }

    // 渲染所有消息
    renderMessages() {
        this.messagesContainer.innerHTML = '';

        this.messages.forEach(msg => {
            const el = this.createMessageElement(msg);
            this.messagesContainer.appendChild(el);
        });

        // 滚动到底部
        this.scrollToBottom();
    }

    // 创建消息元素
    createMessageElement(message) {
        const div = document.createElement('div');
        div.className = `message ${message.role}`;
        if (message.isError) {
            div.classList.add('error-message');
        }

        const time = new Date(message.timestamp).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });

        const avatarSVG = message.role === 'assistant'
            ? `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                 <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="white" opacity="0.3"/>
                 <path d="M12 4v16M4 12h16" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
               </svg>`
            : `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                 <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                 <circle cx="12" cy="7" r="4" fill="white"/>
               </svg>`;

        // request_id 显示
        const requestIdHTML = message.requestId
            ? `<span class="message-request-id" title="请求 ID: ${message.requestId}">ID: ${message.requestId.slice(0, 8)}</span>`
            : '';

        // 错误重试按钮
        const retryHTML = message.isError
            ? `<button class="retry-btn" onclick="window.mediAgent.retryMessage('${message.content.replace(/'/g, "\\'")}')">
                 <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                     <path d="M1 4v6h6"/>
                     <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                 </svg>
                 重试
               </button>`
            : '';

        // 风险提示（针对药品查询等外部数据）
        const riskHTML = message.risk_level && message.risk_level !== 'low' && message.risk_level !== 'none'
            ? this.renderRiskWarning(message.risk_level, message.risk_warning, message.source)
            : '';

        // 数据来源标签
        const sourceHTML = message.source
            ? `<span class="source-badge ${this.getSourceClass(message.source)}">${message.source}</span>`
            : '';

        // 缓存命中标记
        const cacheHTML = message.cache_hit
            ? `<span class="cache-hit-badge">⚡ 缓存</span>`
            : '';

        // 推理轨迹（可折叠）
        const traceHTML = message.thinking && message.thinking.length > 0
            ? this.renderReasoningTrace(message.thinking)
            : '';

        div.innerHTML = `
            <div class="message-avatar">${avatarSVG}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-role">${message.role === 'assistant' ? 'MediAgent' : '你'}</span>
                    ${sourceHTML}${cacheHTML}
                    ${requestIdHTML}
                    <span class="message-time">${time}</span>
                    ${retryHTML}
                </div>
                ${riskHTML}
                <div class="message-body">${this.formatMessage(message.content)}</div>
                ${traceHTML}
            </div>
        `;

        return div;
    }

    // 渲染风险提示横幅
    renderRiskWarning(riskLevel, riskWarning, source) {
        if (!riskLevel || riskLevel === 'none') return '';

        const warningText = riskWarning || '⚠️ 信息来自外部数据源，请核实后使用。';
        const icons = {
            high: '🚨',
            medium: '⚠️',
            low: '✓'
        };

        return `
            <div class="risk-warning-banner ${riskLevel}">
                <span class="risk-icon">${icons[riskLevel] || '⚠️'}</span>
                <div class="risk-text">
                    <div>${warningText}</div>
                    ${source ? `<div class="risk-source">数据来源: ${source}</div>` : ''}
                </div>
            </div>
        `;
    }

    // 获取数据来源的CSS类名
    getSourceClass(source) {
        if (source === '内置数据库') return 'internal';
        if (source && source.includes('搜索')) return 'web';
        return 'external';
    }

    // 渲染可折叠推理轨迹
    renderReasoningTrace(thinking) {
        if (!thinking || thinking.length === 0) return '';

        const traceId = 'trace_' + Math.random().toString(36).substr(2, 9);
        const typeLabels = {
            thought: '思考',
            action: '行动',
            observation: '观察',
            plan: '计划',
            error: '错误'
        };

        let stepsHTML = '';
        thinking.forEach((step, index) => {
            const type = step.type || 'thought';
            const label = typeLabels[type] || type;
            stepsHTML += `
                <div class="trace-step trace-step-${type}">
                    <div class="trace-step-header">
                        <span class="trace-step-number">${index + 1}</span>
                        <spa