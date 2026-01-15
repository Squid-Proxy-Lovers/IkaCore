class TraceRenderer {
    constructor() {
        this.expandedToolCalls = new Set();
        this.expandedToolResults = new Set();
        this.TOOL_RESULT_PREVIEW_LENGTH = 500;
    }

    static el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = text;
        return node;
    }

    static prefilled(text) {
        const pre = document.createElement('pre');
        pre.textContent = text;
        return pre;
    }

    static append(parent, ...children) {
        children.forEach((child) => {
            if (Array.isArray(child)) {
                child.forEach((nested) => nested && parent.appendChild(nested));
            } else if (child) {
                parent.appendChild(child);
            }
        });
        return parent;
    }

    render(trace, container) {
        if (!trace || !container) throw new Error('Invalid trace data or container');
        container.innerHTML = '';

        if (trace.system_prompt) container.appendChild(this.createSystemMessage(trace.system_prompt));
        if (trace.tools?.length) container.appendChild(this.createToolsSection(trace.tools));
        trace.steps.forEach((step, index) => container.appendChild(this.renderStep(step, index)));
    }

    createSystemMessage(systemPrompt) {
        const message = TraceRenderer.el('div', 'message system');
        const header = this.createHeader(TraceParser.getMessageIcon('system'), 'System Message', [
            TraceRenderer.el('span', '', 'Initial Setup')
        ]);
        const content = TraceRenderer.el('div', 'message-content');
        content.appendChild(TraceRenderer.prefilled(systemPrompt));
        return TraceRenderer.append(message, header, content);
    }

    createToolsSection(tools) {
        const container = TraceRenderer.el('div', 'tools-section');
        const header = TraceRenderer.el('div', 'tools-header');
        const headerContent = TraceRenderer.el('div', 'tools-header-content');

        TraceRenderer.append(
            headerContent,
            TraceRenderer.el('div', 'tools-icon', '🛠️'),
            TraceRenderer.el('h3', '', 'Available Tools')
        );

        const toolCount = TraceRenderer.el(
            'span',
            'tool-count',
            `${tools.length} tool${tools.length !== 1 ? 's' : ''}`
        );

        const list = TraceRenderer.el('div', 'tools-list');
        tools.forEach((tool, index) => list.appendChild(this.createToolItem(tool, index)));

        TraceRenderer.append(header, headerContent, toolCount);
        return TraceRenderer.append(container, header, list);
    }

    createToolItem(tool, index) {
        const toolItem = TraceRenderer.el('div', 'tool-item');
        const toolId = `tool-${index}`;
        if (this.expandedToolCalls.has(toolId)) toolItem.classList.add('expanded');

        const header = TraceRenderer.el('div', 'tool-header');
        header.onclick = () => this.toggleTool(toolId, toolItem);
        header.appendChild(this.createExpandIcon());
        header.appendChild(TraceRenderer.el('span', 'tool-name', tool.name));

        if (tool.description) {
            const truncated = tool.description.length > 100
                ? `${tool.description.slice(0, 97)}...`
                : tool.description;
            header.appendChild(TraceRenderer.el('span', 'tool-description-inline', truncated));
        }

        const content = TraceRenderer.el('div', 'tool-content');

        if (tool.description) {
            const section = TraceRenderer.el('div', 'tool-section');
            TraceRenderer.append(
                section,
                TraceRenderer.el('h4', '', 'Description'),
                TraceRenderer.el('p', '', tool.description)
            );
            content.appendChild(section);
        }

        const inputs = tool.inputs && Object.keys(tool.inputs).length ? tool.inputs : null;
        if (inputs) {
            const section = TraceRenderer.el('div', 'tool-section');
            section.appendChild(TraceRenderer.el('h4', '', 'Arguments'));
            const list = TraceRenderer.el('div', 'tool-inputs');

            Object.entries(inputs).forEach(([key, value]) => {
                const item = TraceRenderer.el('div', 'input-item');
                TraceRenderer.append(
                    item,
                    TraceRenderer.el('span', 'input-name', key),
                    TraceRenderer.el(
                        'span',
                        'input-description',
                        typeof value === 'object' ? JSON.stringify(value) : String(value)
                    )
                );
                list.appendChild(item);
            });

            section.appendChild(list);
            content.appendChild(section);
        }

        if (tool.output_type) {
            const section = TraceRenderer.el('div', 'tool-section');
            TraceRenderer.append(
                section,
                TraceRenderer.el('h4', '', 'Output Type'),
                TraceRenderer.el('p', '', tool.output_type)
            );
            content.appendChild(section);
        }

        return TraceRenderer.append(toolItem, header, content);
    }

    toggleTool(toolId, element) {
        if (this.expandedToolCalls.has(toolId)) {
            this.expandedToolCalls.delete(toolId);
            element.classList.remove('expanded');
        } else {
            this.expandedToolCalls.add(toolId);
            element.classList.add('expanded');
        }
    }

    renderStep(step, index) {
        const container = TraceRenderer.el('div', 'step-container');
        if (step.message) container.appendChild(this.renderMessage(step.message, index));
        if (step.tool_results) container.appendChild(this.renderToolResults(step.tool_results));
        if (step.error) container.appendChild(this.renderError(step.error));
        return container;
    }

    renderMessage(message, stepIndex) {
        const messageType = TraceParser.getMessageType(message);
        const wrapper = TraceRenderer.el('div', `message ${messageType}`);
        wrapper.appendChild(this.createMessageHeader(message, messageType, stepIndex));

        const reasoning = Array.isArray(message.reasoning) && message.reasoning.length
            ? this.createReasoningSection(message.reasoning)
            : null;
        const content = message.content ? this.createMessageContent(message.content) : null;

        if (message.role === 'assistant' && reasoning && content) {
            wrapper.appendChild(reasoning);
            wrapper.appendChild(content);
        } else {
            if (content) wrapper.appendChild(content);
            if (reasoning) wrapper.appendChild(reasoning);
        }

        if (message.tool_calls?.length) wrapper.appendChild(this.renderToolCalls(message.tool_calls));
        return wrapper;
    }

    createMessageHeader(message, messageType, stepIndex) {
        const meta = [];
        if (message.token_usage) {
            const tokenUsage = TraceRenderer.el('span', 'token-usage');
            tokenUsage.textContent = this.formatTokenUsage(message.token_usage);
            meta.push(tokenUsage);
        }
        meta.push(TraceRenderer.el('span', '', `Step ${stepIndex}`));
        return this.createHeader(
            TraceParser.getMessageIcon(messageType),
            this.getRoleDisplayName(message.role),
            meta
        );
    }

    createHeader(iconText, label, metaChildren = []) {
        const header = TraceRenderer.el('div', 'message-header');
        const role = TraceRenderer.el('div', 'message-role');
        TraceRenderer.append(
            role,
            TraceRenderer.el('div', 'message-icon', iconText),
            TraceRenderer.el('span', '', label)
        );

        const meta = TraceRenderer.el('div', 'message-meta');
        metaChildren.filter(Boolean).forEach((child) => meta.appendChild(child));

        return TraceRenderer.append(header, role, meta);
    }

    createMessageContent(content) {
        const container = TraceRenderer.el('div', 'message-content');
        container.appendChild(TraceRenderer.prefilled(content));
        return container;
    }

    createReasoningSection(reasoning) {
        const details = TraceRenderer.el('details', 'reasoning-section');
        details.appendChild(TraceRenderer.el('summary', '', `Reasoning (${reasoning.length})`));

        const wrapper = TraceRenderer.el('div', 'reasoning-content');
        reasoning.forEach((chunk, index) => {
            const entry = TraceRenderer.el('div', 'reasoning-entry');
            if (reasoning.length > 1) {
                entry.appendChild(TraceRenderer.el('div', 'reasoning-label', `Thought ${index + 1}`));
            }
            const markdown = TraceRenderer.el('div', 'markdown-body');
            markdown.innerHTML = this.renderMarkdown(String(chunk));
            entry.appendChild(markdown);
            wrapper.appendChild(entry);
        });

        details.appendChild(wrapper);
        return details;
    }

    renderToolCalls(toolCalls) {
        const container = TraceRenderer.el('div', 'tool-calls');
        toolCalls.forEach((call) => container.appendChild(this.createToolCall(call)));
        return container;
    }

    createToolCall(toolCall) {
        const call = TraceRenderer.el('div', 'tool-call');
        const callId = toolCall.id;
        if (this.expandedToolCalls.has(callId)) call.classList.add('expanded');

        const header = TraceRenderer.el('div', 'tool-call-header');
        header.onclick = () => this.toggleToolCall(callId, call);
        header.appendChild(this.createExpandIcon());

        const inline = TraceRenderer.el('span', 'function-call');
        inline.innerHTML = this.formatToolCallInline(toolCall);
        header.appendChild(inline);

        const content = TraceRenderer.el('div', 'tool-call-content');
        if (toolCall.arguments) {
            const block = TraceRenderer.el('div', 'tool-input');
            block.appendChild(TraceRenderer.el('h4', '', 'Arguments'));
            block.appendChild(TraceRenderer.prefilled(JSON.stringify(toolCall.arguments, null, 2)));
            content.appendChild(block);
        }

        return TraceRenderer.append(call, header, content);
    }

    toggleToolCall(callId, element) {
        if (this.expandedToolCalls.has(callId)) {
            this.expandedToolCalls.delete(callId);
            element.classList.remove('expanded');
        } else {
            this.expandedToolCalls.add(callId);
            element.classList.add('expanded');
        }
    }

    renderToolResults(toolResults) {
        const container = TraceRenderer.el('div', 'tool-results');
        Object.values(toolResults).forEach((result) => container.appendChild(this.createToolResult(result)));
        return container;
    }

    createToolResult(result) {
        const message = TraceRenderer.el('div', 'message tool');
        const meta = result.tool_call?.name ? [TraceRenderer.el('span', '', result.tool_call.name)] : [];
        const header = this.createHeader('🔧', 'Tool Result', meta);

        const content = TraceRenderer.el('div', 'message-content');
        if (result.output) {
            const section = TraceRenderer.el('div', 'tool-output');
            section.appendChild(TraceRenderer.el('h4', '', 'Output'));
            const formatted = this.formatToolOutput(result.output);
            section.appendChild(this.createExpandableOutput(formatted, result.id));
            content.appendChild(section);
        }

        if (result.observation) {
            const section = TraceRenderer.el('div', 'tool-observation');
            section.appendChild(TraceRenderer.el('h4', '', 'Observation'));
            section.appendChild(TraceRenderer.prefilled(result.observation));
            content.appendChild(section);
        }

        return TraceRenderer.append(message, header, content);
    }

    renderError(error) {
        const message = TraceRenderer.el('div', 'message error');
        const header = this.createHeader('❌', 'Error');
        const content = TraceRenderer.el('div', 'message-content');
        const payload = typeof error === 'string' ? error : JSON.stringify(error, null, 2);
        content.appendChild(TraceRenderer.prefilled(payload));
        return TraceRenderer.append(message, header, content);
    }

    createExpandIcon() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        Object.entries({
            class: 'expand-icon',
            width: '16',
            height: '16',
            viewBox: '0 0 24 24',
            fill: 'none',
            stroke: 'currentColor',
            'stroke-width': '2'
        }).forEach(([key, value]) => svg.setAttribute(key, value));

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'm9 18 6-6-6-6');
        svg.appendChild(path);
        return svg;
    }

    formatToolCallInline(toolCall) {
        let result = `${toolCall.name}(`;
        if (toolCall.arguments && Object.keys(toolCall.arguments).length) {
            const args = Object.entries(toolCall.arguments).map(([key, value]) => {
                let valueStr = typeof value === 'string' ? `"${value}"` : JSON.stringify(value);
                if (valueStr.length > 50) valueStr = `${valueStr.slice(0, 47)}...`;
                return `<span class="arg-name">${key}</span>: <span class="arg-value">${valueStr}</span>`;
            });
            result += args.join(', ');
        }
        return `${result})`;
    }

    formatTokenUsage(usage) {
        const parts = [
            ['input_tokens', 'in'],
            ['output_tokens', 'out'],
            ['input_cached_tokens', 'cached'],
            ['reasoning_tokens', 'reasoning']
        ]
            .map(([key, label]) => usage[key] !== undefined && `${TraceParser.formatNumber(Number(usage[key]) || 0)} ${label}`)
            .filter(Boolean);

        return parts.length ? `Tokens: ${parts.join(' · ')}` : 'Tokens: —';
    }

    getRoleDisplayName(role) {
        const names = {
            system: 'System Message',
            user: 'User Message',
            assistant: 'Assistant Message',
            tool: 'Tool Result'
        };
        return names[role] || role;
    }

    formatToolOutput(output) {
        if (typeof output === 'string') {
            try {
                return JSON.stringify(JSON.parse(output), null, 2);
            } catch {
                return output;
            }
        }
        return JSON.stringify(output, null, 2);
    }

    createExpandableOutput(output, resultId) {
        if (output.length <= this.TOOL_RESULT_PREVIEW_LENGTH) {
            return TraceRenderer.prefilled(output);
        }

        const container = TraceRenderer.el('div', 'expandable-output');
        const preview = output.slice(0, this.TOOL_RESULT_PREVIEW_LENGTH);
        const remainder = output.slice(this.TOOL_RESULT_PREVIEW_LENGTH);

        const pre = TraceRenderer.el('pre', 'expandable-content');
        const previewSpan = TraceRenderer.el('span', null, preview);
        const remainderSpan = TraceRenderer.el('span', 'expandable-remainder', remainder);
        remainderSpan.style.display = 'none';

        pre.appendChild(previewSpan);
        pre.appendChild(remainderSpan);

        const button = TraceRenderer.el('button', 'expand-button');
        const moreText = `... Show ${remainder.length} more characters`;
        button.textContent = moreText;
        button.dataset.moreText = moreText;
        button.onclick = () => this.toggleToolResult(resultId, button, remainderSpan);

        container.appendChild(pre);
        container.appendChild(button);
        return container;
    }

    toggleToolResult(resultId, button, remainderSpan) {
        if (this.expandedToolResults.has(resultId)) {
            this.expandedToolResults.delete(resultId);
            remainderSpan.style.display = 'none';
            button.textContent = button.dataset.moreText;
        } else {
            this.expandedToolResults.add(resultId);
            remainderSpan.style.display = 'inline';
            button.textContent = 'Show less';
        }
    }

    updateStats(trace) {
        const stats = trace.stats;
        const title = document.getElementById('traceTitle');
        if (title) title.textContent = 'Agent Conversation';

        const costValue = document.querySelector('#totalCost .stat-value');
        if (costValue && stats.total_cost !== undefined) {
            costValue.textContent = TraceParser.formatCurrency(stats.total_cost);
        }

        const stepsValue = document.querySelector('#totalSteps .stat-value');
        if (stepsValue) stepsValue.textContent = String(stats.total_steps);

        const tokensElement = document.getElementById('totalTokens');
        if (tokensElement) {
            const value = tokensElement.querySelector('.stat-value');
            const detail = tokensElement.querySelector('.stat-detail');

            if (value) value.textContent = TraceParser.formatNumber(stats.total_tokens || 0);

            if (detail) {
                const segments = [
                    `${TraceParser.formatNumber(stats.input_tokens || 0)} in`,
                    `${TraceParser.formatNumber(stats.output_tokens || 0)} out`
                ];
                if (stats.cached_tokens) {
                    segments.push(`${TraceParser.formatNumber(stats.cached_tokens)} cached`);
                }
                if (stats.reasoning_tokens) {
                    segments.push(`${TraceParser.formatNumber(stats.reasoning_tokens)} reasoning`);
                }
                detail.textContent = segments.join(' · ');
            }
        }
    }

    renderMarkdown(text) {
        if (!text) return '';

        const escapeHtml = (str) => str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');

        const renderInline = (str) => str
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/_(.+?)_/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        const lines = escapeHtml(text).split(/\r?\n/);
        let html = '';
        let inCodeBlock = false;
        const codeBuffer = [];
        const listBuffers = { ul: [], ol: [] };

        const flushCode = () => {
            if (!inCodeBlock) return;
            html += `<pre><code>${codeBuffer.join('\n')}</code></pre>`;
            codeBuffer.length = 0;
            inCodeBlock = false;
        };

        const flushLists = () => {
            if (listBuffers.ul.length) {
                html += `<ul>${listBuffers.ul.join('')}</ul>`;
                listBuffers.ul.length = 0;
            }
            if (listBuffers.ol.length) {
                html += `<ol>${listBuffers.ol.join('')}</ol>`;
                listBuffers.ol.length = 0;
            }
        };

        const headings = [
            [/^######\s+/, 'h6'],
            [/^#####\s+/, 'h5'],
            [/^####\s+/, 'h4'],
            [/^###\s+/, 'h3'],
            [/^##\s+/, 'h2'],
            [/^#\s+/, 'h1']
        ];

        lines.forEach((line) => {
            if (/^```/.test(line)) {
                if (inCodeBlock) {
                    flushCode();
                } else {
                    flushLists();
                    inCodeBlock = true;
                }
                return;
            }

            if (inCodeBlock) {
                codeBuffer.push(line);
                return;
            }

            if (/^\s*[-*+]\s+/.test(line)) {
                listBuffers.ol.length = 0;
                listBuffers.ul.push(`<li>${renderInline(line.replace(/^\s*[-*+]\s+/, ''))}</li>`);
                return;
            }

            if (/^[0-9]+\.\s+/.test(line)) {
                listBuffers.ul.length = 0;
                listBuffers.ol.push(`<li>${renderInline(line.replace(/^[0-9]+\.\s+/, ''))}</li>`);
                return;
            }

            flushLists();

            let handled = false;
            for (const [regex, tag] of headings) {
                if (regex.test(line)) {
                    html += `<${tag}>${renderInline(line.replace(regex, ''))}</${tag}>`;
                    handled = true;
                    break;
                }
            }
            if (handled) return;

            if (/^>\s?/.test(line)) {
                html += `<blockquote>${renderInline(line.replace(/^>\s?/, ''))}</blockquote>`;
            } else if (!line.trim()) {
                html += '<br />';
            } else {
                html += `<p>${renderInline(line)}</p>`;
            }
        });

        flushCode();
        flushLists();
        return html;
    }
}
