class TraceParser {
    static validate(data) {
        if (!data || typeof data !== 'object') throw new Error('Invalid JSON data');
        if (!data.system_prompt) throw new Error('Missing system_prompt field');
        if (!Array.isArray(data.tools)) throw new Error('Missing or invalid tools array');
        if (!Array.isArray(data.steps)) throw new Error('Missing or invalid steps array');
    }

    static parse(jsonData) {
        try {
            const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;
            this.validate(data);
            return this.processTrace(data);
        } catch (error) {
            throw new Error(`Parse error: ${error.message}`);
        }
    }

    static processTrace(data) {
        const { system_prompt, tools, steps } = data;
        return {
            system_prompt,
            tools: this.processTools(tools),
            steps: this.processSteps(steps),
            stats: this.calculateStats(data)
        };
    }

    static processTools(tools) {
        return tools.map(({ name, description, inputs = {}, output_type }) => ({
            name,
            description,
            inputs,
            output_type
        }));
    }

    static processSteps(steps) {
        return steps.map((step) => ({
            step_number: step.step_number,
            message: this.processMessage(step.message),
            tool_results: this.processToolResults(step.tool_results),
            error: step.error
        }));
    }

    static processMessage(message) {
        if (!message) return null;
        const { role, content, reasoning, tool_calls, tool_call_id, token_usage } = message;
        return {
            role,
            content,
            reasoning: Array.isArray(reasoning) ? reasoning.map(String) : null,
            tool_calls: this.processToolCalls(tool_calls),
            tool_call_id,
            token_usage
        };
    }

    static processToolCalls(toolCalls) {
        if (!Array.isArray(toolCalls)) return null;
        return toolCalls.map(({ id, name, arguments: args }) => ({
            id,
            name,
            arguments: args
        }));
    }

    static processToolResults(results) {
        if (!results || typeof results !== 'object') return null;
        return Object.fromEntries(
            Object.entries(results).map(([id, result]) => [
                id,
                {
                    id: result.id,
                    output: result.output,
                    observation: result.observation,
                    tool_call: result.tool_call
                }
            ])
        );
    }

    static calculateStats(data) {
        const stats = {
            total_cost: 0,
            total_tokens: 0,
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            reasoning_tokens: 0,
            total_steps: data.steps.length,
            tools_used: this.getUniqueToolsUsed(data.steps)
        };

        const INPUT_COST_PER_1K = 0.03;
        const OUTPUT_COST_PER_1K = 0.06;

        data.steps.forEach(({ message }) => {
            const usage = message?.token_usage;
            if (!usage) return;

            const {
                input_tokens,
                output_tokens,
                input_cached_tokens,
                reasoning_tokens,
                input_cost,
                output_cost
            } = usage;

            if (input_tokens) stats.input_tokens += input_tokens;
            if (output_tokens) stats.output_tokens += output_tokens;
            if (input_cached_tokens) stats.cached_tokens += input_cached_tokens;
            if (reasoning_tokens) stats.reasoning_tokens += reasoning_tokens;

            if (input_cost !== undefined) {
                stats.total_cost += input_cost;
            } else if (input_tokens) {
                stats.total_cost += (input_tokens / 1000) * INPUT_COST_PER_1K;
            }

            if (output_cost !== undefined) {
                stats.total_cost += output_cost;
            } else if (output_tokens) {
                stats.total_cost += (output_tokens / 1000) * OUTPUT_COST_PER_1K;
            }
        });

        stats.total_tokens = stats.input_tokens + stats.output_tokens;
        return stats;
    }

    static getUniqueToolsUsed(steps) {
        const tools = new Set();
        steps.forEach(({ message }) => {
            message?.tool_calls?.forEach(({ name }) => tools.add(name));
        });
        return Array.from(tools);
    }

    static formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 3,
            maximumFractionDigits: 3
        }).format(amount ?? 0);
    }

    static formatNumber(num) {
        return new Intl.NumberFormat('en-US').format(num ?? 0);
    }

    static getMessageType(message) {
        const role = message?.role;
        return ['system', 'user', 'assistant', 'tool'].includes(role) ? role : 'unknown';
    }

    static getMessageIcon(messageType) {
        const icons = { system: '⚙️', user: '👤', assistant: '🤖', tool: '🔧', unknown: '❓' };
        return icons[messageType] || icons.unknown;
    }

    static formatTimestamp(timestamp) {
        if (!timestamp) return '';
        try {
            return new Date(timestamp).toLocaleTimeString();
        } catch {
            return String(timestamp);
        }
    }
}
