class TraceViewer {
    constructor() {
        this.renderer = new TraceRenderer();
        this.currentTrace = null;
        this.theme = 'system';
        this.elements = {};
        this.errorTimeout = null;

        this.initializeEventListeners();
        this.initializeTheme();
        this.loadFromLocalStorage();
    }

    get(id) {
        return this.elements[id] || (this.elements[id] = document.getElementById(id));
    }

    on(target, event, handler) {
        const element = typeof target === 'string' ? this.get(target) : target;
        if (element) element.addEventListener(event, handler);
        return element;
    }

    initializeEventListeners() {
        const fileInput = this.get('fileInput');
        this.on('loadFileBtn', 'click', () => fileInput?.click());
        if (fileInput) {
            this.on(fileInput, 'change', (event) => {
                const file = event.target.files?.[0];
                if (file) this.loadTraceFile(file);
            });
        }

        const dropZone = this.get('dropZone');
        if (dropZone) {
            ['dragover', 'dragleave'].forEach((eventName) => {
                this.on(dropZone, eventName, (event) => {
                    event.preventDefault();
                    dropZone.classList.toggle('drag-over', eventName === 'dragover');
                });
            });

            this.on(dropZone, 'drop', (event) => {
                event.preventDefault();
                dropZone.classList.remove('drag-over');

                const [file] = event.dataTransfer?.files || [];
                if (!file) return;

                if (file.type === 'application/json' || file.name.endsWith('.json')) {
                    this.loadTraceFile(file);
                } else {
                    this.showError('Please select a JSON file');
                }
            });

            this.on(dropZone, 'click', () => fileInput?.click());
        }

        this.on('closeError', 'click', () => this.hideError());
        this.on('pasteBtn', 'click', () => this.showPasteModal());
        this.on('clearBtn', 'click', () => this.clearTrace());

        ['closeModal', 'cancelPaste'].forEach((id) => this.on(id, 'click', () => this.hidePasteModal()));
        this.on('loadPaste', 'click', () => this.loadFromPaste());

        const pasteModal = this.get('pasteModal');
        this.on(pasteModal, 'click', (event) => {
            if (event.target === pasteModal) this.hidePasteModal();
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') this.hidePasteModal();
        });

        this.on('themeSelect', 'change', (event) => this.setTheme(event.target.value));
    }

    async loadTraceFile(file) {
        try {
            this.showLoading();
            const text = await this.readFileAsText(file);
            this.setTrace(TraceParser.parse(text));
        } catch (error) {
            this.showError(`Failed to load trace file: ${error.message}`);
            console.error('Error loading trace file:', error);
        } finally {
            this.hideLoading();
        }
    }

    setTrace(trace, persist = true) {
        this.currentTrace = trace;
        if (persist) this.saveToLocalStorage(trace);
        this.displayTrace(trace);
        this.hideError();
    }

    displayTrace(trace) {
        this.toggleTraceVisibility(true);
        this.renderer.updateStats(trace);
        const conversationFlow = this.get('conversationFlow');
        if (conversationFlow) this.renderer.render(trace, conversationFlow);
        window.scrollTo(0, 0);
    }

    toggleTraceVisibility(show) {
        const dropZone = this.get('dropZone');
        const traceContainer = this.get('traceContainer');
        const clearBtn = this.get('clearBtn');

        if (dropZone) dropZone.style.display = show ? 'none' : 'block';
        if (traceContainer) traceContainer.style.display = show ? 'block' : 'none';
        if (clearBtn) clearBtn.style.display = show ? 'inline-flex' : 'none';
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => resolve(event.target.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    showLoading() {
        const dropZone = this.get('dropZone');
        if (!dropZone) return;
        if (!dropZone.dataset.originalContent) {
            dropZone.dataset.originalContent = dropZone.innerHTML;
        }
        dropZone.innerHTML = '<div class="loading">Loading trace file...</div>';
    }

    hideLoading() {
        const dropZone = this.get('dropZone');
        if (!dropZone || !dropZone.dataset.originalContent) return;
        dropZone.innerHTML = dropZone.dataset.originalContent;
        delete dropZone.dataset.originalContent;
    }

    showError(message) {
        const errorMessage = this.get('errorMessage');
        const errorText = this.get('errorText');
        if (!errorMessage || !errorText) return;

        errorText.textContent = message;
        errorMessage.style.display = 'flex';

        if (this.errorTimeout) clearTimeout(this.errorTimeout);
        this.errorTimeout = setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        const errorMessage = this.get('errorMessage');
        if (errorMessage) errorMessage.style.display = 'none';
        if (this.errorTimeout) {
            clearTimeout(this.errorTimeout);
            this.errorTimeout = null;
        }
    }

    showPasteModal() {
        const modal = this.get('pasteModal');
        const textarea = this.get('pasteTextarea');
        if (!modal || !textarea) return;

        modal.style.display = 'flex';
        textarea.value = '';
        textarea.focus();
    }

    hidePasteModal() {
        const modal = this.get('pasteModal');
        if (modal) modal.style.display = 'none';
    }

    async loadFromPaste() {
        try {
            const textarea = this.get('pasteTextarea');
            if (!textarea) return;

            const jsonData = textarea.value.trim();
            if (!jsonData) {
                this.showError('Please paste some JSON data');
                return;
            }

            this.hidePasteModal();
            this.showLoading();
            this.setTrace(TraceParser.parse(jsonData));
        } catch (error) {
            this.showError(`Failed to load trace from paste: ${error.message}`);
            console.error('Error loading trace from paste:', error);
        } finally {
            this.hideLoading();
        }
    }

    clearTrace() {
        if (!confirm('Are you sure you want to clear the current trace?')) return;

        this.currentTrace = null;
        localStorage.removeItem('traceviewer_latest_trace');
        this.toggleTraceVisibility(false);
        this.hideError();
    }

    exportAsMarkdown() {
        if (!this.currentTrace) {
            this.showError('No trace loaded to export');
            return;
        }

        try {
            const markdown = this.convertTraceToMarkdown(this.currentTrace);
            this.downloadAsFile('agent-trace.md', markdown, 'text/markdown');
        } catch (error) {
            this.showError(`Failed to export trace: ${error.message}`);
        }
    }

    convertTraceToMarkdown(trace) {
        const lines = ['# Agent Trace', ''];
        const stats = trace.stats || {};

        lines.push('## Statistics', '');
        lines.push(`- **Total Cost**: ${TraceParser.formatCurrency(stats.total_cost)}`);
        lines.push(`- **Total Steps**: ${stats.total_steps}`);
        lines.push(`- **Total Tokens**: ${TraceParser.formatNumber(stats.total_tokens)}`, '');

        if (trace.system_prompt) {
            lines.push('## System Prompt', '', '```', trace.system_prompt, '```', '');
        }

        lines.push('## Conversation', '');

        trace.steps.forEach((step, index) => {
            const message = step.message;
            if (message) {
                lines.push(`### Step ${index}: ${this.capitalizeFirst(message.role)}`, '');
                if (message.content) lines.push(`${message.content}`, '');
                if (Array.isArray(message.reasoning) && message.reasoning.length) {
                    lines.push('**Reasoning:**', '');
                    message.reasoning.forEach((thought, thoughtIndex) => {
                        const prefix = message.reasoning.length > 1 ? `${thoughtIndex + 1}. ` : '- ';
                        lines.push(`${prefix}${thought}`, '');
                    });
                }

                const usage = message.token_usage;
                if (usage) {
                    const segments = [
                        usage.input_tokens !== undefined && `${usage.input_tokens} in`,
                        usage.output_tokens !== undefined && `${usage.output_tokens} out`,
                        usage.input_cached_tokens !== undefined && `${usage.input_cached_tokens} cached`,
                        usage.reasoning_tokens !== undefined && `${usage.reasoning_tokens} reasoning`
                    ].filter(Boolean);

                    if (segments.length) lines.push(`*Tokens*: ${segments.join(' · ')}`, '');
                }

                if (Array.isArray(message.tool_calls) && message.tool_calls.length) {
                    lines.push('**Tool Calls:**', '');
                    message.tool_calls.forEach((call) => {
                        lines.push(`- \`${call.name}()\``);
                        if (call.arguments) {
                            lines.push('  ```json', `  ${JSON.stringify(call.arguments, null, 2)}`, '  ```');
                        }
                    });
                    lines.push('');
                }
            }

            if (step.tool_results) {
                Object.values(step.tool_results).forEach((result) => {
                    lines.push('**Tool Result:**', '');
                    if (result.output) {
                        lines.push('```', result.output, '```', '');
                    }
                });
            }
        });

        return lines.join('\n');
    }

    downloadAsFile(filename, content, mimeType = 'text/plain') {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');

        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        URL.revokeObjectURL(url);
    }

    capitalizeFirst(str) {
        return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
    }

    saveToLocalStorage(trace) {
        try {
            localStorage.setItem('traceviewer_latest_trace', JSON.stringify(trace));
        } catch (error) {
            console.warn('Failed to save trace to localStorage:', error);
        }
    }

    loadFromLocalStorage() {
        try {
            const traceData = localStorage.getItem('traceviewer_latest_trace');
            if (!traceData) return;

            const trace = JSON.parse(traceData);
            this.setTrace(trace, false);
        } catch (error) {
            console.warn('Failed to load trace from localStorage:', error);
        }
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('traceviewer_theme');
        if (savedTheme) this.theme = savedTheme;
        this.applyTheme();
        this.updateThemeDropdown();

        this.systemThemeListener = () => {
            if (this.theme === 'system') this.applyTheme();
        };
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', this.systemThemeListener);
    }

    setTheme(theme) {
        this.theme = theme;
        this.applyTheme();
        this.updateThemeDropdown();
        localStorage.setItem('traceviewer_theme', theme);
    }

    applyTheme() {
        const root = document.documentElement;
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const useDark = this.theme === 'dark' || (this.theme === 'system' && prefersDark);

        if (useDark) {
            root.setAttribute('data-theme', 'dark');
        } else {
            root.removeAttribute('data-theme');
        }
    }

    updateThemeDropdown() {
        const select = this.get('themeSelect');
        if (select) select.value = this.theme;
    }

    reset() {
        this.currentTrace = null;
        this.toggleTraceVisibility(false);
        this.hideError();
    }
}

window.TraceViewer = TraceViewer;

document.addEventListener('DOMContentLoaded', () => {
    const app = new TraceViewer();
    window.app = app;
    console.log('Traceviewer initialized');
});
