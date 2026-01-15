# Traceviewer 🤖

A modern, responsive web application for visualizing agent conversation traces from JSON files. Built as a static single-page application with no dependencies and designed to surface rich reasoning context.

## Features

### Core Functionality
- 📁 **File Loading**: Drag & drop JSON trace files or use the file picker
- 💾 **Auto-Save**: Automatically saves and restores your latest trace on page refresh
- 💬 **Conversation View**: Sequential message display with role-based styling
- 🧠 **Reasoning Timeline**: Collapsible, markdown-aware reasoning panels surface internal thoughts ahead of tool usage
- 🔧 **Tool Inspection**: Expandable tool calls with inline argument preview and full details
- 📊 **Enhanced Statistics**: Accurate cost tracking with input/output, cached, and reasoning token visibility

### User Experience
- 🌙 **Dark Mode**: System-aware theme with dropdown selector (🌐 System/☀️ Light/🌙 Dark)
- 📱 **Fully Responsive**: Optimized for desktop, tablet, and mobile devices with smart breakpoints
- ✨ **Smart Text Handling**: Automatic text wrapping and collapsible long outputs (>500 chars)
- 🎨 **Modern UI**: Glassmorphism-inspired layout, refreshed stats cards, and updated typography
- 🔍 **Tool Insights**: Comprehensive system prompt with tool descriptions and specifications
- 🛠️ **Enhanced Tool Calls**: Inline argument preview with responsive truncation and syntax highlighting

### Technical Features
- ⚡ **Static Build**: No server required - runs entirely client-side
- 📦 **Zero Dependencies**: Pure HTML/CSS/JavaScript implementation
- 🚀 **Performance**: Efficient rendering with smart truncation and lazy expansion
- 💻 **Cross-Platform**: Works in all modern browsers

## Usage

### Getting Started
1. **Open the Application**
   - Open `index.html` in any modern web browser
   - No installation or server setup required

2. **Load a Trace File**
   - Click "Load Trace File" to select a JSON file
   - Or drag & drop a JSON file onto the drop zone
   - Your trace will automatically be saved for next visit

3. **Explore the Trace**
   - View the enhanced system prompt with comprehensive tool descriptions
   - Navigate through conversation steps with improved message layout and proper spacing
   - Expand reasoning panels to inspect markdown-formatted thoughts before tool calls
   - Click tool calls to see inline arguments with syntax highlighting or expand for full details
   - Long tool outputs (>500 characters) automatically collapse with "Show more" buttons
   - Use the theme dropdown to switch between System/Light/Dark modes

### Theme Options
The theme selector dropdown in the top-right corner provides three options:
- **🌐 System**: Automatically follows your device's theme preference
- **☀️ Light**: Always use light theme
- **🌙 Dark**: Always use dark theme with improved contrast and readability

## File Structure

```
traceviewer/
├── index.html              # Main application
├── css/
│   └── styles.css          # Enhanced styling with CSS variables and dark mode
├── js/
│   ├── main.js            # Application logic with theme management
│   ├── parser.js          # JSON parsing with enhanced cost calculation
│   └── renderer.js        # UI rendering with markdown reasoning and enhanced UX
└── README.md             # This file
```

## JSON Trace Format

The application expects JSON files with the following structure:

```json
{
  "system_prompt": "System instructions...",
  "tools": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "inputs": { 
        "param_name": "parameter description or type"
      },
      "output_type": "Description of output format"
    }
  ],
  "steps": [
    {
      "step_number": 0,
      "message": {
        "role": "user|assistant|system|tool",
        "content": "Message content",
        "reasoning": [
          "Optional markdown-formatted reasoning text as a list of strings",
          "Each entry renders as its own expandable thought"
        ],
        "tool_calls": [
          {
            "id": "call_id",
            "name": "tool_name", 
            "arguments": { "param": "value" }
          }
        ],
        "token_usage": {
          "input_tokens": 100,
          "output_tokens": 50,
          "input_cached_tokens": 25,
          "reasoning_tokens": 12,
          "input_cost": 0.003,    // Optional: actual cost values
          "output_cost": 0.006    // Will fallback to calculation if not provided
        }
      },
      "tool_results": {
        "call_id": {
          "id": "call_id",
          "output": "Tool response...",
          "tool_call": { "name": "tool_name" }
        }
      },
      "error": null
    }
  ]
}
```

### Message & Token Dataclasses

Traceviewer follows a structured conversation model that mirrors the application data classes:

```python
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    input_cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    input_cost: float = 0.0
    output_cost: float = 0.0


@dataclass
class Message:
    role: MessageRole
    content: Optional[str] = None
    reasoning: Optional[list[str]] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
```

In the UI, reasoning entries appear before tool calls in expandable sections that honor markdown formatting, and token usage now surfaces cached and reasoning tokens alongside input/output values whenever they are provided.

### Enhanced Cost Calculation
The application now supports both:
- **Provided Costs**: Uses `input_cost` and `output_cost` fields when available
- **Calculated Costs**: Falls back to approximate GPT-4 pricing for estimation

## Browser Compatibility

- Chrome/Edge 80+
- Firefox 75+ 
- Safari 13+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Responsive Breakpoints

- **Desktop**: > 1024px - Full layout
- **Tablet**: 768px - 1024px - Adapted spacing
- **Mobile**: 480px - 768px - Stacked layout
- **Small Mobile**: < 480px - Compact design

## Development

This is a modern vanilla HTML/CSS/JavaScript application with no build process required. The codebase uses:

- **CSS Custom Properties**: For theming and maintainable styles
- **ES6+ Features**: Modern JavaScript with class-based architecture
- **Local Storage API**: For persistent user preferences
- **Media Query API**: For system theme detection

### Key Components

- **TraceParser**: Enhanced JSON validation and processing with cost calculation
- **TraceRenderer**: Advanced UI rendering with markdown reasoning, expandable content, and theme support
- **TraceViewer**: Main application controller with theme and storage management

### Architecture Decisions

- **Static-First**: No build tools or dependencies to maintain simplicity
- **Progressive Enhancement**: Core functionality works, enhancements add polish
- **Responsive-Native**: Built mobile-first with desktop enhancements
- **Theme-Aware**: Respects user preferences while providing manual control

## License

MIT License - feel free to use and modify as needed.
