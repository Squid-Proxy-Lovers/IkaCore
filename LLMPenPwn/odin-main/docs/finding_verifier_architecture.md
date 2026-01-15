# Finding Verifier Architecture

## Overview

The Finding Verifier validates vulnerability findings before submission, ensuring quality and accuracy with up to 3 retry cycles for corrections.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Agent["MultiStepAgent"]
        LLM["LLM Model"]
        Tools["Tool Registry"]
    end
    subgraph Workflow["StaticWorkflow"]
        CLI["CLI Arguments"]
        Config["Verification Config"]
    end
    subgraph FindingTool["AddVulnerabilityFindingTool"]
        Forward["forward()"]
        RetryTracker["Retry Tracker"]
        Callback["finding_callback()"]
        FindingsList["findings list"]
    end
    subgraph Verifier["FindingVerifier"]
        Verify["verify()"]
        subgraph Checks["Verification Checks"]
            FileCheck["File Path Validation"]
            SectionCheck["Required Sections Check"]
            CodeCheck["Code Snippet Verification"]
            LLMReview["LLM Quality Review"]
            KillChain["Kill Chain Validation"]
        end
        Aggregate["Aggregate Results"]
    end
    subgraph Result["VerificationResult"]
        Passed["passed: bool"]
        Errors["errors: list"]
        Warnings["warnings: list"]
        Suggestions["suggestions: list"]
    end
    CLI --> Config
    Config --> FindingTool
    LLM -->|tool_call| Forward
    Forward --> Verify
    Verify --> FileCheck
    Verify --> SectionCheck
    Verify --> CodeCheck
    Verify --> LLMReview
    Verify --> KillChain
    FileCheck --> Aggregate
    SectionCheck --> Aggregate
    CodeCheck --> Aggregate
    LLMReview --> Aggregate
    KillChain --> Aggregate
    Aggregate --> Result
    Result -->|passed=true| Callback
    Callback --> FindingsList
    FindingsList -->|Success| LLM
    Result -->|failed retries less than 3| RetryTracker
    RetryTracker -->|Error Message| LLM
    Result -->|failed retries 3 or more| Callback
```

## Retry Cycle Flow

```mermaid
sequenceDiagram
    participant A as Agent LLM
    participant T as FindingTool
    participant V as Verifier
    participant C as Callback
    A->>T: add_vulnerability_finding
    T->>V: verify finding
    V-->>T: FAILED errors
    T-->>A: VERIFICATION FAILED 1 of 3
    Note over A: Agent fixes issues
    A->>T: add_vulnerability_finding fixed
    T->>V: verify finding
    V-->>T: FAILED errors
    T-->>A: VERIFICATION FAILED 2 of 3
    Note over A: Agent fixes issues
    A->>T: add_vulnerability_finding fixed
    T->>V: verify finding
    V-->>T: FAILED errors
    T-->>A: VERIFICATION FAILED 3 of 3
    Note over A: Agent attempts fix
    A->>T: add_vulnerability_finding fixed
    T->>V: verify finding
    V-->>T: FAILED still has errors
    Note over T: Max retries reached
    T->>C: Accept with warnings
    T-->>A: Successfully added finding
```

## Component Relationships

```mermaid
classDiagram
    class StaticWorkflow {
        +add_arguments(group)
        +run(env) List
        -create_model()
        -create_agent()
    }
    class AddVulnerabilityFindingTool {
        +name str
        +description str
        +inputs dict
        -findings List
        -verifier FindingVerifier
        +forward() str
    }
    class FindingVerifier {
        +REQUIRED_SECTIONS list
        -code_root str
        -verification_model Model
        +verify(finding) VerificationResult
    }
    class VerificationResult {
        +passed bool
        +errors List
        +warnings List
        +suggestions List
    }
    class VulnerabilityFinding {
        +title str
        +report str
        +file_path str
        +severity SeverityLevel
    }
    StaticWorkflow --> AddVulnerabilityFindingTool : creates
    AddVulnerabilityFindingTool --> FindingVerifier : uses
    AddVulnerabilityFindingTool --> VulnerabilityFinding : creates
    FindingVerifier --> VerificationResult : returns
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--enable-verification` | `true` | Enable finding verification |
| `--disable-verification` | - | Disable verification |
| `--enable-kill-chain` | `false` | Execute exploit code to validate |
| `--max-verification-retries` | `3` | Max retry attempts |
| `--verification-model` | `gpt-5-nano` | Model for LLM review |

## Verification Checks Summary

| Check | Purpose | On Failure |
|-------|---------|------------|
| **File Path** | Verify file exists in codebase | Error + similar file suggestions |
| **Required Sections** | Ensure all 6 sections present | Error listing missing sections |
| **Code Snippets** | Match code blocks to source | Warning or error |
| **LLM Review** | Quality and accuracy check | Errors + suggestions |
| **Kill Chain** | Execute exploit code | Error if execution fails |

## Required Report Sections

1. **Overview** - Brief description of the vulnerability
2. **Where it occurs** - File path, function, line number
3. **Vulnerability Details** - Technical explanation
4. **Impact** - Business/security impact
5. **Steps to Reproduce** - Exploit instructions
6. **Remediation** - How to fix
