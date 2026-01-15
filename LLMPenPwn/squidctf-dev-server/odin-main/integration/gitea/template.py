import json
from dataclasses import asdict

from odin.constants import (ConfidenceLevel, ExploitabilityLevel,
                            SeverityLevel, VulnerabilityFinding)

STATIC_ANALYSIS_ISSUE_TEMPLATE = """\
# {title}

> **Summary**
> Odin identified a potential security vulnerability in `{file_path}`. {summary}

## Overview
| Metric | Value |
|:--|:--|
| **Severity** | **{sev}** |
| **Exploitability** | **{exp}** |
| **Confidence** | **{conf}** |

## Technical Details
{report}

## Affected Components
- `{file_path}`

---

<details>
<summary><sub>Raw scanner payload</sub></summary>
```json
{raw_json}
```
</details>""".strip()

def get_static_analysis_issue_body(finding: VulnerabilityFinding) -> str:
    return STATIC_ANALYSIS_ISSUE_TEMPLATE.format(
        title=finding.title,
        file_path=(finding.file_path or "N / A").removeprefix("/opt/resources/"),
        summary=finding.summary,
        sev=finding.severity.value.upper(),
        conf=finding.confidence.value.upper(),
        exp=finding.exploitability.value.upper(),
        report=finding.report,
        raw_json=json.dumps(asdict(finding), indent=2),
    )