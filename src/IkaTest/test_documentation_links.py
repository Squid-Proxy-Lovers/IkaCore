"""Check repository-relative Markdown file links and heading anchors, including archives."""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def _without_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _anchors(path):
    text = _without_fences(path.read_text())
    seen, anchors = {}, set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        anchors.add(slug + (f"-{count}" if count else ""))
        seen[slug] = count + 1
    anchors.update(re.findall(r'(?:id|name)="([^"]+)"', text))
    return anchors


def test_local_documentation_links_resolve():
    files = [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md"),
             *ROOT.glob("examples/**/*.md"), *ROOT.glob("src/IkaMem/docs/**/*.md")]
    failures = []
    for path in files:
        text = _without_fences(path.read_text())
        for destination in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text):
            if re.match(r"[a-zA-Z][\w+.-]*:", destination):
                continue
            filename, _, fragment = unquote(destination).partition("#")
            target = (path.parent / filename).resolve() if filename else path
            if not target.exists():
                failures.append(f"{path.relative_to(ROOT)}: missing {destination}")
            elif fragment and target.suffix == ".md" and fragment not in _anchors(target):
                failures.append(f"{path.relative_to(ROOT)}: missing anchor {destination}")
    assert not failures, "\n".join(failures)
