"""Render docs/runbooks/runpod-end-to-end.md into a self-contained HTML file.

Self-contained = no external CSS/JS references. The HTML embeds a small
<style> block in <head> with a readable typography stack and code-block
styling. Opens cleanly in any browser offline.

Uses python-markdown (already a dev dependency via mkdocs in the wider
project) with the `extra` extension for tables and fenced code blocks.
Avoids pandoc dependency on local Windows env.

Run: python scripts/_render_runbook_html.py
"""

from __future__ import annotations

from pathlib import Path

import markdown

SRC = Path("docs/runbooks/runpod-end-to-end.md")
DST = Path("docs/runbooks/runpod-end-to-end.html")

CSS = """
/* Force a single high-contrast light theme regardless of OS dark mode.
   Previous version honoured prefers-color-scheme, which produced
   unreadable text on dark-mode browsers. Simpler and predictable: pin
   light. */
:root {
    color-scheme: light only;
}
html, body {
    background: #ffffff !important;
    color: #0a0a0a !important;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    max-width: 880px;
    margin: 2.5rem auto;
    padding: 0 1.5rem 4rem;
    line-height: 1.6;
}
h1, h2, h3, h4 {
    line-height: 1.25;
    margin-top: 2.2rem;
    margin-bottom: 0.8rem;
    color: #000000 !important;
}
h1 { font-size: 2rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }
h2 { font-size: 1.5rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }
h3 { font-size: 1.2rem; }
h4 { font-size: 1rem; }
p { margin: 0.6rem 0; color: #0a0a0a !important; }
a { color: #0550ae !important; text-decoration: underline; }
a:hover { text-decoration: underline; }
code {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 0.92em;
    background: #f0f1f3 !important;
    padding: 0.15em 0.35em;
    border-radius: 4px;
    color: #8a1c1c !important;
}
pre {
    background: #f6f8fa !important;
    padding: 0.9rem 1rem;
    border-radius: 6px;
    overflow-x: auto;
    border: 1px solid #d0d7de;
    line-height: 1.45;
    color: #0a0a0a !important;
}
pre code {
    background: transparent !important;
    padding: 0;
    color: #0a0a0a !important;
    font-size: 0.88rem;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
    font-size: 0.94em;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 0.5rem 0.8rem;
    text-align: left;
    vertical-align: top;
    color: #0a0a0a !important;
    background: #ffffff;
}
th {
    background: #f6f8fa;
    font-weight: 600;
}
blockquote {
    margin: 1rem 0;
    padding: 0.3rem 1rem;
    border-left: 4px solid #d0d7de;
    color: #444444 !important;
    background: #fafbfc;
}
hr {
    border: none;
    height: 1px;
    background: #d0d7de;
    margin: 2rem 0;
}
ul, ol {
    padding-left: 1.5rem;
}
li {
    margin: 0.25rem 0;
    color: #0a0a0a !important;
}
.nav {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin: 1rem 0 2rem;
    font-size: 0.92em;
    color: #0a0a0a !important;
}
.nav strong { display: block; margin-bottom: 0.3rem; color: #000000 !important; }
"""


def main() -> None:
    src_md = SRC.read_text(encoding="utf-8")

    body_html = markdown.markdown(
        src_md,
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>finpost RunPod end-to-end runbook</title>
<style>{CSS}</style>
</head>
<body>
<div class="nav">
<strong>finpost runbook</strong>
This document is self-contained &mdash; open it in any browser offline. Markdown source: <code>docs/runbooks/runpod-end-to-end.md</code>.
</div>
{body_html}
</body>
</html>
"""

    DST.write_text(html, encoding="utf-8")
    print(f"Rendered {SRC} -> {DST}")
    print(f"HTML size: {DST.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
