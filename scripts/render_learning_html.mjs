import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { basename } from "node:path";

const PAGES = [
  {
    source: "STUDY.md",
    output: "STUDY.html",
    title: "finpost study guide: FinChain-first RLVR",
  },
  {
    source: "STUDY.md",
    output: "docs/finchain-rlvr-professor-study.html",
    title: "finpost study guide: FinChain-first RLVR",
  },
  {
    source: "docs/distributed-training-and-platforms.md",
    output: "docs/distributed-training-and-platforms.html",
    title: "Distributed training and GPU platform guide",
  },
  {
    source: "docs/runbooks/finchain-rlvr-study-flow.md",
    output: "docs/runbooks/finchain-rlvr-study-flow.html",
    title: "FinChain RLVR study flow",
  },
];

const CSS = `
:root {
  color-scheme: light only;
  --bg: #fbfbfa;
  --fg: #18202f;
  --muted: #5c6678;
  --line: #d9dee7;
  --panel: #ffffff;
  --soft: #eef3fb;
  --note: #fff8e6;
  --ok: #236b3d;
  --warn: #9a6200;
  --bad: #9f3030;
  --code: #f2f4f8;
  --max: 1040px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.62 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: var(--max); margin: 0 auto; padding: 42px 24px 84px; }
header { border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 26px; }
h1 { margin: 0 0 8px; font-size: 2.15rem; line-height: 1.12; letter-spacing: 0; }
h2 { margin: 42px 0 14px; padding-bottom: 7px; border-bottom: 1px solid var(--line); font-size: 1.45rem; }
h3 { margin: 28px 0 10px; font-size: 1.12rem; }
h4 { margin: 22px 0 8px; font-size: 1rem; color: #2a4d80; }
p { margin: 0 0 14px; }
ul, ol { margin: 0 0 16px 24px; padding: 0; }
li { margin: 6px 0; }
a { color: #244f86; text-decoration: none; border-bottom: 1px dotted currentColor; }
a:hover { border-bottom-style: solid; }
code {
  background: var(--code);
  border-radius: 4px;
  padding: 0.12rem 0.28rem;
  font-family: "Cascadia Mono", Consolas, ui-monospace, monospace;
  font-size: 0.9em;
}
pre {
  background: var(--code);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
}
pre code { background: transparent; padding: 0; }
blockquote {
  margin: 18px 0;
  padding: 14px 18px;
  border-left: 4px solid #2a4d80;
  background: var(--soft);
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0 24px;
  background: var(--panel);
  border: 1px solid var(--line);
}
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { background: var(--soft); font-weight: 700; }
tr:last-child td { border-bottom: 0; }
.toc {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 20px 0 34px;
}
.toc strong { display: block; margin-bottom: 8px; }
.toc ol { margin-bottom: 0; columns: 2; }
@media (max-width: 740px) { .toc ol { columns: 1; } main { padding: 30px 18px 70px; } }
.meta { color: var(--muted); margin: 0; }
.active-direction-banner {
  border: 1px solid #d8c58a;
  background: var(--note);
  color: #4a3510;
  border-radius: 8px;
  padding: 12px 14px;
  margin: 0 0 22px;
}
`;

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function inline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return out;
}

function renderTable(lines, start) {
  const rows = [];
  let i = start;
  while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
    rows.push(lines[i].trim());
    i += 1;
  }
  const parsed = rows.map((row) => row.slice(1, -1).split("|").map((c) => c.trim()));
  const header = parsed[0] || [];
  const body = parsed.slice(2);
  const html = [
    "<table>",
    "<thead><tr>" + header.map((cell) => `<th>${inline(cell)}</th>`).join("") + "</tr></thead>",
    "<tbody>",
    ...body.map((row) => "<tr>" + row.map((cell) => `<td>${inline(cell)}</td>`).join("") + "</tr>"),
    "</tbody></table>",
  ].join("\n");
  return { html, next: i };
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  const headings = [];
  let i = 0;
  let listType = null;
  let inCode = false;
  let codeLines = [];

  function closeList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {
      if (!inCode) {
        closeList();
        inCode = true;
        codeLines = [];
      } else {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        inCode = false;
      }
      i += 1;
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      i += 1;
      continue;
    }

    if (!line.trim()) {
      closeList();
      i += 1;
      continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[-: |]+\|\s*$/.test(lines[i + 1])) {
      closeList();
      const rendered = renderTable(lines, i);
      html.push(rendered.html);
      i = rendered.next;
      continue;
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      closeList();
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = slugify(text);
      if (level <= 3) headings.push({ level, text, id });
      html.push(`<h${level} id="${id}">${inline(text)}</h${level}>`);
      i += 1;
      continue;
    }

    if (line.startsWith("> ")) {
      closeList();
      html.push(`<blockquote><p>${inline(line.slice(2))}</p></blockquote>`);
      i += 1;
      continue;
    }

    const bullet = /^-\s+(.+)$/.exec(line);
    if (bullet) {
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }
      html.push(`<li>${inline(bullet[1])}</li>`);
      i += 1;
      continue;
    }

    const numbered = /^\d+\.\s+(.+)$/.exec(line);
    if (numbered) {
      if (listType !== "ol") {
        closeList();
        html.push("<ol>");
        listType = "ol";
      }
      html.push(`<li>${inline(numbered[1])}</li>`);
      i += 1;
      continue;
    }

    closeList();
    const paragraph = [line.trim()];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,4})\s+/.test(lines[i]) &&
      !/^-\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !lines[i].startsWith("> ") &&
      !lines[i].startsWith("```") &&
      !/^\s*\|.*\|\s*$/.test(lines[i])
    ) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    html.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }
  closeList();

  const tocItems = headings
    .filter((h) => h.level === 2)
    .map((h) => `<li><a href="#${h.id}">${inline(h.text)}</a></li>`)
    .join("\n");
  const toc = tocItems ? `<nav class="toc"><strong>Contents</strong><ol>${tocItems}</ol></nav>` : "";
  return { body: toc + "\n" + html.join("\n"), headings };
}

function renderPage({ source, output, title }) {
  const markdown = readFileSync(source, "utf8");
  const { body } = renderMarkdown(markdown);
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>${escapeHtml(title)}</title>
<style>${CSS}</style>
</head>
<body>
<main>
<header>
<h1>${escapeHtml(title)}</h1>
<p class="meta">Rendered from <code>${escapeHtml(source)}</code>. Open offline in any browser.</p>
</header>
${body}
</main>
</body>
</html>
`;
  writeFileSync(output, html, "utf8");
  console.log(`Rendered ${source} -> ${output}`);
}

function injectBanner(path) {
  if (!existsSync(path)) return;
  let content = readFileSync(path, "utf8");
  if (content.includes("active-direction-banner")) return;
  const banner = `
  <div class="active-direction-banner" style="border:1px solid #d8c58a;background:#fff8e6;color:#4a3510;border-radius:8px;padding:12px 14px;margin:0 0 22px;">
    <strong>Active direction note:</strong>
    This page is preserved as a phase artifact. The current roadmap is FinChain-first RLVR.
    Start with <code>STUDY.html</code>, <code>docs/finchain-rlvr-professor-study.html</code>,
    and <code>docs/distributed-training-and-platforms.html</code> before using this older page operationally.
  </div>
`;
  if (!content.includes(".active-direction-banner")) {
    content = content.replace("</style>", `${CSS.match(/\\.active-direction-banner[\\s\\S]*?\\}/)?.[0] ?? ""}\n</style>`);
  }
  content = content.replace(/<main[^>]*>/, (m) => `${m}\n${banner}`);
  writeFileSync(path, content, "utf8");
  console.log(`Bannered ${path}`);
}

for (const page of PAGES) renderPage(page);

for (const path of [
  "docs/phase1-sft-study.html",
  "docs/dpo-study.html",
  "RUNPOD_RUNBOOK.html",
  "docs/runbooks/runpod-end-to-end.html",
]) {
  injectBanner(path);
}

console.log("learning HTML render complete");
