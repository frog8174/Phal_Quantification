# Skill: document-markdown-cache

## Purpose

This skill defines a repeatable workflow for converting documents to Markdown using the `markitdown` MCP server, caching the converted Markdown, and reusing cached Markdown when the source document has not changed.

Use this skill whenever the agent needs to read, summarize, inspect, extract, compare, or analyze documents such as PDF, DOCX, PPTX, XLSX, HTML, or other MarkItDown-supported formats.

---

## Available Tool

The agent should use the MCP server named:

```text
markitdown
```

Expected tool:

```text
convert_to_markdown(uri)
```

Expected input:

```json
{
  "uri": "file:///workdir/input/example.pdf"
}
```

Expected output:

```text
Markdown content returned to the agent
```

The MarkItDown MCP server converts the document and returns Markdown. It does not automatically save the Markdown to disk. The agent or wrapper workflow is responsible for caching the returned Markdown.

---

## Workspace Layout

Use the following workspace convention:

```text
/workdir
  /input
  /markdown
  /metadata
  /tmp
```

Directory roles:

```text
/workdir/input
  Stores original source files.

/workdir/markdown
  Stores converted Markdown cache files.

/workdir/metadata
  Stores JSON metadata for each conversion.

/workdir/tmp
  Stores temporary files if needed.
```

If the agent is running outside Kubernetes, it may access the same shared storage through another path, such as a mounted NAS path. In that case, map local paths to MCP-visible paths carefully.

Example path mapping:

```text
Agent local path:
Z:\Workspace\Aaron\markitdown-mcp-files\input\fileA.pdf

MCP container path:
file:///workdir/input/fileA.pdf
```

---

## Cache Key Rule

Prefer using SHA-256 of the source file as the cache key.

Cache filename format:

```text
<safe-basename>.<sha256-prefix>.md
<safe-basename>.<sha256-prefix>.json
```

Example:

```text
/workdir/markdown/fileA.a1b2c3d4e5f6.md
/workdir/metadata/fileA.a1b2c3d4e5f6.json
```

Use at least the first 12 characters of the SHA-256 hash. Use the full SHA-256 in metadata.

If the source is a remote URL and direct hashing is not practical, use a stable cache key derived from:

```text
source_uri + retrieval timestamp or ETag/Last-Modified if available
```

For local files, source file hash is preferred over modification time.

---

## Metadata Format

For every cached Markdown file, create a metadata JSON file.

Example:

```json
{
  "source_uri": "file:///workdir/input/fileA.pdf",
  "source_display_name": "fileA.pdf",
  "source_sha256": "full_sha256_hash_here",
  "markdown_path": "/workdir/markdown/fileA.a1b2c3d4e5f6.md",
  "converted_at": "2026-06-03T16:00:00+08:00",
  "converter": "markitdown-mcp",
  "mcp_server": "markitdown",
  "tool": "convert_to_markdown",
  "status": "success"
}
```

If conversion fails, store failure metadata when useful:

```json
{
  "source_uri": "file:///workdir/input/fileA.pdf",
  "converted_at": "2026-06-03T16:00:00+08:00",
  "converter": "markitdown-mcp",
  "mcp_server": "markitdown",
  "tool": "convert_to_markdown",
  "status": "failed",
  "error": "error message here"
}
```

---

## Workflow

### Step 1: Identify the source document

Determine whether the user is asking about a document, file, URL, PDF, Word document, PowerPoint, spreadsheet, or HTML page.

If the document is inside the MCP container, use:

```text
file:///workdir/input/<filename>
```

If the document is remote, use:

```text
https://...
```

---

### Step 2: Determine whether cache exists

Before calling MarkItDown MCP, check for an existing Markdown cache.

For local files:

1. Locate the source file.
2. Compute SHA-256.
3. Look for matching metadata under `/workdir/metadata`.
4. If matching metadata exists and its Markdown file exists, reuse the Markdown.

A cache is valid only if:

```text
source_sha256 matches current file hash
metadata status is success
markdown_path exists
markdown file is non-empty
```

---

### Step 3: Reuse cache when valid

If valid cache exists:

1. Do not call MarkItDown MCP.
2. Read the cached Markdown.
3. Use only relevant sections for the final answer.
4. Mention that cached Markdown was reused if the user asks about performance, conversion, or repeated reads.

---

### Step 4: Convert only when needed

If no valid cache exists:

1. Call:

```text
markitdown.convert_to_markdown(uri)
```

2. Receive Markdown content.
3. Save Markdown to:

```text
/workdir/markdown/<safe-basename>.<sha256-prefix>.md
```

4. Save metadata to:

```text
/workdir/metadata/<safe-basename>.<sha256-prefix>.json
```

5. Use the saved Markdown as the source for analysis.

---

### Step 5: Use Markdown efficiently

Do not load the entire Markdown into context if it is long.

Recommended reading strategy:

1. Inspect headings.
2. Search for relevant keywords.
3. Extract relevant sections.
4. Summarize or answer based on relevant sections.
5. Preserve important page/section references if available.

---

## Decision Rules

Use MarkItDown MCP when:

```text
The user asks to read or summarize a PDF, DOCX, PPTX, XLSX, HTML, or other document.
The user provides a file path under the shared document workspace.
The agent needs structured Markdown from a rich document.
The document has not yet been converted or the cache is invalid.
```

Reuse cache when:

```text
A Markdown cache exists.
The source file hash matches metadata.
The metadata status is success.
The Markdown file exists and is readable.
```

Do not use MarkItDown MCP when:

```text
The user already pasted the relevant text.
The file is already clean Markdown or plain text.
The source file is inaccessible to the MCP server.
The document is extremely small and already available in context.
```

---

## Error Handling

If conversion fails:

1. Check whether the URI is valid.
2. Check whether the file exists under `/workdir/input`.
3. Check whether the MarkItDown MCP server is reachable.
4. Check whether the format is supported.
5. Report the failure clearly.
6. Do not invent document contents.

Common issues:

```text
file not found:
  The file path is not visible inside the MarkItDown MCP container.

permission denied:
  The mounted volume permissions are insufficient.

connection refused:
  MCP endpoint or Ingress is not reachable.

empty output:
  The document may be image-only, scanned, encrypted, or unsupported.
```

---

## Example: Local file conversion

Source file:

```text
/workdir/input/report.pdf
```

MCP URI:

```text
file:///workdir/input/report.pdf
```

Tool call:

```json
{
  "uri": "file:///workdir/input/report.pdf"
}
```

Cache output:

```text
/workdir/markdown/report.<sha256-prefix>.md
/workdir/metadata/report.<sha256-prefix>.json
```

---

## Example: Remote URL conversion

Source URI:

```text
https://example.com/report.pdf
```

Tool call:

```json
{
  "uri": "https://example.com/report.pdf"
}
```

Cache output:

```text
/workdir/markdown/report-url.<hash-prefix>.md
/workdir/metadata/report-url.<hash-prefix>.json
```

---

## Final Answer Behavior

When answering the user based on converted Markdown:

1. State whether the answer is based on converted Markdown or cached Markdown when relevant.
2. Avoid exposing unnecessary internal metadata.
3. Mention conversion failures or incomplete extraction honestly.
4. Do not claim to have read parts of the document that were not extracted or available.
