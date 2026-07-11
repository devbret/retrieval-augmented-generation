"use strict";

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const docListEl = $("doc-list");
const uploadStatusEl = $("upload-status");

function escapeHtml(s) {
  return s.replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c],
  );
}

function baseName(path) {
  return path.split("/").pop();
}

marked.setOptions({ gfm: true, breaks: true });
marked.use(markedFootnote());

function renderMathHtml(tex, displayMode) {
  if (typeof katex === "undefined") return null;
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,
      output: "html",
    });
  } catch (err) {
    return null;
  }
}

const blockMathExtension = {
  name: "blockMath",
  level: "block",
  start(src) {
    const match = src.match(/(^|\n) {0,3}(?:\$\$|\\\[)/);
    return match ? match.index + match[1].length : undefined;
  },
  tokenizer(src) {
    const match = src.match(
      /^ {0,3}(?:\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\])[ \t]*(?:\n+|$)/,
    );
    if (!match) return undefined;
    return {
      type: "blockMath",
      raw: match[0],
      text: (match[1] ?? match[2]).trim(),
    };
  },
  renderer(token) {
    const html = renderMathHtml(token.text, true);
    return html ?? `<pre><code>${escapeHtml(token.text)}</code></pre>`;
  },
};

const inlineMathExtension = {
  name: "inlineMath",
  level: "inline",
  start(src) {
    const match = src.match(/\$|\\\(|\\\[/);
    return match ? match.index : undefined;
  },
  tokenizer(src) {
    let match = src.match(/^\\\(([\s\S]+?)\\\)/);
    if (match) {
      return {
        type: "inlineMath",
        raw: match[0],
        text: match[1].trim(),
        display: false,
      };
    }
    match = src.match(/^\\\[([\s\S]+?)\\\]/);
    if (match) {
      return {
        type: "inlineMath",
        raw: match[0],
        text: match[1].trim(),
        display: true,
      };
    }
    match = src.match(/^\$\$([\s\S]+?)\$\$/);
    if (match) {
      return {
        type: "inlineMath",
        raw: match[0],
        text: match[1].trim(),
        display: true,
      };
    }
    match = src.match(/^\$([^$\n]+?)\$(?!\d)/);
    if (match && !/^\s/.test(match[1]) && !/\s$/.test(match[1])) {
      return {
        type: "inlineMath",
        raw: match[0],
        text: match[1],
        display: false,
      };
    }
    return undefined;
  },
  renderer(token) {
    const html = renderMathHtml(token.text, token.display);
    return html ?? `<code>${escapeHtml(token.raw)}</code>`;
  },
};

marked.use({ extensions: [blockMathExtension, inlineMathExtension] });

function ensureSeparatorBreaks(markdown) {
  const lines = (markdown || "").split("\n");
  const out = [];
  let inFence = false;
  let fenceChar = "";
  const isDashRule = (line) => /^ {0,3}-{3,}\s*$/.test(line);
  for (const line of lines) {
    const fence = line.match(/^ {0,3}(`|~){3,}/);
    if (fence) {
      const ch = fence[1];
      if (!inFence) {
        inFence = true;
        fenceChar = ch;
      } else if (ch === fenceChar) {
        inFence = false;
        fenceChar = "";
      }
      out.push(line);
      continue;
    }
    if (!inFence && isDashRule(line)) {
      const prev = out.length ? out[out.length - 1] : "";
      if (prev.trim() !== "") out.push("");
    }
    out.push(line);
  }
  return out.join("\n");
}

marked.use({ hooks: { preprocess: ensureSeparatorBreaks } });

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName !== "A") return;
  const href = node.getAttribute("href") || "";
  if (/^https?:\/\//i.test(href)) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

function renderMarkdown(md = "") {
  const rawHtml = marked.parse(md || "");
  return DOMPurify.sanitize(rawHtml);
}

function enhanceCodeBlocks(root) {
  root.querySelectorAll("pre > code").forEach((code) => {
    const pre = code.parentElement;
    if (!pre || pre.dataset.enhanced) return;
    pre.dataset.enhanced = "1";

    const langMatch = (code.className || "").match(/language-([\w+-]+)/i);
    const lang = langMatch ? langMatch[1] : "";

    if (window.hljs) {
      try {
        hljs.highlightElement(code);
      } catch {}
    }

    const wrap = document.createElement("div");
    wrap.className = "code-block";
    pre.parentNode.insertBefore(wrap, pre);

    const head = document.createElement("div");
    head.className = "code-head";

    const label = document.createElement("span");
    label.className = "code-lang";
    label.textContent = lang || "code";

    const copy = document.createElement("button");
    copy.className = "code-copy";
    copy.type = "button";
    copy.textContent = "copy";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        copy.textContent = "copied";
        setTimeout(() => (copy.textContent = "copy"), 1500);
      } catch {}
    });

    head.appendChild(label);
    head.appendChild(copy);
    wrap.appendChild(head);
    wrap.appendChild(pre);
  });
}

function renderAnswerInto(container, markdownText) {
  const bubble = document.createElement("div");
  bubble.className = "bubble md";
  bubble.innerHTML = renderMarkdown(markdownText);
  enhanceCodeBlocks(bubble);
  container.appendChild(bubble);
  return bubble;
}

function applyCitationTooltips(root, sources) {
  if (!sources || !sources.length) return;
  const byId = new Map(
    sources.map((s) => [
      s.id,
      `${baseName(s.source)}\nchunk ${s.chunk} · score ${s.score.toFixed(2)}`,
    ]),
  );
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const candidates = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement.closest("code, pre, .katex, .cite")) continue;
    if (/\[\d+(?:\s*,\s*\d+)*\]/.test(node.nodeValue))
      candidates.push(node);
  }
  for (const node of candidates) {
    const text = node.nodeValue;
    const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
    const frag = document.createDocumentFragment();
    let last = 0;
    let match;
    while ((match = re.exec(text))) {
      const ids = match[1].split(",").map((x) => parseInt(x.trim(), 10));
      const tips = ids.map((id) => byId.get(id)).filter(Boolean);
      if (!tips.length) continue;
      frag.appendChild(
        document.createTextNode(text.slice(last, match.index)),
      );
      const span = document.createElement("span");
      span.className = "cite";
      span.textContent = match[0];
      span.setAttribute(
        "data-tip",
        tips.join("\n\n") + "\n\nclick to read the passage",
      );
      span.addEventListener("click", () => showPassages(ids, sources));
      frag.appendChild(span);
      last = match.index + match[0].length;
    }
    if (last === 0) continue;
    frag.appendChild(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }
}

const modalOverlay = $("modal-overlay");
const modalTitle = $("modal-title");
const modalBody = $("modal-body");

function openModal(title) {
  modalTitle.textContent = title;
  modalBody.innerHTML = "";
  modalOverlay.classList.add("open");
}
function closeModal() {
  modalOverlay.classList.remove("open");
}
$("modal-close").addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

function showPassages(ids, sources) {
  const byId = new Map(sources.map((s) => [s.id, s]));
  const picked = ids.map((id) => byId.get(id)).filter(Boolean);
  if (!picked.length) return;
  openModal(
    picked.length === 1 ? `Passage [${picked[0].id}]` : "Cited passages",
  );
  for (const s of picked) {
    const div = document.createElement("div");
    div.className = "passage";
    const meta = document.createElement("div");
    meta.className = "passage-meta";
    meta.textContent = `[${s.id}] ${baseName(s.source)} · chunk ${s.chunk} · score ${s.score.toFixed(2)}`;
    const body = document.createElement("div");
    body.className = "passage-text";
    body.textContent =
      s.text || "(passage text is not available for this message)";
    div.appendChild(meta);
    div.appendChild(body);
    modalBody.appendChild(div);
  }
}

async function openDocPreview(source) {
  openModal(baseName(source));
  modalBody.textContent = "Loading extracted text…";
  try {
    const r = await fetch(
      `/document-text?source=${encodeURIComponent(source)}`,
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const j = await r.json();
    modalBody.innerHTML = "";

    const head = document.createElement("div");
    head.className = "passage-meta";
    head.style.display = "flex";
    head.style.justifyContent = "space-between";
    head.style.alignItems = "center";
    const meta = document.createElement("span");
    meta.textContent = `${j.chunks} chunk${j.chunks === 1 ? "" : "s"} indexed${j.truncated ? " · preview truncated" : ""}`;
    const sumBtn = document.createElement("button");
    sumBtn.className = "modal-btn";
    sumBtn.type = "button";
    sumBtn.textContent = "Summarize document";
    sumBtn.addEventListener("click", () => summarizeDoc(source));
    head.appendChild(meta);
    head.appendChild(sumBtn);

    const body = document.createElement("div");
    body.className = "passage-text";
    body.textContent = j.text;
    modalBody.appendChild(head);
    modalBody.appendChild(body);
  } catch (e) {
    modalBody.textContent = `Could not load text: ${e.message}`;
  }
}

async function summarizeDoc(source) {
  if (sendBtn.disabled) return;
  closeModal();
  const label = `Summarize “${baseName(source)}”`;
  addMessage("user", `<div class="bubble">${escapeHtml(label)}</div>`);
  pushHistory({ role: "user", text: label });
  const pending = addMessage(
    "bot",
    `<div class="bubble typing">Summarizing…</div>`,
  );
  const typingEl = pending.querySelector(".typing");
  const t0 = Date.now();
  const timer = setInterval(() => {
    typingEl.textContent = `Summarizing… ${Math.round((Date.now() - t0) / 1000)}s`;
  }, 1000);
  sendBtn.disabled = true;
  try {
    const r = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    if (!r.ok || !r.body) {
      const bodyText = await r.text();
      let msg;
      try {
        msg = JSON.parse(bodyText).detail;
      } catch {}
      throw new Error(msg || bodyText.slice(0, 300) || r.statusText);
    }
    const { answer } = await streamNdjsonInto(pending, r, timer);
    if (!answer) throw new Error("The model returned an empty summary.");
    pending.innerHTML = "";
    renderAnswerInto(pending, answer);
    pushHistory({ role: "bot", text: answer });
  } catch (err) {
    pending.innerHTML = `<div class="bubble error">Summarize failed: ${escapeHtml(err.message)}</div>`;
    pushHistory({
      role: "bot",
      text: `Summarize failed: ${err.message}`,
      error: true,
    });
  } finally {
    clearInterval(timer);
    sendBtn.disabled = false;
    stickToBottom(true);
    questionEl.focus();
  }
}

async function checkHealth() {
  const dot = $("status-dot"),
    label = $("status-label");
  try {
    const r = await fetch("/health");
    const j = await r.json();
    if (j.llm && j.llm.reachable === false) {
      dot.className = "warn";
      label.textContent = `can't reach Ollama, is the AI PC up? · ${j.documents} docs`;
    } else if (j.llm && j.llm.model_available === false) {
      dot.className = "warn";
      label.textContent = `model "${j.model}" not found on Ollama, check .env`;
    } else {
      dot.className = "up";
      label.textContent = j.model
        ? `online · ${j.model} · ${j.documents} docs · ${j.chunks} chunks`
        : `online · collection: ${j.collection}`;
    }
    if (j.indexing && !ingestPollTimer) startIngestPolling();
  } catch {
    dot.className = "down";
    label.textContent = "server unreachable";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

async function refreshDocs() {
  let j;
  try {
    const r = await fetch("/documents");
    j = await r.json();
  } catch {
    docListEl.innerHTML =
      '<li style="color:var(--muted)">Could not load documents.</li>';
    return;
  }
  const checked = new Set(
    [...docListEl.querySelectorAll("input:checked")].map((i) => i.value),
  );
  docListEl.innerHTML = "";
  $("delete-all").hidden = !j.documents.length;
  if (!j.documents.length) {
    docListEl.innerHTML =
      '<li style="color:var(--muted)">No documents ingested yet.</li>';
    return;
  }
  for (const d of j.documents) {
    const li = document.createElement("li");
    li.innerHTML = `
<label title="${escapeHtml(d.source)}">
  <input type="checkbox" value="${escapeHtml(d.source)}"
         ${checked.has(d.source) ? "checked" : ""}>
  <span class="name">${escapeHtml(baseName(d.source))}</span>
</label>
<button class="doc-view" title="View extracted text">👁</button>
<button class="doc-del" title="Delete document">✕</button>`;
    li.querySelector(".doc-view").addEventListener("click", () =>
      openDocPreview(d.source),
    );
    li.querySelector(".doc-del").addEventListener("click", () =>
      deleteDoc(d.source),
    );
    docListEl.appendChild(li);
  }
}

async function deleteDoc(source) {
  if (!confirm(`Delete "${baseName(source)}" from the knowledge base?`))
    return;
  try {
    const r = await fetch(
      `/documents?source=${encodeURIComponent(source)}`,
      { method: "DELETE" },
    );
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  } catch (e) {
    alert(`Delete failed: ${e.message}`);
  }
  refreshDocs();
  checkHealth();
}
refreshDocs();

function setAllFilters(checked) {
  docListEl
    .querySelectorAll("input[type=checkbox]")
    .forEach((cb) => (cb.checked = checked));
}
$("filter-all").addEventListener("click", (e) => {
  e.preventDefault();
  setAllFilters(true);
});
$("filter-none").addEventListener("click", (e) => {
  e.preventDefault();
  setAllFilters(false);
});

$("delete-all").addEventListener("click", async () => {
  const count = docListEl.querySelectorAll("input[type=checkbox]").length;
  if (
    !confirm(
      `Delete all ${count} document${count === 1 ? "" : "s"} from the knowledge base AND remove their files from the docs folder? This cannot be undone.`,
    )
  )
    return;
  try {
    const r = await fetch("/documents/all", { method: "DELETE" });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const j = await r.json();
    showUploadResults(
      "",
      `Deleted ${j.documents_removed} document${j.documents_removed === 1 ? "" : "s"} (${j.chunks_removed} chunks).`,
    );
  } catch (e) {
    alert(`Delete all failed: ${e.message}`);
  }
  refreshDocs();
  checkHealth();
});

$("rescan").addEventListener("click", async (e) => {
  e.preventDefault();
  showUploadProgress("Scanning the docs folder…", null);
  try {
    const r = await fetch("/rescan", { method: "POST" });
    if (!r.ok) throw new Error(r.statusText);
    const j = await r.json();
    if (j.queued > 0) {
      startIngestPolling();
    } else {
      showUploadResults(
        "",
        `Everything is up to date (${j.unchanged} file${j.unchanged === 1 ? "" : "s"} already indexed).`,
      );
    }
  } catch (err) {
    showUploadResults(
      uploadResultRow(null, false, `Re-scan failed: ${err.message}`),
    );
  }
});

let ingestPollTimer = null;

function startIngestPolling(extraRows = "") {
  if (ingestPollTimer) return;
  showUploadProgress("Indexing…", null);
  let lastDocsRefresh = 0;
  ingestPollTimer = setInterval(async () => {
    let s;
    try {
      s = await (await fetch("/ingest-status")).json();
    } catch {
      return;
    }
    if (s.active) {
      const label = `Indexing ${Math.min(s.completed + 1, s.total)} of ${s.total}`;
      updateUploadProgress(label, s.total ? s.completed / s.total : null);
      if (Date.now() - lastDocsRefresh > 5000) {
        lastDocsRefresh = Date.now();
        refreshDocs();
      }
    } else {
      clearInterval(ingestPollTimer);
      ingestPollTimer = null;
      const results = s.results || [];
      const failed = results.filter((f) => !f.ok);
      const okCount = results.length - failed.length;
      const ordered = [...failed, ...results.filter((f) => f.ok)];
      const shown = ordered.slice(0, 12);
      let rows =
        shown
          .map((f) =>
            f.ok
              ? uploadResultRow(
                  baseName(f.source),
                  true,
                  `Ready to search · ${f.chunks} chunk${f.chunks === 1 ? "" : "s"}`,
                )
              : uploadResultRow(
                  baseName(f.source),
                  false,
                  friendlyUploadError(f.error),
                ),
          )
          .join("") + extraRows;
      if (ordered.length > shown.length) {
        rows += `<div class="up-note">…and ${ordered.length - shown.length} more</div>`;
      }
      const note = ingestCancelled
        ? `Cancelled: ${okCount} file${okCount === 1 ? "" : "s"} indexed before stopping. Use "re-scan folder" to resume.`
        : `Done: ${okCount} indexed${failed.length ? `, ${failed.length} failed` : ""}.`;
      ingestCancelled = false;
      showUploadResults(rows, note);
      scheduleUploadClear(30000);
      refreshDocs();
      checkHealth();
    }
  }, 1000);
}

(async () => {
  try {
    const s = await (await fetch("/ingest-status")).json();
    if (s.active) startIngestPolling();
  } catch {}
})();

const dropzone = $("dropzone");
const fileInput = $("file-input");

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
  fileInput.value = "";
});
for (const ev of ["dragenter", "dragover"]) {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
}
for (const ev of ["dragleave", "drop"]) {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  });
}
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

let uploadClearTimer = null;
let upLabelEl = null;
let upBarEl = null;

function resetUploadStatus() {
  clearTimeout(uploadClearTimer);
  uploadStatusEl.classList.remove("fading");
}

function scheduleUploadClear(delay = 10000) {
  clearTimeout(uploadClearTimer);
  uploadClearTimer = setTimeout(() => {
    uploadStatusEl.classList.add("fading");
    uploadClearTimer = setTimeout(() => {
      uploadStatusEl.innerHTML = "";
      uploadStatusEl.classList.remove("fading");
    }, 700);
  }, delay);
}

function showUploadProgress(label, fraction) {
  resetUploadStatus();
  uploadStatusEl.innerHTML =
    '<div class="up-progress"><div class="up-head"><div class="up-label"></div>' +
    '<button type="button" class="up-cancel">Cancel</button></div>' +
    '<div class="up-track"><div class="up-bar"></div></div></div>';
  upLabelEl = uploadStatusEl.querySelector(".up-label");
  upBarEl = uploadStatusEl.querySelector(".up-bar");
  uploadStatusEl
    .querySelector(".up-cancel")
    .addEventListener("click", cancelIngest);
  updateUploadProgress(label, fraction);
}

let currentUploadXhr = null;
let ingestCancelled = false;

async function cancelIngest() {
  ingestCancelled = true;
  if (currentUploadXhr) {
    try {
      currentUploadXhr.abort();
    } catch {}
  }
  try {
    await fetch("/ingest-cancel", { method: "POST" });
  } catch {}
  refreshDocs();
}

function updateUploadProgress(label, fraction) {
  if (!upLabelEl) return;
  upLabelEl.textContent = label;
  if (fraction == null) {
    upBarEl.classList.add("indeterminate");
    upBarEl.style.width = "100%";
  } else {
    upBarEl.classList.remove("indeterminate");
    upBarEl.style.width = `${Math.round(fraction * 100)}%`;
  }
}

function friendlyUploadError(error) {
  const m = /unsupported file type '(.+)'/.exec(error || "");
  if (m) return `The ${m[1]} file type isn't supported`;
  if (error === "no extractable text")
    return "No readable text found in this file";
  if (error === "invalid filename")
    return "The file name couldn't be used";
  return error || "Something went wrong";
}

function uploadResultRow(name, ok, desc) {
  return (
    `<div class="up-file ${ok ? "ok" : "err"}">` +
    `<span class="up-icon">${ok ? "✓" : "✕"}</span>` +
    `<span class="up-meta">` +
    (name ? `<span class="up-name">${escapeHtml(name)}</span>` : "") +
    `<span class="up-desc">${escapeHtml(desc)}</span>` +
    `</span></div>`
  );
}

function showUploadResults(rowsHtml, note) {
  resetUploadStatus();
  uploadStatusEl.innerHTML =
    (note ? `<div class="up-note">${escapeHtml(note)}</div>` : "") +
    rowsHtml;
  scheduleUploadClear();
}

function uploadWithProgress(fd, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    currentUploadXhr = xhr;
    const finish = (fn, arg) => {
      currentUploadXhr = null;
      fn(arg);
    };
    xhr.open("POST", "/upload");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          finish(resolve, JSON.parse(xhr.responseText));
        } catch {
          finish(reject, new Error("unexpected server response"));
        }
      } else {
        finish(reject, new Error(xhr.statusText || `HTTP ${xhr.status}`));
      }
    });
    xhr.addEventListener("abort", () =>
      finish(reject, new Error("cancelled")),
    );
    xhr.addEventListener("error", () =>
      finish(reject, new Error("network error")),
    );
    xhr.send(fd);
  });
}

async function uploadFiles(fileList) {
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  const noun =
    fileList.length === 1
      ? fileList[0].name || "1 file"
      : `${fileList.length} files`;
  showUploadProgress(`Uploading ${noun}…`, 0);
  try {
    const j = await uploadWithProgress(fd, (p) => {
      if (p < 1) {
        updateUploadProgress(
          `Uploading ${noun}… ${Math.round(p * 100)}%`,
          p,
        );
      } else {
        updateUploadProgress("Queued, starting the indexer…", null);
      }
    });
    const rejectedRows = (j.rejected || [])
      .map((f) =>
        uploadResultRow(f.filename, false, friendlyUploadError(f.error)),
      )
      .join("");
    if (j.queued && j.queued.length) {
      startIngestPolling(rejectedRows);
    } else {
      showUploadResults(
        rejectedRows ||
          uploadResultRow(null, false, "Nothing was uploaded"),
      );
    }
  } catch (e) {
    if (e.message === "cancelled") {
      ingestCancelled = false;
      showUploadResults("", "Upload cancelled.");
    } else {
      showUploadResults(
        uploadResultRow(null, false, `Upload failed: ${e.message}`),
      );
    }
  }
}

const form = $("ask-form");
const questionEl = $("question");
const sendBtn = $("send");

questionEl.addEventListener("input", () => {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 140) + "px";
});
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

function addMessage(role, html) {
  $("empty-hint")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = html;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function selectedSources() {
  const picked = [...docListEl.querySelectorAll("input:checked")].map(
    (i) => i.value,
  );
  return picked.length ? picked : null;
}

function stickToBottom(force) {
  const nearBottom =
    messagesEl.scrollHeight -
      messagesEl.scrollTop -
      messagesEl.clientHeight <
    120;
  if (force || nearBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function streamNdjsonInto(pending, resp, timer) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let answer = "";
  let srcs = [];
  let bubble = null;
  let lastRender = 0;

  const handleEvent = (ev) => {
    if (ev.type === "sources") {
      srcs = ev.sources || [];
    } else if (ev.type === "token") {
      answer += ev.text;
      if (!bubble) {
        clearInterval(timer);
        pending.innerHTML = "";
        bubble = document.createElement("div");
        bubble.className = "bubble md";
        pending.appendChild(bubble);
      }
      const now = Date.now();
      if (now - lastRender > 120) {
        lastRender = now;
        bubble.innerHTML = renderMarkdown(answer);
        stickToBottom();
      }
    } else if (ev.type === "error") {
      throw new Error(ev.message || "stream error");
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) handleEvent(JSON.parse(line));
    }
  }
  return { answer: answer.trim(), srcs };
}

function buildLlmHistory() {
  return history
    .filter((m) => !m.error && m.text)
    .slice(-8)
    .map((m) => ({
      role: m.role === "user" ? "user" : "assistant",
      content: m.text,
    }));
}

function capSources(sources) {
  return (sources || []).map((s) => ({
    ...s,
    text: typeof s.text === "string" ? s.text.slice(0, 1500) : s.text,
  }));
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q || sendBtn.disabled) return;

  const llmHistory = buildLlmHistory();

  questionEl.value = "";
  questionEl.style.height = "auto";
  addMessage("user", `<div class="bubble">${escapeHtml(q)}</div>`);
  pushHistory({ role: "user", text: q });
  const pending = addMessage(
    "bot",
    `<div class="bubble typing">Thinking…</div>`,
  );
  const typingEl = pending.querySelector(".typing");
  const t0 = Date.now();
  const timer = setInterval(() => {
    typingEl.textContent = `Thinking… ${Math.round((Date.now() - t0) / 1000)}s`;
  }, 1000);
  sendBtn.disabled = true;

  const body = {
    question: q,
    min_score: parseFloat($("opt-minscore").value) || 0,
    answer_style: $("opt-style").value,
    sources: selectedSources(),
    history: llmHistory,
  };

  let answer = "";
  let srcs = [];
  try {
    const r = await fetch("/ask-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok || !r.body) {
      const bodyText = await r.text();
      let msg;
      try {
        msg = JSON.parse(bodyText).detail;
      } catch {}
      throw new Error(msg || bodyText.slice(0, 300) || r.statusText);
    }

    const streamed = await streamNdjsonInto(pending, r, timer);
    answer = streamed.answer;
    srcs = streamed.srcs;
    if (!answer) throw new Error("The model returned an empty answer.");
    pending.innerHTML = "";
    const finalBubble = renderAnswerInto(pending, answer);
    applyCitationTooltips(finalBubble, srcs);
    pushHistory({ role: "bot", text: answer, sources: capSources(srcs) });
  } catch (err) {
    pending.innerHTML = `<div class="bubble error">Request failed: ${escapeHtml(err.message)}</div>`;
    pushHistory({
      role: "bot",
      text: `Request failed: ${err.message}`,
      error: true,
    });
  } finally {
    clearInterval(timer);
    sendBtn.disabled = false;
    stickToBottom(true);
    questionEl.focus();
  }
});

const sidebarEl = document.querySelector("aside");
const resizerEl = $("sidebar-resizer");
const SIDEBAR_KEY = "rag-sidebar-width";

function clampSidebarWidth(w) {
  const max = Math.min(560, window.innerWidth * 0.6);
  return Math.max(220, Math.min(max, w));
}

try {
  const saved = parseInt(localStorage.getItem(SIDEBAR_KEY), 10);
  if (saved) sidebarEl.style.width = `${clampSidebarWidth(saved)}px`;
} catch {}

let sidebarDragging = false;
resizerEl.addEventListener("pointerdown", (e) => {
  if (window.matchMedia("(max-width: 720px)").matches) return;
  sidebarDragging = true;
  resizerEl.classList.add("dragging");
  resizerEl.setPointerCapture(e.pointerId);
  document.body.style.userSelect = "none";
  document.body.style.cursor = "col-resize";
});
resizerEl.addEventListener("pointermove", (e) => {
  if (!sidebarDragging) return;
  sidebarEl.style.width = `${clampSidebarWidth(e.clientX)}px`;
});
resizerEl.addEventListener("pointerup", () => {
  if (!sidebarDragging) return;
  sidebarDragging = false;
  resizerEl.classList.remove("dragging");
  document.body.style.userSelect = "";
  document.body.style.cursor = "";
  try {
    localStorage.setItem(
      SIDEBAR_KEY,
      parseInt(sidebarEl.style.width, 10),
    );
  } catch {}
});
resizerEl.addEventListener("dblclick", () => {
  sidebarEl.style.width = "";
  try {
    localStorage.removeItem(SIDEBAR_KEY);
  } catch {}
});

const OPTS_KEY = "rag-options";
let history = [];
try {
  localStorage.removeItem("rag-chat-history");
} catch {}

function pushHistory(entry) {
  history.push(entry);
  history = history.slice(-50);
}

$("clear-chat").addEventListener("click", () => {
  if (history.length && !confirm("Clear the chat history?")) return;
  history = [];
  messagesEl.innerHTML =
    '<div class="empty-hint" id="empty-hint">' +
    "Upload documents on the left, then ask questions about them here. " +
    "Answers are grounded in your files and cite their sources.</div>";
});

function downloadFile(filename, mime, content) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportStamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
}

function sourceLine(s) {
  return `[${s.id}] ${baseName(s.source)} (chunk ${s.chunk}, score ${(s.score ?? 0).toFixed(2)})`;
}

function exportCsv() {
  const csvEscape = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const rows = [["#", "role", "message", "sources"]];
  history.forEach((m, i) => {
    rows.push([
      i + 1,
      m.role === "user" ? "user" : "assistant",
      m.text || "",
      (m.sources || []).map(sourceLine).join("; "),
    ]);
  });
  const csv = rows.map((r) => r.map(csvEscape).join(",")).join("\r\n");
  downloadFile(
    `rag-chat-${exportStamp()}.csv`,
    "text/csv;charset=utf-8",
    "﻿" + csv,
  );
}

function exportJson() {
  const data = {
    exported_at: new Date().toISOString(),
    messages: history,
  };
  downloadFile(
    `rag-chat-${exportStamp()}.json`,
    "application/json",
    JSON.stringify(data, null, 2),
  );
}

function exportHtml() {
  const parts = [];
  parts.push(
    `<!doctype html><html><head><meta charset="utf-8"><title>RAG chat export</title><style>
  body{font:15px/1.55 system-ui,sans-serif;max-width:780px;margin:32px auto;padding:0 16px;color:#1c1b1a;background:#faf9f7}
  h1{font-size:20px}
  .meta{color:#6f6b66;font-size:13px}
  .m{margin:14px 0;padding:12px 16px;border-radius:12px}
  .user{background:#2563eb;color:#fff;margin-left:15%}
  .bot{background:#fff;border:1px solid #e2e0dd;margin-right:15%}
  .role{font-size:11px;text-transform:uppercase;letter-spacing:.05em;opacity:.65;margin-bottom:4px}
  .sources{margin-top:8px;font-size:12px;color:#6f6b66;border-top:1px solid #eee;padding-top:6px}
  pre{background:#f2f0ed;padding:10px;border-radius:8px;overflow-x:auto}
  code{font-family:ui-monospace,monospace;font-size:.9em}
  table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}
  blockquote{border-left:3px solid #ddd;margin-left:0;padding-left:10px;color:#6f6b66}
  .error{color:#b91c1c}
  </style></head><body>`,
  );
  parts.push(
    `<h1>Personal RAG chat export</h1><p class="meta">Exported ${escapeHtml(new Date().toLocaleString())}</p>`,
  );
  for (const m of history) {
    if (m.role === "user") {
      parts.push(
        `<div class="m user"><div class="role">You</div>${escapeHtml(m.text || "").replaceAll("\n", "<br>")}</div>`,
      );
    } else {
      const body = m.error
        ? `<span class="error">${escapeHtml(m.text || "")}</span>`
        : renderMarkdown(m.text || "");
      let srcs = "";
      if (m.sources && m.sources.length) {
        srcs =
          `<div class="sources">` +
          m.sources.map((s) => escapeHtml(sourceLine(s))).join("<br>") +
          `</div>`;
      }
      parts.push(
        `<div class="m bot"><div class="role">Assistant</div>${body}${srcs}</div>`,
      );
    }
  }
  parts.push("</body></html>");
  downloadFile(
    `rag-chat-${exportStamp()}.html`,
    "text/html",
    parts.join("\n"),
  );
}

const exportMenu = $("export-menu");
$("export-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  exportMenu.classList.toggle("open");
});
document.addEventListener("click", () =>
  exportMenu.classList.remove("open"),
);
exportMenu.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-fmt]");
  if (!btn) return;
  exportMenu.classList.remove("open");
  if (!history.length) {
    alert("There's no chat to export yet.");
    return;
  }
  if (btn.dataset.fmt === "csv") exportCsv();
  else if (btn.dataset.fmt === "json") exportJson();
  else exportHtml();
});

try {
  const o = JSON.parse(localStorage.getItem(OPTS_KEY)) || {};
  if (o.minscore != null) $("opt-minscore").value = o.minscore;
  if (o.style) $("opt-style").value = o.style;
} catch {}
for (const id of ["opt-minscore", "opt-style"]) {
  $(id).addEventListener("change", () => {
    try {
      localStorage.setItem(
        OPTS_KEY,
        JSON.stringify({
          minscore: $("opt-minscore").value,
          style: $("opt-style").value,
        }),
      );
    } catch {}
  });
}

if (location.hash === "#mdtest") {
  const demo = [
    "## Markdown test",
    "",
    "Some **bold**, *italic*, `inline code`, and a [link](https://example.com).",
    "",
    "- bullet one [1]",
    "- bullet two",
    "",
    "| Col A | Col B |",
    "| --- | --- |",
    "| 1 | 2 |",
    "",
    "> A blockquote with a citation [2]",
    "",
    "```python",
    "def hello(name):",
    '    return f"hi {name}"',
    "```",
    "",
    "Inline math $E = mc^2$ and display math:",
    "",
    "$$\\int_0^1 x^2\\,dx = \\tfrac{1}{3}$$",
    "",
    "A footnote reference[^1].",
    "",
    "[^1]: The footnote body.",
  ].join("\n");
  const holder = addMessage("bot", "");
  const bubble = renderAnswerInto(holder, demo);
  applyCitationTooltips(bubble, [
    {
      id: 1,
      source: "./docs/example-document.pdf",
      chunk: 3,
      score: 0.71,
      text: "This is the demo passage that citation [1] points at. In real answers this shows the exact chunk of your document that the model used.",
    },
    {
      id: 2,
      source: "./docs/another-file.md",
      chunk: 0,
      score: 0.55,
      text: "Demo passage for citation [2].",
    },
  ]);
  showUploadResults(
    uploadResultRow(
      "quarterly-report-with-a-long-name.pdf",
      true,
      "Ready to search · 12 chunks",
    ) +
      uploadResultRow(
        "holiday-photo.png",
        false,
        "The .png file type isn't supported",
      ),
  );
}
