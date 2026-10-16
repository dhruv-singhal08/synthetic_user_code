"use strict";
// Synthetic User UI. All data is inserted with textContent (never innerHTML) - inputs are untrusted.

const STEPS = ["index", "scenarios", "questions", "execute", "evaluate"];
const KINDS = ["persona", "product", "catalog"];
const GROUPS = {
  scenario: ["scenario_id", "scenario_title", "business_goal", "task_type", "capability"],
  cheat: ["capability_rationale", "cheat_tables", "cheat_columns", "cheat_joins", "cheat_metrics",
    "cheat_dimensions", "cheat_operations", "cheat_filters", "cheat_semantic_considerations"],
  question: ["question_id", "question", "question_style", "ambiguity_level", "clarification_expected",
    "acceptable_behaviors", "unacceptable_behaviors", "interpretation_notes", "schema_leak_warnings"],
  execution: ["execution_id", "product_outcome", "generated_sql", "product_message", "latency_ms"],
  evaluation: ["evaluation_id", "final_verdict", "llm_verdict", "evaluation_status", "failed_checks",
    "inconclusive_checks", "findings", "bug_ids", "bug_titles", "evaluation_notes"],
};

const state = { config: null, samples: {}, uploads: {}, runId: null, run: null, rows: [], busy: false };

// --- tiny DOM helpers ---------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
function log(message, isError = false) {
  const box = $("#log");
  box.prepend(el("div", { class: isError ? "err" : "", text: `${new Date().toLocaleTimeString()}  ${message}` }));
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${response.status}`);
  return body;
}
const postJson = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

// --- inputs / previews --------------------------------------------------------------------
function summary(kind, d) {
  const dl = el("dl");
  const add = (k, v) => dl.append(el("dt", { text: k }), el("dd", {}, v));
  const list = (items) => items && items.length ? el("ul", {}, items.map((i) => el("li", { text: i }))) : "-";
  if (kind === "persona") {
    add("id", `${d.id} (v${d.version})`);
    add("role", d.identity.role);
    add("expertise", `domain ${d.expertise.domain} / sql ${d.expertise.sql}`);
    add("style", d.behavior.style);
    add("motivation", d.primary_motivation);
    add("typical tasks", list(d.typical_tasks));
    add("example queries", list(d.example_utterances));
  } else if (kind === "product") {
    add("product", `${d.name} (${d.product_id} v${d.product_version})`);
    add("supported", list(d.supported));
    add("not supported yet", list(d.not_supported_yet));
    add("known gaps", list(d.known_gaps));
  } else {
    add("catalog", `${d.catalog_id} v${d.catalog_version}`);
    add("tables", list(d.tables.map((t) => `${t.name} (${t.columns.length} cols)${t.description ? "" : " - no description"}`)));
    add("relationships", list(d.relationships.map((r) => `${r.source_table}.${r.source_column} -> ${r.target_table}.${r.target_column} [${r.verification}]`)));
  }
  return dl;
}

function renderInput(kind) {
  const col = $(`.input-col[data-kind="${kind}"]`);
  const upload = state.uploads[kind];
  const data = upload ? upload.preview : state.samples[kind];
  $(".source", col).textContent = upload ? `Using uploaded file: ${upload.file.name}` : "Using sample";
  const preview = $(".preview", col);
  preview.replaceChildren();
  if (!data) return;
  preview.append(summary(kind, data.parsed), el("details", {}, el("summary", { text: "Raw YAML" }), el("pre", { text: data.raw })));
}

async function onFile(kind, input) {
  const file = input.files[0];
  input.value = "";
  if (!file) return;
  const col = $(`.input-col[data-kind="${kind}"]`);
  $$(".error-box", col).forEach((n) => n.remove());
  const form = new FormData();
  form.append("file", file);
  try {
    const preview = await api(`/api/preview/${kind}`, { method: "POST", body: form });
    state.uploads[kind] = { file, preview };
    renderInput(kind);
  } catch (err) {
    $(".preview", col).before(el("div", { class: "error-box", text: `Invalid ${kind}: ${err.message}` }));
  }
}

async function createRun() {
  const form = new FormData();
  for (const kind of KINDS) if (state.uploads[kind]) form.append(kind, state.uploads[kind].file);
  form.append("persona_queries", $("#persona-queries").value);
  await withBusy("Creating run", async () => {
    const body = await api("/api/runs", { method: "POST", body: form });
    state.runId = body.state.run_id;
    history.replaceState(null, "", `?run=${encodeURIComponent(state.runId)}`);
    applyRun(body);
    log(`Run ${state.runId} created (persona: ${body.state.sources.persona}, product: ${body.state.sources.product}, catalog: ${body.state.sources.catalog}, sample queries: ${body.state.persona_queries.length})`);
  });
}

// --- product contract ---------------------------------------------------------------------
async function requestContract() {
  const url = $("#product-url").value.trim();
  await withBusy("Requesting contract", async () => {
    const body = await postJson(`/api/runs/${state.runId}/contract`, { url });
    state.run = body.state;
    renderContract(body.contract);
    updateSteps();
    log(`Contract received from ${url} (MOCK)`);
  });
}
function renderContract(contract) {
  const box = $("#contract");
  box.replaceChildren();
  if (!contract) return;
  const dl = el("dl");
  const add = (k, v) => dl.append(el("dt", { text: k }), el("dd", { text: v }));
  add("status", contract.mock ? "MOCK contract - answers come from the mock product" : "connected");
  add("product", contract.product_id);
  add("url", contract.product_url);
  add("ask endpoint", `${contract.ask.method} ${contract.ask.path}`);
  add("sql dialect", contract.sql_dialect);
  box.append(dl, el("details", {}, el("summary", { text: "Contract JSON" }), el("pre", { text: JSON.stringify(contract, null, 2) })));
}

// --- pipeline -----------------------------------------------------------------------------
function stepIndex(step) { return STEPS.indexOf(step); }
function updateSteps() {
  const done = new Set(state.run ? state.run.steps_done : []);
  const next = STEPS.find((s) => !done.has(s));
  for (const node of $$(".step")) {
    const step = node.dataset.step;
    const pill = $(".pill", node);
    const prevDone = stepIndex(step) === 0 || done.has(STEPS[stepIndex(step) - 1]);
    $(".step-btn", node).disabled = !state.runId || state.busy || !prevDone;
    $(".step-btn", node).textContent = done.has(step) ? "Re-run" : "Run";
    if (!pill.classList.contains("running") && !pill.classList.contains("error")) {
      pill.className = `pill ${done.has(step) ? "done" : ""}`;
      pill.textContent = done.has(step) ? "done" : "pending";
    }
  }
  $("#run-all").disabled = !state.runId || state.busy || !next;
  $("#request-contract").disabled = !state.runId || state.busy;
  $("#create-run").disabled = state.busy;
  $("#execute-target").textContent = state.run && state.run.contract ? "mock product (contract)" : `${state.config ? state.config.product_client : ""} product`;
  const csv = $("#download-csv");
  csv.classList.toggle("disabled", !state.runId);
  csv.href = state.runId ? `/api/runs/${state.runId}/results.csv` : "#";
}

async function runStep(step) {
  const node = $(`.step[data-step="${step}"]`);
  const pill = $(".pill", node);
  pill.className = "pill running"; pill.textContent = "running…";
  const count = step === "scenarios" ? $("#scenario-count").value : step === "questions" ? $("#question-count").value : "";
  const started = performance.now();
  try {
    const body = await postJson(`/api/runs/${state.runId}/steps/${step}`, count ? { count: Number(count) } : {});
    pill.className = "pill done"; pill.textContent = "done";
    applyRun(body);
    log(`${step}: ${JSON.stringify(body.detail).replace(/[{}"]/g, "").replace(/,/g, ", ")} (${((performance.now() - started) / 1000).toFixed(1)}s)`);
    return true;
  } catch (err) {
    pill.className = "pill error"; pill.textContent = "error";
    log(`${step} failed: ${err.message}`, true);
    return false;
  }
}

async function runAll() {
  await withBusy("Running pipeline", async () => {
    const done = new Set(state.run.steps_done);
    for (const step of STEPS) {
      if (done.has(step)) continue;
      if (!(await runStep(step))) break;
    }
  });
}

async function withBusy(label, fn) {
  state.busy = true; $("#busy").textContent = `${label}…`; updateSteps();
  try { await fn(); } catch (err) { log(`${label} failed: ${err.message}`, true); }
  finally { state.busy = false; $("#busy").textContent = ""; updateSteps(); }
}

// --- results ------------------------------------------------------------------------------
function applyRun(body) {
  state.run = body.state;
  state.rows = body.rows || [];
  $("#run-label").textContent = `Run: ${state.runId}`;
  for (const node of $$(".step .pill")) if (!node.classList.contains("running")) node.classList.remove("error");
  renderContract(state.run.contract);
  renderResults(body);
  updateSteps();
}

function visibleColumns() {
  const cols = [];
  for (const box of $$(".toolbar input[data-group]")) if (box.checked) cols.push(...GROUPS[box.dataset.group]);
  return cols;
}

function cell(column, value) {
  const td = el("td");
  if (column === "final_verdict" || column === "llm_verdict") {
    if (value) td.append(el("span", { class: `v ${value}`, text: value }));
  } else if (column === "evaluation_status" && value === "recorded_only") {
    td.append(el("span", { class: "v recorded_only", text: value }));
  } else {
    if (column === "generated_sql") td.className = "sql";
    td.textContent = value.length > 400 ? `${value.slice(0, 400)}…` : value;
  }
  return td;
}

function renderResults(body) {
  const cols = visibleColumns();
  const thead = $("#results thead"); const tbody = $("#results tbody");
  thead.replaceChildren(el("tr", {}, cols.map((c) => el("th", { text: c }))));
  tbody.replaceChildren(...state.rows.map((row, i) => {
    const tr = el("tr", {}, cols.map((c) => cell(c, row[c] || "")));
    tr.addEventListener("click", () => openDrawer(i));
    return tr;
  }));
  if (!state.rows.length) tbody.append(el("tr", {}, el("td", { colspan: String(cols.length || 1), class: "muted", text: "No results yet - run the pipeline steps." })));

  const rows = state.rows;
  const count = (k, v) => rows.filter((r) => r[k] === v).length;
  const chips = [
    `scenarios: ${new Set(rows.map((r) => r.scenario_id)).size}`,
    `questions: ${rows.filter((r) => r.question_id).length}`,
    `sql: ${count("product_outcome", "sql_generated")}`, `clarification: ${count("product_outcome", "clarification")}`,
    `denied/error/timeout: ${rows.filter((r) => ["denied", "error", "timeout"].includes(r.product_outcome)).length}`,
    `pass: ${count("final_verdict", "pass")}`, `fail: ${count("final_verdict", "fail")}`,
    `inconclusive: ${count("final_verdict", "inconclusive")}`, `bugs: ${rows.filter((r) => r.bug_ids).length}`,
  ];
  $("#summary").replaceChildren(...chips.map((c) => el("span", { class: "chip", text: c })));

  const rejected = $("#rejected");
  rejected.replaceChildren();
  for (const r of body.rejected_scenarios || []) rejected.append(el("div", { class: "rejected-item" }, el("b", { text: `Scenario: ${r.payload.title || "(no title)"}` }), el("div", { text: r.reasons.join("; ") })));
  for (const r of body.rejected_questions || []) rejected.append(el("div", { class: "rejected-item" }, el("b", { text: `Question (${r.scenario_id}): ${r.payload.utterance || "(none)"}` }), el("div", { text: r.reasons.join("; ") })));
  if (!rejected.children.length) rejected.append(el("p", { class: "muted", text: "Nothing rejected." }));

  $("#artifacts").replaceChildren(...(body.artifacts || []).map((name) =>
    el("a", { class: "btn small", href: `/api/runs/${state.runId}/artifacts/${encodeURIComponent(name)}`, text: name })));
}

function openDrawer(index) {
  const row = state.rows[index];
  $("#drawer-title").textContent = row.question_id ? `${row.question_id} - ${row.scenario_id}` : row.scenario_id;
  const dl = el("dl");
  for (const [k, v] of Object.entries(row)) if (v) dl.append(el("dt", { text: k }), el("dd", { text: v.replaceAll(" | ", "\n") }));
  $("#drawer-body").replaceChildren(dl);
  $("#drawer").classList.remove("hidden");
}

// --- boot ---------------------------------------------------------------------------------
async function boot() {
  try {
    state.config = await api("/api/config");
    $("#config-badge").textContent = `LLM: ${state.config.llm_provider} / ${state.config.llm_model} · product: ${state.config.product_client} · dialect: ${state.config.sql_dialect}`;
    $("#scenario-count").value = state.config.scenario_count;
    $("#question-count").value = state.config.question_count;
    state.samples = await api("/api/samples");
  } catch (err) {
    log(`Could not load config/samples: ${err.message}`, true);
  }
  for (const kind of KINDS) {
    const col = $(`.input-col[data-kind="${kind}"]`);
    $("input[type=file]", col).addEventListener("change", (e) => onFile(kind, e.target));
    $(".reset", col).addEventListener("click", () => { delete state.uploads[kind]; $$(".error-box", col).forEach((n) => n.remove()); renderInput(kind); });
    renderInput(kind);
  }
  $("#create-run").addEventListener("click", createRun);
  $("#request-contract").addEventListener("click", requestContract);
  $("#run-all").addEventListener("click", runAll);
  for (const node of $$(".step")) $(".step-btn", node).addEventListener("click", () => withBusy(`Running ${node.dataset.step}`, () => runStep(node.dataset.step)));
  for (const box of $$(".toolbar input[data-group]")) box.addEventListener("change", () => (state.runId ? refreshRun() : renderResults({})));
  $("#drawer-close").addEventListener("click", () => $("#drawer").classList.add("hidden"));
  const fromUrl = new URLSearchParams(location.search).get("run");
  if (fromUrl) {
    state.runId = fromUrl;
    await refreshRun();
    if (state.run) $("#persona-queries").value = state.run.persona_queries.join(String.fromCharCode(10));
    log(`Reopened run ${fromUrl}`);
  }
  updateSteps();
}

async function refreshRun() {
  if (!state.runId) return;
  try { applyRun(await api(`/api/runs/${state.runId}`)); } catch (err) { log(err.message, true); }
}

boot();
