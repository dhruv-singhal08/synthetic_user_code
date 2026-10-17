"use strict";
// Synthetic User UI. All data is inserted with textContent (never innerHTML) - inputs are untrusted.

const STEPS = ["scenarios", "questions", "execute", "evaluate"];
const CHECK_LABELS = {
  relevance: "Relevance", goal_completeness: "Goal completeness", persona_fit: "Persona fit",
  capability_consistency: "Capability consistency", clarification_handling: "Clarification handling",
  safety_and_honesty: "Safety & honesty",
};
const MAIN_COLUMNS = [
  ["question_id", "Question"], ["scenario_title", "Scenario"], ["question", "Asked"],
  ["product_outcome", "Outcome"], ["final_verdict", "Verdict"], ["failed_checks", "Failed checks"],
  ["bug_ids", "Bug"],
];

const state = {
  config: null, samplePersona: null, personaSource: "sample", useIndex: false, uploads: {},
  runId: null, run: null, rows: [], scenarios: [], evaluations: [], filter: "all", search: "",
  allColumns: false, busy: false,
};

// --- helpers ----------------------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null && c !== false) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
const verdictBadge = (v) => el("span", { class: `v ${v || "none"}`, text: (v || "—").replaceAll("_", " ") });
function log(message, isError = false) {
  $("#log").prepend(el("div", { class: isError ? "err" : "", text: `${new Date().toLocaleTimeString()}  ${message}` }));
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : body.detail ? JSON.stringify(body.detail) : `HTTP ${response.status}`);
  return body;
}
const postJson = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const list = (items) => (items && items.length ? el("ul", {}, items.map((i) => el("li", { text: i }))) : el("span", { class: "muted", text: "—" }));
const split = (value) => (value || "").split(" | ").filter(Boolean);

// --- inputs -----------------------------------------------------------------------------------
function previewOf(kind, d) {
  const dl = el("dl");
  const add = (k, v) => dl.append(el("dt", { text: k }), el("dd", {}, v));
  if (kind === "persona") {
    add("Persona", `${d.identity.name} — ${d.identity.role}`);
    add("Expertise", `domain ${d.expertise.domain}, sql ${d.expertise.sql}`);
    add("Style", d.behavior.style);
    add("Typical tasks", list(d.typical_tasks));
  } else if (kind === "product") {
    add("Product", `${d.name} (${d.product_id} v${d.product_version})`);
    add("Type", d.product_type);
    add("Supported", list(d.supported));
    add("Not supported", list(d.not_supported_yet));
    add("Known gaps", list(d.known_gaps));
  } else {
    add("Index", `${d.index_id} v${d.index_version}`);
    add("Source", d.source_catalog_id);
    add("Tables", list(d.tables.map((t) => `${t.name} (${t.columns.length} columns)`)));
    add("Relationships", String(d.relationships.length));
  }
  return dl;
}

function panel(kind) { return $(`.input-panel[data-kind="${kind}"]`); }
function currentInput(kind) {
  if (kind === "persona" && state.personaSource === "sample") return state.samplePersona ? { preview: state.samplePersona, label: "Using the sample persona" } : null;
  if (kind === "reference_index" && !state.useIndex) return null;
  const upload = state.uploads[kind];
  return upload ? { preview: upload.preview, label: `Uploaded: ${upload.file.name}` } : null;
}

function renderInput(kind) {
  const root = panel(kind);
  const input = currentInput(kind);
  const box = $(".preview", root);
  box.replaceChildren();
  $(".source", root).textContent = input ? input.label : "";
  if (input) {
    box.append(previewOf(kind, input.preview.parsed), el("details", {}, el("summary", { text: kind === "reference_index" ? "View raw JSON" : "View raw YAML" }), el("pre", { text: input.preview.raw })));
  }
  const status = $("[data-status]", root);
  if (status) { status.className = `badge ${input ? "ok" : "req"}`; status.textContent = input ? "Ready" : "Required"; }
  updateReadiness();
}

async function onFile(kind, file) {
  if (!file) return;
  const root = panel(kind);
  $$(".error-box", root).forEach((n) => n.remove());
  const form = new FormData();
  form.append("file", file);
  try {
    const preview = await api(`/api/preview/${kind}`, { method: "POST", body: form });
    state.uploads[kind] = { file, preview };
  } catch (err) {
    delete state.uploads[kind];
    $(".preview", root).before(el("div", { class: "error-box", text: `Not accepted: ${err.message}` }));
  }
  renderInput(kind);
}

function wireDropzone(kind) {
  for (const zone of $$("[data-drop]", panel(kind))) {
    const input = $("input[type=file]", zone);
    input.addEventListener("change", () => { onFile(kind, input.files[0]); input.value = ""; });
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); onFile(kind, e.dataTransfer.files[0]); });
  }
}

function setPersonaSource(source) {
  state.personaSource = source;
  for (const b of $$(".segmented button", panel("persona"))) b.classList.toggle("on", b.dataset.source === source);
  $("[data-drop]", panel("persona")).classList.toggle("hidden", source !== "upload");
  renderInput("persona");
}

function setUseIndex(on) {
  state.useIndex = on;
  $("#use-index").checked = on;
  $("#index-mode-label").textContent = on ? "On — grounded" : "Off — ungrounded";
  $("#index-on").classList.toggle("hidden", !on);
  $("#index-off").classList.toggle("hidden", on);
  renderInput("reference_index");
}

function readiness() {
  const persona = Boolean(currentInput("persona"));
  const product = Boolean(currentInput("product"));
  const index = !state.useIndex || Boolean(currentInput("reference_index"));
  return { persona, product, index, ok: persona && product && index };
}

function updateReadiness() {
  const r = readiness();
  $("#chk-persona").className = r.persona ? "ok" : "";
  $("#chk-product").className = r.product ? "ok" : "";
  $("#chk-index").className = r.index ? "ok" : "";
  $("#chk-index").textContent = state.useIndex ? "Reference index" : "Reference index (off)";
  $("#start-hint").textContent = r.ok ? (state.runId ? "Starting again creates a new run." : "") : "Complete the required inputs to start.";
  $("#create-run").disabled = !r.ok || state.busy;
}

async function createRun() {
  const form = new FormData();
  if (state.personaSource === "upload" && state.uploads.persona) form.append("persona", state.uploads.persona.file);
  form.append("product", state.uploads.product.file);
  if (state.useIndex) form.append("reference_index", state.uploads.reference_index.file);
  form.append("use_reference_index", state.useIndex ? "true" : "false");
  form.append("persona_queries", $("#persona-queries").value);
  await withBusy("Starting run", async () => {
    const body = await api("/api/runs", { method: "POST", body: form });
    state.runId = body.state.run_id;
    history.replaceState(null, "", `?run=${encodeURIComponent(state.runId)}`);
    for (const node of $$(".step")) { node.classList.remove("done", "error", "running"); $(".result", node).textContent = ""; }
    applyRun(body);
    log(`Run ${state.runId} started (${state.useIndex ? "grounded" : "ungrounded"})`);
    $("#connection").scrollIntoView({ behavior: "smooth" });
  });
}

// --- product connection -----------------------------------------------------------------------
async function requestContract() {
  const url = $("#product-url").value.trim();
  await withBusy("Requesting contract", async () => {
    const body = await postJson(`/api/runs/${state.runId}/contract`, { url });
    state.run = body.state;
    renderContract(body.contract);
    updateUi();
    log(`Contract received from ${url}`);
  });
}

function renderContract(contract) {
  const box = $("#contract");
  box.replaceChildren();
  const badge = $("#conn-badge");
  if (!contract) {
    badge.className = "badge info"; badge.textContent = "Not connected";
    return;
  }
  badge.className = contract.mock ? "badge warn" : "badge ok";
  badge.textContent = contract.mock ? "Mock contract" : "Connected";
  if (contract.product_url && !$("#product-url").value) $("#product-url").value = contract.product_url;
  const kv = (k, v) => el("div", { class: "kv" }, el("b", { text: k }), el("span", { text: v }));
  box.append(
    el("div", { class: "contract" },
      kv("Product", contract.product_id), kv("Endpoint", `${contract.ask.method} ${contract.ask.path}`),
      kv("Outcomes", contract.outcomes.join(", ")), kv("Contract", `${contract.contract_version}${contract.mock ? " (mock)" : ""}`)),
    el("details", {}, el("summary", { text: "View contract JSON" }), el("pre", { text: JSON.stringify(contract, null, 2) })),
  );
  if (contract.mock) box.append(el("div", { class: "note", style: "margin-top:10px", text: "The real HTTP connection is not implemented yet: this mock contract routes questions to a mock product (an LLM stand-in that answers from the reference index)." }));
}

// --- pipeline ---------------------------------------------------------------------------------
function done() { return new Set(state.run ? state.run.steps_done : []); }

function updateUi() {
  const d = done();
  STEPS.forEach((step, i) => {
    const node = $(`.step[data-step="${step}"]`);
    const ready = Boolean(state.runId) && (i === 0 || d.has(STEPS[i - 1]));
    const btn = $(".step-btn", node);
    btn.disabled = !ready || state.busy;
    btn.textContent = d.has(step) ? "Re-run" : "Run";
    btn.classList.toggle("primary", ready && !d.has(step));
    if (!node.classList.contains("running") && !node.classList.contains("error")) {
      node.classList.toggle("done", d.has(step));
      $(".status", node).textContent = d.has(step) ? "Done" : ready ? "Ready" : "Pending";
    }
  });
  const completed = STEPS.filter((s) => d.has(s)).length;
  $("#progress-bar").style.width = `${(completed / STEPS.length) * 100}%`;
  $("#run-all").disabled = !state.runId || state.busy || completed === STEPS.length;
  $("#request-contract").disabled = !state.runId || state.busy;
  updateReadiness();

  const contract = state.run && state.run.contract;
  const adapter = state.config ? state.config.product_adapter : "configured";
  $("#execute-desc").textContent = contract
    ? "Questions go to the product behind the requested contract (mock product)."
    : `Questions go to the configured "${adapter}" product adapter. Answers are recorded, never executed.`;
  $("#adapter-note").textContent = contract ? "" : `Without a contract, questions are answered by the configured "${adapter}" product adapter.`;
  const grounded = state.run ? state.run.sources.reference_index !== "off" : state.useIndex;
  $("#scenarios-desc").textContent = grounded
    ? "Business goals, success criteria and a reference cheat sheet from the index."
    : "Business goals and success criteria (ungrounded: no cheat sheet).";

  const csv = $("#download-csv");
  csv.classList.toggle("disabled", !state.runId);
  csv.href = state.runId ? `/api/runs/${state.runId}/results.csv` : "#";
  $("#rail-run").replaceChildren(state.runId ? el("code", { text: state.runId }) : document.createTextNode("No run started"));
  $("#rail-mode").replaceChildren(...(state.run ? [el("span", { class: `badge ${grounded ? "blue" : "info"}`, text: grounded ? "Grounded" : "Ungrounded" })] : []));
  const nav = { inputs: Boolean(state.runId), connection: Boolean(contract), pipeline: completed === STEPS.length, results: state.rows.some((r) => r.evaluation_status) };
  for (const [k, v] of Object.entries(nav)) $(`#nav-${k}`).classList.toggle("done", v);
}

function describe(step, detail) {
  if (step === "scenarios") return `${detail.accepted} accepted, ${detail.rejected} rejected${detail.warnings && detail.warnings.length ? `, ${detail.warnings.length} index warning(s)` : ""}`;
  if (step === "questions") return `${detail.accepted} questions, ${detail.rejected} rejected`;
  if (step === "execute") return `${detail.executions} answers via ${detail.product}`;
  return `${detail.evaluations} evaluated, ${detail.bug_candidates} bug candidate(s)`;
}

async function runStep(step) {
  const node = $(`.step[data-step="${step}"]`);
  node.classList.remove("done", "error"); node.classList.add("running");
  $(".status", node).textContent = "Running…";
  const count = step === "scenarios" ? $("#scenario-count").value : step === "questions" ? $("#question-count").value : "";
  const started = performance.now();
  try {
    const body = await postJson(`/api/runs/${state.runId}/steps/${step}`, count ? { count: Number(count) } : {});
    node.classList.remove("running");
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    $(".result", node).textContent = `${describe(step, body.detail)} · ${secs}s`;
    for (const later of STEPS.slice(STEPS.indexOf(step) + 1)) {
      const laterNode = $(`.step[data-step="${later}"]`);
      laterNode.classList.remove("error");
      $(".result", laterNode).textContent = "";
    }
    applyRun(body);
    log(`${step}: ${describe(step, body.detail)} (${secs}s)`);
    for (const w of body.detail.warnings || []) log(`index warning: ${w}`, true);
    return true;
  } catch (err) {
    node.classList.remove("running"); node.classList.add("error");
    $(".status", node).textContent = "Error";
    $(".result", node).textContent = err.message;
    log(`${step} failed: ${err.message}`, true);
    return false;
  }
}

async function runAll() {
  await withBusy("Running pipeline", async () => {
    for (const step of STEPS) {
      if (done().has(step)) continue;
      if (!(await runStep(step))) break;
    }
  });
  $("#results").scrollIntoView({ behavior: "smooth" });
}

async function withBusy(label, fn) {
  state.busy = true; $("#busy").textContent = `${label}…`; updateUi();
  $$("#inputs .startbar .error-box").forEach((n) => n.remove());
  try { await fn(); } catch (err) { log(`${label} failed: ${err.message}`, true); alertBox(err.message); }
  finally { state.busy = false; $("#busy").textContent = ""; updateUi(); }
}
function alertBox(message) {
  $("#inputs .startbar").prepend(el("div", { class: "error-box", text: message }));
}

// --- results ----------------------------------------------------------------------------------
function applyRun(body) {
  state.run = body.state;
  state.rows = body.rows || [];
  state.scenarios = body.scenarios || [];
  state.evaluations = body.evaluations || [];
  renderContract(state.run.contract);
  renderResults(body);
  updateUi();
}

function filteredRows() {
  const q = state.search.trim().toLowerCase();
  return state.rows.filter((r) => {
    if (state.filter === "bugs" && !r.bug_ids) return false;
    if (["pass", "fail", "inconclusive"].includes(state.filter) && r.final_verdict !== state.filter) return false;
    return !q || Object.values(r).some((v) => v && v.toLowerCase().includes(q));
  });
}

function renderKpis() {
  const rows = state.rows.filter((r) => r.question_id);
  const by = (k, v) => rows.filter((r) => r[k] === v).length;
  const recorded = rows.filter((r) => r.evaluation_status === "recorded_only").length;
  const tiles = [
    ["", new Set(state.rows.map((r) => r.scenario_id)).size, "Scenarios"],
    ["", rows.length, "Questions"],
    ["", by("product_outcome", "answered"), "Answered"],
    ["pass", by("final_verdict", "pass"), "Pass"],
    ["fail", by("final_verdict", "fail"), "Fail"],
    ["inconclusive", by("final_verdict", "inconclusive"), "Inconclusive"],
    ["fail", rows.filter((r) => r.bug_ids).length, "Bug candidates"],
  ];
  $("#kpis").replaceChildren(...tiles.map(([cls, n, label]) => el("div", { class: `kpi ${cls}` }, el("b", { text: String(n) }), el("span", { text: label }))));
  const counts = { pass: by("final_verdict", "pass"), fail: by("final_verdict", "fail"), inconclusive: by("final_verdict", "inconclusive"), recorded };
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  $("#dist").replaceChildren(...Object.entries(counts).filter(([, n]) => n && total).map(([cls, n]) => el("div", { class: cls, style: `width:${(n / total) * 100}%`, title: `${cls}: ${n}` })));
}

function renderTable() {
  const columns = state.allColumns && state.rows.length ? Object.keys(state.rows[0]).map((k) => [k, k]) : MAIN_COLUMNS;
  const rows = filteredRows();
  $("#results-table thead").replaceChildren(el("tr", {}, columns.map(([, label]) => el("th", { text: label }))));
  const body = $("#results-table tbody");
  if (!rows.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: String(columns.length), class: "empty", text: state.rows.length ? "No rows match the filter." : "No results yet — start a run and run the pipeline." })));
    return;
  }
  body.replaceChildren(...rows.map((row) => {
    const tr = el("tr", {}, columns.map(([key]) => {
      const value = row[key] || "";
      if (key === "final_verdict" || key === "llm_verdict") return el("td", {}, value ? verdictBadge(value) : row.evaluation_status === "recorded_only" ? verdictBadge("recorded_only") : "");
      if (key === "product_outcome") return el("td", {}, el("span", { class: "outcome", text: value }));
      if (key === "question_id" || key === "bug_ids") return el("td", { class: "id", text: value });
      if (key === "question") return el("td", { class: "q", text: value });
      if (key === "failed_checks") return el("td", { text: split(value).map((c) => CHECK_LABELS[c] || c.replaceAll("_", " ")).join(", ") });
      return el("td", { text: value.length > 300 ? `${value.slice(0, 300)}…` : value });
    }));
    tr.addEventListener("click", () => openDrawer(row));
    return tr;
  }));
}

function renderResults(body) {
  renderKpis();
  renderTable();
  const rejected = $("#rejected");
  rejected.replaceChildren();
  for (const r of body.rejected_scenarios || []) rejected.append(el("div", { class: "rejected-item" }, el("b", { text: `Scenario: ${r.payload.title || "(no title)"}` }), el("div", { text: r.reasons.join("; ") })));
  for (const r of body.rejected_questions || []) rejected.append(el("div", { class: "rejected-item" }, el("b", { text: `Question (${r.scenario_id}): ${r.payload.utterance || "(none)"}` }), el("div", { text: r.reasons.join("; ") })));
  if (!rejected.children.length) rejected.append(el("p", { class: "muted", text: "Nothing rejected." }));
  $("#artifacts").replaceChildren(...(body.artifacts || []).map((name) => el("a", { class: "btn small", href: `/api/runs/${state.runId}/artifacts/${encodeURIComponent(name)}`, text: name })));
}

// --- drawer -----------------------------------------------------------------------------------
function section(title, ...content) { return el("div", { class: "dsec" }, el("h3", {}, ...[].concat(title)), ...content); }
function pairs(entries) {
  const dl = el("dl");
  for (const [k, v] of entries) if (v != null && v !== "" && !(Array.isArray(v) && !v.length)) dl.append(el("dt", { text: k }), el("dd", {}, Array.isArray(v) ? list(v) : v));
  return dl;
}

function openDrawer(row) {
  const scenario = state.scenarios.find((s) => s.id === row.scenario_id);
  const evaluation = state.evaluations.find((e) => e.question_id === row.question_id);
  $("#drawer-title").textContent = row.question_id ? `${row.question_id} · ${row.scenario_title}` : row.scenario_title;
  const parts = [];
  parts.push(section(["Scenario", el("span", { class: `badge ${row.mode === "grounded" ? "blue" : "info"}`, text: row.mode })], pairs([
    ["Goal", row.business_goal], ["Task / capability", `${row.task_type} · ${row.capability}`],
    ["Rationale", row.capability_rationale], ["Success criteria", scenario ? scenario.success_criteria : split(row.success_criteria)],
  ])));
  if (scenario && scenario.reference_cheat_sheet) {
    const cs = scenario.reference_cheat_sheet;
    parts.push(section("Reference cheat sheet", pairs([
      ["Tables", cs.tables], ["Columns", cs.columns], ["Joins", cs.relationships.map((r) => `${r.source} = ${r.target} (${r.verification})`)],
      ["Metrics", cs.metrics.map((m) => `${m.name}: ${m.definition}`)], ["Dimensions", cs.dimensions], ["Considerations", cs.semantic_considerations],
    ])));
  }
  if (row.question_id) {
    parts.push(section("Question", pairs([
      ["Asked", row.question], ["Style", `${row.question_style} · ambiguity ${row.ambiguity_level}`],
      ["Clarification", row.clarification_expected], ["Acceptable", split(row.acceptable_behaviors)],
      ["Unacceptable", split(row.unacceptable_behaviors)], ["Interpretations", split(row.interpretation_notes)],
    ])));
  }
  if (row.product_outcome) {
    const content = [pairs([["Outcome", row.product_outcome], ["Adapter", row.adapter], ["Answer", row.answer_text], ["Message", row.product_message], ["Latency", row.latency_ms ? `${row.latency_ms} ms` : ""]])];
    if (row.generated_sql) content.push(el("pre", { class: "sql", text: row.generated_sql }));
    parts.push(section("Product response", ...content));
  }
  if (evaluation) parts.push(...evaluationSections(evaluation));
  $("#drawer-body").replaceChildren(...parts);
  $("#drawer").classList.remove("hidden"); $("#scrim").classList.remove("hidden");
}

function checkTable(entries, rationale) {
  const table = el("table", { class: "check-table" });
  for (const [name, status] of entries) table.append(el("tr", {}, el("td", { text: CHECK_LABELS[name] || name.replaceAll("_", " ") }), el("td", {}, verdictBadge(status)), el("td", { text: rationale[name] || "" })));
  return table;
}

function evaluationSections(ev) {
  const out = [];
  out.push(section(["Evaluation", verdictBadge(ev.verdict || "recorded_only")], pairs([
    ["Status", ev.evaluation_status.replaceAll("_", " ")], ["LLM opinion", ev.llm_verdict ? verdictBadge(ev.llm_verdict) : ""],
    ["Plugins", ev.plugins_applied.join(", ")], ["Notes", ev.notes], ["Cheat sheet conflicts", ev.cheat_sheet_conflicts],
  ])));
  if (ev.checks) {
    const applicable = Object.entries(ev.checks).filter(([, s]) => s !== "not_applicable");
    out.push(section("Generic checks", checkTable(applicable, ev.check_rationale)));
  }
  if (ev.criteria_results.length) {
    const table = el("table", { class: "check-table" });
    for (const c of ev.criteria_results) table.append(el("tr", {}, el("td", {}, verdictBadge(c.status)), el("td", { text: c.criterion }), el("td", { class: "muted", text: c.rationale })));
    out.push(section("Goal completeness by success criterion", table));
  }
  if (Object.keys(ev.grounded_checks).length) {
    out.push(section(`Grounded checks (${ev.plugins_applied.join(", ")})`, checkTable(Object.entries(ev.grounded_checks), ev.check_rationale)));
  }
  const sql = ev.plugin_analysis && ev.plugin_analysis.sql;
  if (sql) {
    out.push(section("SQL static analysis", pairs([
      ["Parsed", sql.parse_ok ? "yes" : `no — ${sql.parse_errors.join("; ")}`], ["Tables", sql.tables],
      ["Joins", (sql.joins || []).map((j) => `${j.kind} ${j.table}${j.condition ? ` ON ${j.condition}` : ""}`)],
      ["Filters", sql.filters], ["Date expressions", sql.date_expressions], ["Aggregations", sql.aggregations],
      ["Group / order / limit", [sql.group_by.join(", "), sql.order_by.join(", "), sql.limit].filter(Boolean).join(" · ")],
      ["Unknown columns", sql.unknown_columns], ["Unverifiable tables", sql.unverifiable_tables], ["Join issues", sql.join_issues],
    ])));
  }
  if (ev.findings.length) {
    out.push(section(`Findings (${ev.findings.length})`, ...ev.findings.map((f) => el("div", { class: `finding sev-${f.severity}` },
      el("div", { class: "meta", text: `${f.finding_id} · ${f.category.replaceAll("_", " ")} · ${f.severity} · ${f.evidence_status} · ${f.source}${f.matches_known_gap ? " · known gap" : ""}` }),
      el("div", { text: f.description }),
      el("div", { class: "muted small", text: `Expected: ${f.expected_behavior} — Observed: ${f.observed_behavior}` }),
      f.evidence ? el("q", { text: f.evidence }) : null))));
  }
  for (const bug of ev.bug_candidates) {
    out.push(section(["Bug candidate", el("span", { class: "badge req", text: bug.status.replaceAll("_", " ") })], pairs([
      ["Id", bug.bug_id], ["Title", bug.title], ["Description", bug.description], ["Reproduce with", bug.reproduction_question],
      ["Product", bug.product_id], ["Findings", bug.finding_ids.join(", ")],
    ])));
  }
  return out;
}

function closeDrawer() { $("#drawer").classList.add("hidden"); $("#scrim").classList.add("hidden"); }

// --- boot -------------------------------------------------------------------------------------
async function boot() {
  try {
    state.config = await api("/api/config");
    $("#config-badge").textContent = `LLM ${state.config.llm_provider} · ${state.config.llm_model} · adapter ${state.config.product_adapter}`;
    $("#scenario-count").value = state.config.scenario_count;
    $("#question-count").value = state.config.question_count;
    const samples = await api("/api/samples");
    state.samplePersona = samples.persona;
  } catch (err) {
    log(`Could not load configuration: ${err.message}`, true);
  }
  for (const kind of ["persona", "product", "reference_index"]) wireDropzone(kind);
  for (const b of $$(".segmented button", panel("persona"))) b.addEventListener("click", () => setPersonaSource(b.dataset.source));
  $("#use-index").addEventListener("change", (e) => setUseIndex(e.target.checked));
  $("#create-run").addEventListener("click", createRun);
  $("#request-contract").addEventListener("click", requestContract);
  $("#run-all").addEventListener("click", runAll);
  for (const node of $$(".step")) $(".step-btn", node).addEventListener("click", () => withBusy(`Running ${node.dataset.step}`, () => runStep(node.dataset.step)));
  for (const b of $$("#filter button")) b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    for (const x of $$("#filter button")) x.classList.toggle("on", x === b);
    renderTable();
  });
  $("#search").addEventListener("input", (e) => { state.search = e.target.value; renderTable(); });
  $("#all-columns").addEventListener("change", (e) => { state.allColumns = e.target.checked; renderTable(); });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  setPersonaSource(state.samplePersona ? "sample" : "upload");
  setUseIndex(false);
  renderInput("product");

  const fromUrl = new URLSearchParams(location.search).get("run");
  if (fromUrl) {
    state.runId = fromUrl;
    try {
      applyRun(await api(`/api/runs/${encodeURIComponent(fromUrl)}`));
      $("#persona-queries").value = state.run.persona_queries.join(String.fromCharCode(10));
      log(`Reopened run ${fromUrl}`);
    } catch (err) {
      state.runId = null;
      log(`Could not reopen ${fromUrl}: ${err.message}`, true);
    }
  }
  renderKpis(); renderTable(); updateUi();
}

boot();
