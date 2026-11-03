"use strict";
// Synthetic User web UI (D199, D201, D211): workflow pages + test report over the JSON API.
// Metrics are chosen per run in the Evaluation tab (no evaluation profiles); the metric catalog,
// labels and statuses come from the server (/api/config, the run's frozen metric_config).
// All data is inserted with textContent / DOM nodes (never innerHTML): inputs are untrusted.

const STEPS = ["scenarios", "questions", "execute", "evaluate"];
const VIEWS = ["setup", "evaluation", "connect", "run", "report"];
const NON_SCORED_LABELS = {
  not_applicable: "n/a", cannot_generate: "no expectation", evidence_unavailable: "no evidence",
  inconclusive: "?", evaluation_error: "error", unavailable: "unavailable", not_enabled: "off",
};
const SEVERITY_RANK = { low: 1, medium: 2, high: 3, critical: 4 };
const OUTCOME_LABELS = { answered: "answered", clarification: "asked to clarify", denied: "declined", error: "error", timeout: "timed out" };
const SOURCE_LABELS = { persona: "Persona", product: "Product description", reference: "Reference data", reference_guide: "Usage instructions", custom_metrics: "Custom metric workbook" };

const state = {
  config: null, samples: {}, personaSource: "sample", useReference: false, instructionsSource: "none", uploads: {},
  questionComplexity: "", selectedMetrics: new Set(), contextSource: "response_field", removeSavedWorkbook: false,
  runId: null, run: null, rows: [], scenarios: [], packs: [], evaluations: [], executions: [], kpis: null, issues: null,
  questions: [], metricConfig: null, metricIds: [],
  filter: "all", filterIds: null, filterLabel: "", search: "", allColumns: false, breakdownBy: "by_task_type", busy: false,
  contractSource: "upload", contractFile: null, recent: [],
};

// --- helpers ----------------------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null && c !== false) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "ic");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}
const human = (v) => String(v || "").replaceAll("_", " ");
const pill = (text, cls) => el("span", { class: `pill pill-${cls}`, text: human(text) });
const verdictPill = (v) => pill(v === "recorded_only" ? "not judged" : v || "—", v || "none");
const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const ms = (v) => (v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const when = (iso) => { try { return new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); } catch (err) { return iso; } };
const fileName = (source) => (source || "").replace(/^uploaded: /, "");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
function metricDef(m) {
  const frozen = state.metricConfig && state.metricConfig.definitions.find((d) => d.metric_id === m);
  if (frozen) return { label: frozen.display_name, help: frozen.description, category: frozen.category };
  const builtin = state.config && state.config.metric_catalog.find((d) => d.id === m);
  if (builtin) return { label: builtin.display_name, help: builtin.description, category: builtin.category };
  const kpi = state.kpis && state.kpis.metrics.find((x) => x.metric === m);
  return { label: (kpi && kpi.display_name) || human(m), help: "", category: kpi ? kpi.category : "" };
}
const metricLabel = (m) => metricDef(m).label;
const list = (items) => (items && items.length ? el("ul", {}, items.map((i) => el("li", {}, i))) : el("span", { class: "muted", text: "—" }));
// CSV cells list several items as "• item" lines (items may span lines, e.g. SQL).
const split = (value) => {
  const text = (value || "").trim();
  if (!text) return [];
  if (!text.startsWith("• ")) return [text];
  return text.slice(2).split(/\n• /).map((item) => item.trim()).filter(Boolean);
};
// The structured success_criteria cell (JSON, D240): "general" (the question's criteria), one key
// per metric with a question-specific expectation and "cheat_sheet" ({items, rules}).
const criteriaOf = (row) => {
  try { return JSON.parse(row.success_criteria || "{}") || {}; } catch (err) { return {}; }
};
const metricCriteria = (criteria) => Object.entries(criteria)
  .filter(([key]) => key !== "general" && key !== "cheat_sheet")
  .map(([id, value]) => {
    // Not generated: {status, reason}; generated: the metric's own fields.
    const notGenerated = value && Object.keys(value).sort().join() === "reason,status";
    return `${metricLabel(id)}: ${notGenerated ? `${human(value.status)}${value.reason ? ` (${value.reason})` : ""}` : JSON.stringify(value)}`;
  });
const cheatSheetItems = (criteria) => {
  const sheet = criteria.cheat_sheet;
  return sheet ? [...(sheet.items || []), ...(sheet.rules || [])] : [];
};
function kvList(entries) {
  const dl = el("dl", { class: "kv-list" });
  for (const [k, v] of entries) {
    if (v == null || v === "" || (Array.isArray(v) && !v.length)) continue;
    dl.append(el("dt", { text: k }), el("dd", {}, Array.isArray(v) ? list(v) : v));
  }
  return dl;
}
function callout(kind, title, body, action) {
  const name = kind === "info" ? "info" : "warn";
  return el("div", { class: `callout callout-${kind}` }, icon(name),
    el("div", { class: "callout-body" }, title ? el("b", { text: title }) : null, body ? el("div", {}, body) : null), action || null);
}

function log(message, isError = false) {
  $("#log").prepend(el("div", { class: isError ? "err" : "", text: `${new Date().toLocaleTimeString()}  ${message}` }));
}
function toast(message, kind = "info") {
  const node = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, icon(kind === "error" ? "warn" : "check"), el("div", { text: message }));
  $("#toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 9000 : 4000);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : body.detail ? JSON.stringify(body.detail) : `HTTP ${response.status}`);
  return body;
}
const postJson = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
const putJson = (path, data) => api(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

async function withBusy(label, fn) {
  state.busy = true;
  $("#busy-line").classList.add("on");
  updateUi();
  try { await fn(); } catch (err) { log(`${label} failed: ${err.message}`, true); toast(`${label} failed: ${err.message}`, "error"); }
  finally { state.busy = false; $("#busy-line").classList.remove("on"); renderEvaluation(); updateUi(); }
}

// --- views ------------------------------------------------------------------------------------
function showView(name, { scroll = true } = {}) {
  const view = VIEWS.includes(name) ? name : "setup";
  for (const node of $$(".view")) node.classList.toggle("active", node.dataset.view === view);
  for (const node of $$("[data-nav]")) {
    node.classList.toggle("active", node.dataset.nav === view);
    if (node.dataset.nav === view) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
  }
  if (location.hash !== `#${view}`) history.replaceState(null, "", `${location.pathname}${location.search}#${view}`);
  if (scroll) window.scrollTo({ top: 0 });
}

// --- inputs -----------------------------------------------------------------------------------
function previewOf(kind, d) {
  if (kind === "persona") {
    const expertise = Object.entries(d.expertise || {}).filter(([k]) => k !== "notes").map(([k, v]) => `${human(k)}: ${v}`);
    const prefs = d.question_preferences || {};
    const questionStyle = [`complexity ${prefs.complexity || "mixed"}`, `ambiguity ${prefs.ambiguity || "config default"}`, prefs.follow_up_questions ? "asks follow-ups" : "no follow-ups"].join(" · ");
    return kvList([["Persona", `${d.identity.name} — ${d.identity.role}`], ["Expertise", expertise], ["Style", d.behavior.style], ["Traits", d.behavior.traits],
      ["Typical tasks", d.typical_tasks], ["Motivation", d.primary_motivation], ["Decision inputs", d.decision_inputs], ["Questions", questionStyle]]);
  }
  if (kind === "product") {
    return kvList([["Product", `${d.name} (${d.product_id} v${d.product_version})`], ["Problem", d.business_problem],
      ["Data sources", (d.data_sources || []).map((s) => `${s.name}: ${s.description}`)], ["Supported", d.supported],
      ["Not supported", d.not_supported], ["Known limitations", d.known_limitations], ["Typical questions", d.typical_user_questions],
      ["Expected outputs", d.expected_outputs]]);
  }
  if (kind === "reference") {
    const counts = d.kind === "data_catalog"
      ? `${plural(d.tables, "table")}, ${plural(d.relationships, "relationship")}, ${plural(d.rules, "rule")}`
      : plural(d.documents, "document");
    const agent = d.agent_tools == null ? "none (Tool Correctness cannot generate expectations)"
      : `${plural(d.agent_tools, "tool")}, ${plural(d.agent_data_sources, "data source")}, ${plural(d.agent_workflow_stages, "workflow stage")}`;
    return kvList([["Format", human(d.kind)], ["Name", d.name], ["Contents", counts], ["Tool definitions", agent], ["Reference id", d.reference_id], ["Size", `${d.chars.toLocaleString()} characters`]]);
  }
  if (kind === "custom_metrics") {
    return kvList([["Workbook", d.source_name], ["Metrics", d.metrics.map((m) => `${m.provenance}: ${m.text}`)],
      ["Row problems", d.errors.map((e) => `${e.provenance}: ${e.reason}`)]]);
  }
  return kvList([["Instructions", d.name], ["Size", `${d.chars.toLocaleString()} characters`]]);
}

function panel(kind) { return $(`.input[data-kind="${kind}"]`); }
function sample(kind) { return state.samples[kind] || null; }

function currentInput(kind) {
  if (kind === "persona") {
    if (state.personaSource === "sample") return sample("persona") ? { preview: sample("persona"), label: "Default persona", isDefault: true } : null;
  }
  if ((kind === "reference" || kind === "reference_guide") && !state.useReference) return null;
  if (kind === "reference_guide" && state.instructionsSource !== "upload") return null;
  const upload = state.uploads[kind];
  if (upload) return { preview: upload.preview, label: upload.file.name, isDefault: false };
  if ((kind === "product" || kind === "reference") && sample(kind)) {
    return { preview: sample(kind), label: kind === "product" ? "Default product description" : "Default reference", isDefault: true };
  }
  return null;
}

function renderInput(kind) {
  const root = panel(kind);
  const input = currentInput(kind);
  const box = $(`[data-preview-for="${kind}"]`) || $(".preview", root);
  const source = $(".source", root);
  source.className = input ? "source ok" : "source";
  source.replaceChildren(...(input ? [icon("check"), input.label] : []));
  if (input) {
    const children = [previewOf(kind, input.preview.parsed)];
    if (input.preview.raw) children.push(el("details", { class: "raw" }, el("summary", { text: kind === "persona" || kind === "product" ? "View YAML" : "View file" }), el("pre", { text: input.preview.raw })));
    box.replaceChildren(...children);
  } else if (box.dataset.empty) {
    box.replaceChildren(el("div", { class: "preview-empty", text: box.dataset.empty }));
  } else {
    box.replaceChildren();
  }
  const num = $(`[data-num-for="${kind}"]`);
  if (num) num.classList.toggle("ok", Boolean(input));
  if (kind === "product") {
    const status = $("[data-status-for=product]");
    status.className = `pill ${input ? "pill-ok" : "pill-req"}`;
    status.textContent = input ? (input.isDefault ? "Default" : "Uploaded") : "Required";
    $("#product-clear").classList.toggle("hidden", !(state.uploads.product && sample("product")));
    renderComplexityOptions();
  }
  if (kind === "reference") $("#reference-clear").classList.toggle("hidden", !(state.uploads.reference && sample("reference")));
  if (kind === "custom_metrics" && !input && savedWorkbook()) source.replaceChildren(icon("check"), `Saved with the run: ${savedWorkbook()}`);
  if (kind === "persona") renderComplexityOptions();
  renderEvaluation();
  updateUi();
}

async function onFile(kind, file) {
  if (!file) return;
  const root = panel(kind);
  $$(".callout-error", root).forEach((n) => n.remove());
  const form = new FormData();
  form.append("file", file);
  try {
    const preview = await api(`/api/preview/${kind}`, { method: "POST", body: form });
    state.uploads[kind] = { file, preview };
  } catch (err) {
    delete state.uploads[kind];
    root.append(el("div", { class: "callout callout-error" }, icon("warn"), el("div", { text: `${file.name} was not accepted: ${err.message}` })));
  }
  renderInput(kind);
  if (kind === "reference") updateInstructionsPlaceholder();
}

function clearUpload(kind) {
  delete state.uploads[kind];
  $$(".callout-error", panel(kind)).forEach((n) => n.remove());
  renderInput(kind);
  if (kind === "reference") updateInstructionsPlaceholder();
}

function wireDrop(zone, onPick) {
  const input = $("input[type=file]", zone);
  input.addEventListener("change", () => { onPick(input.files[0]); input.value = ""; });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); onPick(e.dataTransfer.files[0]); });
}

function setPersonaSource(source) {
  state.personaSource = source;
  for (const b of $$("#persona-source button")) b.classList.toggle("on", b.dataset.source === source);
  $("[data-drop]", panel("persona")).classList.toggle("hidden", source !== "upload");
  renderInput("persona");
}

function setUseReference(on) {
  state.useReference = on;
  $("#use-reference").checked = on;
  $("#reference-label").textContent = on ? "On · grounded" : "Off · ungrounded";
  $("#reference-on").classList.toggle("hidden", !on);
  $("#reference-off").classList.toggle("hidden", on);
  renderInput("reference");
  renderInput("reference_guide");
}

function setInstructionsSource(source) {
  state.instructionsSource = source;
  for (const b of $$("#instructions-source button")) b.classList.toggle("on", b.dataset.source === source);
  $("#instructions-default").classList.toggle("hidden", source !== "none");
  panel("reference_guide").classList.toggle("hidden", source !== "upload");
  $("#instructions-write").classList.toggle("hidden", source !== "write");
  updateInstructionsPlaceholder();
  renderInput("reference_guide");
}

function updateInstructionsPlaceholder() {
  const defaults = (state.config && state.config.default_instructions) || {};
  const ref = currentInput("reference");
  const kind = ref ? ref.preview.parsed.kind : "data_catalog";
  $("#instructions-text").placeholder = defaults[kind] || "";
}

function renderComplexityOptions() {
  const select = $("#question-complexity");
  const persona = currentInput("persona");
  const prefs = persona ? persona.preview.parsed.question_preferences || {} : {};
  const options = (state.config && state.config.question_complexities) || ["simple", "medium", "complex", "mixed"];
  select.replaceChildren(el("option", { value: "", text: `Persona preference (${prefs.complexity || "mixed"})` }),
    ...options.map((c) => el("option", { value: c, text: c.charAt(0).toUpperCase() + c.slice(1) })));
  select.value = state.questionComplexity;
}

// --- evaluation (run-level metric selection) --------------------------------------------------
function catalog() { return (state.config && state.config.metric_catalog) || []; }
// The run's saved custom-metric workbook (kept by the server unless the user removes it).
function savedWorkbook() {
  const source = state.run && state.run.sources && state.run.sources.custom_metrics;
  return source && source.startsWith("uploaded: ") && !state.removeSavedWorkbook ? fileName(source) : null;
}
function customGiven() { return Boolean($("#custom-metrics-text").value.trim() || state.uploads.custom_metrics || savedWorkbook()); }
function evaluationLocked() { return Boolean(state.run && state.run.evaluation_locked); }
// The selection the run uses: only once locked (a draft is edited in the controls).
function frozenSelection() { return evaluationLocked() ? state.run.metric_selection : null; }

// Fill the Evaluation controls from the run's saved draft / locked selection.
function loadEvaluationDraft() {
  if (!state.run) return;
  const sel = state.run.metric_selection;
  if (sel) {
    state.selectedMetrics = new Set(sel.selected_metrics);
    $("#custom-metrics-text").value = sel.custom_inputs.filter((c) => c.provenance.startsWith("free text item")).map((c) => c.text).join("\n\n");
  }
  $("#scenario-count").value = state.run.scenario_count || (state.config && state.config.scenario_count) || "";
  $("#question-count").value = state.run.question_count || (state.config && state.config.question_count) || "";
  state.questionComplexity = state.run.question_complexity || "";
  state.removeSavedWorkbook = false;
  delete state.uploads.custom_metrics;
  renderComplexityOptions();
  renderInput("custom_metrics");
}

function evaluationForm(lock) {
  const form = new FormData();
  for (const m of catalog().map((x) => x.id).filter((id) => state.selectedMetrics.has(id))) form.append("selected_metrics", m);
  form.append("custom_metrics_text", $("#custom-metrics-text").value);
  if (state.uploads.custom_metrics) form.append("custom_metrics", state.uploads.custom_metrics.file);
  else if (state.removeSavedWorkbook) form.append("remove_custom_metrics", "true");
  if (state.questionComplexity) form.append("question_complexity", state.questionComplexity);
  if ($("#scenario-count").value) form.append("scenario_count", $("#scenario-count").value);
  if ($("#question-count").value) form.append("questions_per_scenario", $("#question-count").value);
  if (lock !== undefined) form.append("lock", lock ? "true" : "false");
  return form;
}

async function lockEvaluation() {
  await withBusy("Locking evaluation", async () => {
    const body = await api(`/api/runs/${encodeURIComponent(state.runId)}/evaluation`, { method: "POST", body: evaluationForm(true) });
    applyRun(body);
    loadEvaluationDraft();
    const sel = state.run.metric_selection;
    log(`Evaluation locked (${plural(sel.selected_metrics.length, "metric")}${sel.custom_inputs.length ? ` + ${sel.custom_inputs.length} custom` : ""})`);
    toast("Evaluation locked");
    showView("connect");
  });
}

async function unlockEvaluation() {
  await withBusy("Unlocking evaluation", async () => {
    const body = await api(`/api/runs/${encodeURIComponent(state.runId)}/evaluation/lock`, { method: "DELETE" });
    applyRun(body);
    loadEvaluationDraft();
    log("Evaluation unlocked for editing");
  });
}
// --- evaluation guidance (judges only; editable at any time, D237) -------------------------
function fillGuidance() {
  const g = state.guidance;
  $("#guidance-text").value = g && g.source === "typed in the UI" ? g.text : "";
  $("#guidance-file").value = "";
}

function renderGuidance() {
  const g = state.guidance;
  $("#guidance-status").textContent = !state.runId
    ? "Start the run first; guidance is saved per run."
    : g ? `Saved: ${g.source} · ${g.text.length} characters · sha256 ${g.sha256.slice(0, 12)}` : "No guidance saved.";
  const given = $("#guidance-text").value.trim() || $("#guidance-file").files.length;
  $("#save-guidance").disabled = !state.runId || state.busy || !given;
  $("#remove-guidance").classList.toggle("hidden", !g);
  $("#remove-guidance").disabled = state.busy;
}

async function saveGuidance(remove) {
  const form = new FormData();
  const file = $("#guidance-file").files[0];
  if (remove) form.append("remove", "true");
  else if (file) form.append("guidance_file", file);
  else form.append("guidance_text", $("#guidance-text").value);
  await withBusy(remove ? "Removing guidance" : "Saving guidance", async () => {
    const body = await api(`/api/runs/${encodeURIComponent(state.runId)}/evaluation-guidance`, { method: "POST", body: form });
    applyRun(body);
    log(remove ? "Evaluation guidance removed" : `Evaluation guidance saved (${state.guidance.source})`);
    toast(remove ? "Guidance removed" : "Guidance saved; the next Evaluate step uses it");
  });
}

// Metrics whose evidence is retrieved context: the connection page then asks for its source.
function needsContext() {
  if (state.metricConfig) return state.metricConfig.definitions.some((d) => (d.required_execution_evidence || []).some((k) => ["retrieved_context", "ranked_context", "citations", "grounding_evidence"].includes(k)));
  const ids = frozenSelection() ? frozenSelection().selected_metrics : [...state.selectedMetrics];
  return catalog().some((m) => m.needs_context && ids.includes(m.id));
}

function renderEvaluation() {
  if (!state.config) return;
  renderGuidance();
  const frozen = frozenSelection();
  const chosen = new Set(frozen ? frozen.selected_metrics : [...state.selectedMetrics]);
  const labels = state.config.metric_categories || {};
  const groups = {};
  for (const m of catalog()) (groups[m.category] = groups[m.category] || []).push(m);
  $("#metric-groups").replaceChildren(...Object.entries(groups).map(([category, metrics]) => el("div", { class: "metric-group" },
    el("h3", { text: labels[category] || human(category) }),
    el("div", { class: "param-grid" }, metrics.map((m) => {
      const box = el("input", { type: "checkbox", disabled: frozen || state.busy ? "" : null });
      box.checked = chosen.has(m.id);
      box.addEventListener("change", () => { if (box.checked) state.selectedMetrics.add(m.id); else state.selectedMetrics.delete(m.id); renderEvaluation(); updateUi(); });
      const notes = [m.needs_context ? "needs retrieved context" : "", m.needs_expectations ? "question-specific expectation" : "", m.counts_toward_verdict ? "" : "reported, not in the verdict"].filter(Boolean).join(" · ");
      return el("label", { class: `param${box.checked ? " active" : ""}` },
        el("div", { class: "param-name" }, box, " ", m.display_name), el("div", { class: "param-help", text: m.description }),
        notes ? el("div", { class: "param-inputs", text: notes }) : null);
    })))));
  const count = chosen.size + (frozen ? frozen.custom_inputs.length : 0);
  const pillNode = $("#metrics-pill");
  pillNode.className = `pill ${count || customGiven() ? "pill-ok" : "pill-req"}`;
  pillNode.textContent = frozen ? `${plural(count, "metric")} (frozen)` : `${plural(chosen.size, "built-in metric")}${customGiven() ? " + custom" : ""}`;
  for (const id of ["custom-metrics-text", "scenario-count", "question-count", "question-complexity"]) $(`#${id}`).disabled = Boolean(frozen) || state.busy;
  for (const zone of $$("[data-drop]", panel("custom_metrics"))) zone.classList.toggle("hidden", Boolean(frozen));
  $("#custom-clear").classList.toggle("hidden", Boolean(frozen) || !(state.uploads.custom_metrics || savedWorkbook()));
  const scenariosDone = done().has("scenarios");
  $("#lock-evaluation").classList.toggle("hidden", Boolean(frozen));
  $("#lock-evaluation").disabled = !state.runId || state.busy || !(state.selectedMetrics.size > 0 || customGiven());
  $("#unlock-evaluation").classList.toggle("hidden", !frozen || scenariosDone || !state.run.metric_selection);
  $("#unlock-evaluation").disabled = state.busy;
  $("#lock-hint").textContent = !state.runId ? "Start the run in Test setup first; your choices here are kept as its draft."
    : frozen ? (scenariosDone ? "Locked: scenarios were generated with this evaluation." : "Locked. Edit it again any time before the scenarios are generated.")
      : state.selectedMetrics.size > 0 || customGiven() ? "Lock the evaluation to continue to the product connection."
        : "Select at least one metric (or add a custom metric).";
  let node;
  if (frozen) {
    const unavailable = state.metricConfig ? state.metricConfig.definitions.filter((d) => d.availability === "unavailable") : [];
    node = callout("info", `Run ${state.run.run_id} uses this selection.`,
      el("div", {}, scenariosDone ? "The selection is part of the run and frozen in its metric configuration. Start a new run to change it." : "The evaluation is locked. Use Edit evaluation to change it before the scenarios are generated.",
        frozen.custom_inputs.length ? list(frozen.custom_inputs.map((c) => `${c.provenance}: ${c.text}`)) : null,
        unavailable.length ? el("div", { class: "hint-warn", text: `Not compiled: ${unavailable.map((d) => `${d.metric_id} (${d.unavailable_reason})`).join("; ")}` }) : null));
  } else if (state.run && state.run.legacy_evaluation_profile) {
    node = callout("warn", "Legacy run.", `This run was created with the ${state.run.legacy_evaluation_profile.toUpperCase()} evaluation profile; its results stay readable. Start a new run to use metric selection.`);
  } else if (state.run) {
    node = callout("warn", `Run ${state.run.run_id}: evaluation draft.`, "Review the metrics, custom metrics and generation settings, then lock the evaluation. Scenarios can be generated only after it is locked.");
  } else {
    node = callout("info", "Applies to the whole run.", "Each selected metric gets one result per question: a score, or a status with its reason (not applicable, cannot generate, evidence unavailable, inconclusive, evaluation error). Retrieval metrics need a context source in Product connection.");
  }
  $("#evaluation-callout").replaceChildren(node);
}

// --- readiness / run creation -----------------------------------------------------------------
function readiness() {
  const persona = Boolean(currentInput("persona"));
  const product = Boolean(currentInput("product"));
  const reference = currentInput("reference");
  const problems = [];
  if (state.useReference && !reference) problems.push("Upload reference data or switch it off.");
  const metrics = state.selectedMetrics.size > 0 || customGiven();
  if (!metrics) problems.push("Select at least one metric (or add a custom metric) in the Evaluation tab.");
  const referenceOk = !state.useReference || Boolean(reference);
  return { persona, product, reference: referenceOk, metrics, problems, ok: persona && product && referenceOk && metrics };
}

function updateReadiness() {
  const r = readiness();
  $("#chk-persona").className = r.persona ? "ok" : "";
  $("#chk-product").className = r.product ? "ok" : "";
  $("#chk-reference").className = r.reference ? "ok" : "";
  $("#chk-metrics").className = r.metrics ? "ok" : "";
  $("#chk-reference").textContent = state.useReference ? "Reference data" : "Reference data (off · ungrounded)";
  $("#chk-metrics").textContent = `Evaluation metrics · ${state.selectedMetrics.size}${customGiven() ? " + custom" : ""}`;
  $("#start-hint").textContent = r.problems.length ? r.problems[0]
    : r.ok ? (state.runId ? "Starting again creates a new run." : "Ready to start.")
      : "Complete the required inputs to start.";
  $("#start-hint").classList.toggle("hint-warn", r.problems.length > 0);
  $("#create-run").disabled = !r.ok || state.busy;
}

async function createRun() {
  const form = new FormData();
  if (state.personaSource === "upload" && state.uploads.persona) form.append("persona", state.uploads.persona.file);
  if (state.uploads.product) form.append("product", state.uploads.product.file);
  if (state.useReference) {
    if (state.uploads.reference) form.append("reference", state.uploads.reference.file);
    if (state.instructionsSource === "upload" && state.uploads.reference_guide) form.append("reference_guide", state.uploads.reference_guide.file);
    if (state.instructionsSource === "write") form.append("reference_instructions", $("#instructions-text").value);
  }
  form.append("use_reference", state.useReference ? "true" : "false");
  form.append("persona_queries", $("#persona-queries").value);
  if (state.questionComplexity) form.append("question_complexity", state.questionComplexity);
  for (const m of catalog().map((x) => x.id).filter((id) => state.selectedMetrics.has(id))) form.append("selected_metrics", m);
  form.append("custom_metrics_text", $("#custom-metrics-text").value);
  if (state.uploads.custom_metrics) form.append("custom_metrics", state.uploads.custom_metrics.file);
  if ($("#scenario-count").value) form.append("scenario_count", $("#scenario-count").value);
  if ($("#question-count").value) form.append("questions_per_scenario", $("#question-count").value);
  await withBusy("Starting run", async () => {
    const body = await api("/api/runs", { method: "POST", body: form });
    state.runId = body.state.run_id;
    history.replaceState(null, "", `${location.pathname}?run=${encodeURIComponent(state.runId)}#evaluation`);
    for (const node of $$(".step")) { node.classList.remove("done", "error", "running"); $(".step-result", node).textContent = ""; }
    applyRun(body);
    log(`Run ${state.runId} started (${plural(state.run.metric_selection.selected_metrics.length, "metric")}${state.run.metric_selection.custom_inputs.length ? ` + ${state.run.metric_selection.custom_inputs.length} custom` : ""}, ${state.useReference ? "grounded" : "ungrounded"})`);
    loadEvaluationDraft();
    toast(`Run ${state.runId} started: review and lock the evaluation`);
    showView("evaluation");
    loadRecent();
  });
}

// --- product connection -----------------------------------------------------------------------
function setContractSource(source) {
  state.contractSource = source;
  for (const b of $$("#contract-source button")) b.classList.toggle("on", b.dataset.source === source);
  $("#contract-upload").classList.toggle("hidden", source !== "upload");
  $("#contract-mock").classList.toggle("hidden", source !== "mock");
  updateUi();
}

function setContractFile(file) {
  state.contractFile = file || null;
  $("#contract-file").textContent = file ? `Selected: ${file.name}` : "Request body template and response mapping";
  $("[data-contract-drop]").classList.toggle("chosen", Boolean(file));
  updateUi();
}

function setContextSource(source) {
  state.contextSource = source;
  for (const b of $$("#context-source button")) b.classList.toggle("on", b.dataset.source === source);
  const help = {
    response_field: "The contract maps the retrieved chunks (and citations) in the product response: response_mapping.context / citations.",
    context_api: "The contract has a context_api section: the context of the SAME execution is fetched after the answer, correlated by request / session / trace id.",
    none: "No retrieved context: retrieval and grounding metrics are reported as evidence unavailable.",
  };
  $("#context-help").textContent = help[source];
}

function canConnect() {
  if (!state.runId || state.busy) return false;
  const url = $("#product-url").value.trim();
  return state.contractSource === "mock" ? Boolean(url) : Boolean(url && state.contractFile);
}

async function connect() {
  const url = $("#product-url").value.trim();
  await withBusy("Connecting", async () => {
    if (state.contractSource === "mock") {
      const body = await postJson(`/api/runs/${state.runId}/contract`, { url });
      state.run = body.state;
      log(`Mock product contract for ${url}`);
      toast("Mock product ready");
    } else {
      const form = new FormData();
      form.append("url", url);
      form.append("contract", state.contractFile);
      if (needsContext()) form.append("context_source", state.contextSource);
      const body = await api(`/api/runs/${state.runId}/connection`, { method: "POST", body: form });
      state.run = body.state;
      log(`Connected to ${state.run.connection.product_url} (${state.contractFile.name})`);
      toast("Product connected");
    }
    renderConnection();
  });
}

async function disconnect() {
  await withBusy("Disconnecting", async () => {
    const body = await api(`/api/runs/${state.runId}/connection`, { method: "DELETE" });
    state.run = body.state;
    renderConnection();
    log("Product disconnected - the configured adapter answers");
  });
}

function renderConnection() {
  const box = $("#contract");
  box.replaceChildren();
  const status = $("#conn-pill");
  const run = state.run || {};
  const fact = (k, v) => el("div", { class: "fact" }, el("b", { text: k }), el("span", { text: v }));
  const source = run.connection ? "upload" : run.contract ? "mock" : null;
  if (source && source !== state.contractSource) {
    state.contractSource = source;
    for (const b of $$("#contract-source button")) b.classList.toggle("on", b.dataset.source === source);
    $("#contract-upload").classList.toggle("hidden", source !== "upload");
    $("#contract-mock").classList.toggle("hidden", source !== "mock");
  }
  if (run.connection) {
    const c = run.connection;
    status.className = "pill pill-ok"; status.replaceChildren(icon("check"), "Connected");
    if (!$("#product-url").value) $("#product-url").value = c.product_url;
    const mapping = c.response_mapping;
    const outcomes = ["answered", mapping.clarification_when && "clarification", mapping.denied_when && "denied", "error"].filter(Boolean);
    const adapter = c.adapter_type === "http_sse" ? "streaming (SSE) · " : "";
    box.append(
      el("div", { class: "facts" },
        fact("Endpoint", `${adapter}${c.method} ${c.product_url}`), fact("Contract", c.contract_file),
        fact("Headers", c.header_names.length ? `${c.header_names.join(", ")} (values hidden)` : "none"),
        fact("Recognised outcomes", outcomes.join(", ")),
        fact("Retrieved context", c.context_source === "context_api" ? `context API ${c.context_api_url}` : c.context_source === "response_field" ? "in the product response" : "not available")),
      el("details", { class: "raw" }, el("summary", { text: "View request template and response mapping" }),
        el("pre", { text: JSON.stringify({ timeout: c.timeout, request_template: c.request_template, response_mapping: mapping, stream: c.stream || undefined }, null, 2) })),
    );
    return;
  }
  if (run.contract) {
    const contract = run.contract;
    status.className = "pill pill-warn"; status.textContent = "Mock product";
    if (!$("#product-url").value) $("#product-url").value = contract.product_url;
    box.append(
      el("div", { class: "facts" },
        fact("Product", contract.product_id), fact("Endpoint", `${contract.ask.method} ${contract.ask.path}`),
        fact("Outcomes", contract.outcomes.join(", ")), fact("Contract", `${contract.contract_version} (mock)`)),
      el("details", { class: "raw" }, el("summary", { text: "View contract JSON" }), el("pre", { text: JSON.stringify(contract, null, 2) })),
      callout("warn", "Demo mode.", "No real call is made: answers come from an AI stand-in for the described product, not from the product itself."),
    );
    return;
  }
  status.className = "pill pill-neutral"; status.textContent = "Not connected";
}

// --- pipeline ---------------------------------------------------------------------------------
function done() { return new Set(state.run ? state.run.steps_done : []); }
function adapterName() { return state.config ? state.config.product_adapter : "configured"; }

function updateUi() {
  const d = done();
  STEPS.forEach((step, i) => {
    const node = $(`.step[data-step="${step}"]`);
    const ready = Boolean(state.runId) && (i === 0 ? evaluationLocked() : d.has(STEPS[i - 1]));
    const btn = $(".step-btn", node);
    btn.disabled = !ready || state.busy;
    btn.textContent = d.has(step) ? "Re-run" : "Run";
    btn.classList.toggle("btn-primary", ready && !d.has(step));
    if (!node.classList.contains("running") && !node.classList.contains("error")) {
      node.classList.toggle("done", d.has(step));
      node.classList.toggle("ready", ready && !d.has(step));
      const status = $(".step-status", node);
      status.replaceChildren(...(d.has(step) ? [icon("check"), "Done"] : [ready ? "Ready" : "Pending"]));
      $(".step-dot", node).textContent = d.has(step) ? "" : String(i + 1);
      if (d.has(step)) $(".step-dot", node).append(icon("check"));
    }
  });
  const completed = STEPS.filter((s) => d.has(s)).length;
  $("#progress-bar").style.width = `${(completed / STEPS.length) * 100}%`;
  $("#progress-label").textContent = `${completed} of ${STEPS.length} steps`;
  $("#run-all").disabled = !state.runId || state.busy || completed === STEPS.length || !evaluationLocked();
  $("#connect").disabled = !canConnect();
  $("#disconnect").disabled = !state.runId || state.busy || !(state.run && (state.run.connection || state.run.contract));
  updateReadiness();

  const connection = state.run && state.run.connection;
  const mock = state.run && state.run.contract;
  const adapter = adapterName();
  $("#execute-desc").textContent = connection
    ? `Each question is sent to ${connection.product_url}; answers are recorded, never executed.`
    : mock ? "Each question goes to the mock product (AI stand-in); answers are recorded."
      : `Each question goes to the configured "${adapter}" adapter; answers are recorded.`;
  $("#adapter-note").textContent = connection || mock ? "" : state.runId
    ? `Not connected: questions would go to the configured "${adapter}" adapter.`
    : "Start a run in Test setup first.";
  const grounded = state.run ? state.run.sources.reference !== "off" : state.useReference;
  $("#scenarios-desc").textContent = grounded
    ? "Creates business scenarios from Persona, Product and the operational capabilities in Reference data."
    : "Business goals this persona would pursue with the product.";
  const defaults = state.config || {};
  const scenarios = (state.run && state.run.scenario_count) || defaults.scenario_count;
  const perScenario = (state.run && state.run.question_count) || defaults.question_count;
  $("#scenario-count-note").textContent = scenarios ? `${scenarios} scenario(s) · set in Evaluation` : "";
  $("#question-count-note").textContent = perScenario ? `${perScenario} per scenario · set in Evaluation` : "";
  $("#context-block").classList.toggle("hidden", !needsContext());
  renderRunCallout(connection, mock, d);

  const csv = $("#download-csv");
  csv.classList.toggle("disabled", !state.runId);
  csv.href = state.runId ? `/api/runs/${state.runId}/results.csv` : "#";
  const kpiCsv = $("#download-kpis");
  kpiCsv.classList.toggle("disabled", !state.kpis);
  kpiCsv.href = state.kpis ? `/api/runs/${state.runId}/artifacts/kpis.csv` : "#";

  // run card + crumb
  const runId = $("#run-id");
  runId.className = state.runId ? "run-id" : "run-id muted";
  runId.textContent = state.runId || "No run started";
  $("#run-mode").replaceChildren(...(state.run ? [pill(grounded ? "Grounded" : "Ungrounded", grounded ? "blue" : "neutral")] : []));
  const sources = $("#run-sources");
  sources.replaceChildren();
  if (state.run) {
    const show = (value) => (value === "sample" ? "Sample" : value === "off" ? "Off" : value === "none" ? "none" : fileName(value) || "—");
    for (const [kind, label] of Object.entries(SOURCE_LABELS)) sources.append(el("dt", { text: label }), el("dd", { text: show(state.run.sources[kind]) }));
    sources.append(el("dt", { text: "Question complexity" }), el("dd", { text: state.run.question_complexity || "persona preference" }));
    const sel = state.run.metric_selection;
    sources.append(el("dt", { text: "Metrics" }), el("dd", { text: sel ? `${sel.selected_metrics.map(metricLabel).join(", ") || "none"}${sel.custom_inputs.length ? ` + ${sel.custom_inputs.length} custom` : ""}` : `legacy ${state.run.legacy_evaluation_profile || ""} profile` }));
    sources.append(el("dt", { text: "Answers from" }), el("dd", { text: connection ? connection.product_url : mock ? "Mock product" : `Configured adapter (${adapter})` }));
  }
  $("#crumb").replaceChildren(...(state.runId ? [el("span", { class: "sep", text: "/" }), el("code", { text: state.runId })] : []));

  // navigation status
  const k = state.kpis;
  const sel = frozenSelection();
  const metricCount = sel ? sel.selected_metrics.length + sel.custom_inputs.length : state.selectedMetrics.size;
  const navStatus = {
    setup: state.runId ? "Run created" : readiness().ok ? "Ready to start" : "Persona, product, reference",
    evaluation: `${plural(metricCount, "metric")}${!sel && customGiven() ? " + custom" : ""}${state.run ? (sel ? " · locked" : " · draft") : ""}`,
    connect: connection ? "Connected" : mock ? "Mock product" : "Not connected",
    run: `${completed} of ${STEPS.length} steps`,
    report: k ? `Accuracy ${pct(k.accuracy)}` : "Not available yet",
  };
  const navDone = { setup: Boolean(state.runId), evaluation: evaluationLocked(), connect: Boolean(connection || mock), run: completed === STEPS.length, report: Boolean(k) };
  for (const node of $$("[data-nav]")) {
    $("[data-nav-status]", node).textContent = navStatus[node.dataset.nav];
    node.classList.toggle("done", navDone[node.dataset.nav]);
  }
}
function maxStep() { return (state.config && state.config.web && state.config.web.max_step_count) || 100; }

function renderRunCallout(connection, mock, d) {
  const box = $("#run-callout");
  const go = (view, text) => el("a", { class: "btn btn-sm", href: `#${view}`, text });
  let node;
  if (!state.runId) node = callout("info", "No run yet.", "Complete the test setup and start a run first.", go("setup", "Go to test setup"));
  else if (!evaluationLocked()) node = callout("warn", "Evaluation not locked.", "Review the metrics and generation settings, then lock the evaluation before generating scenarios.", go("evaluation", "Go to evaluation"));
  else if (connection) node = callout("info", "Connected.", `Questions go to ${connection.product_url}.`);
  else if (mock) node = callout("warn", "Mock product (demo).", "Answers come from an AI stand-in, not from the real product.", go("connect", "Change"));
  else if (!d.has("execute")) node = callout("warn", "No product connected.", `Questions will go to the configured "${adapterName()}" adapter (test data), not to your product.`, go("connect", "Connect product"));
  else node = callout("warn", "No product connected.", `The answers in this run came from the configured "${adapterName()}" adapter (test data), not from your product. Connect the product and run "Ask the product" again.`, go("connect", "Connect product"));
  box.replaceChildren(...(node ? [node] : []));
}

function describe(step, detail) {
  if (step === "scenarios") return `${detail.accepted} accepted, ${detail.rejected} rejected${detail.reference_packs ? ` · ${detail.reference_packs} reference pack(s)` : ""}${detail.unavailable_metrics && detail.unavailable_metrics.length ? ` · ${detail.unavailable_metrics.length} custom metric(s) not compiled` : ""}`;
  if (step === "questions") return `${detail.accepted} questions, ${detail.rejected} rejected`;
  if (step === "execute") return `${detail.executions} answers via ${detail.product}${detail.resumed_from_question ? ` · resumed from question ${detail.resumed_from_question}` : ""}`;
  return `${detail.evaluations} evaluated · ${detail.bug_candidates} bug candidate(s)${detail.accuracy != null ? ` · accuracy ${pct(detail.accuracy)}` : ""}`;
}

function stepBody(step) {
  const body = {};
  if (step === "execute" && $("#start-from").value) body.start_from_question = Number($("#start-from").value);
  return body;
}

async function runStep(step) {
  const node = $(`.step[data-step="${step}"]`);
  node.classList.remove("done", "error", "ready"); node.classList.add("running");
  $(".step-status", node).replaceChildren(el("span", { class: "spinner" }), "Running");
  $(".step-result", node).textContent = "";
  const started = performance.now();
  try {
    const body = await postJson(`/api/runs/${state.runId}/steps/${step}`, stepBody(step));
    node.classList.remove("running");
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    $(".step-result", node).textContent = `${describe(step, body.detail)} · ${secs}s`;
    if (step === "execute") $("#start-from").value = "";
    for (const later of STEPS.slice(STEPS.indexOf(step) + 1)) {
      const laterNode = $(`.step[data-step="${later}"]`);
      laterNode.classList.remove("error");
      $(".step-result", laterNode).textContent = "";
    }
    applyRun(body);
    log(`${step}: ${describe(step, body.detail)} (${secs}s)`);
    for (const w of body.detail.warnings || []) log(`reference: ${w}`, true);
    return true;
  } catch (err) {
    node.classList.remove("running"); node.classList.add("error");
    $(".step-status", node).replaceChildren(icon("warn"), "Error");
    $(".step-result", node).textContent = err.message;
    log(`${step} failed: ${err.message}`, true);
    toast(`${human(step)} failed: ${err.message}`, "error");
    return false;
  }
}

async function runAll() {
  let ok = true;
  await withBusy("Running pipeline", async () => {
    for (const step of STEPS) {
      if (done().has(step)) continue;
      if (!(await runStep(step))) { ok = false; break; }
    }
  });
  if (ok && done().has("evaluate")) { toast("Evaluation complete"); showView("report"); }
}

function renderContracts() {
  const rows = questionRows();
  $("#contracts-card").classList.toggle("hidden", !rows.length);
  $("#contracts-table thead").replaceChildren(el("tr", {}, ["Question", "Expected behaviour", "Success criteria"].map((h) => el("th", { text: h }))));
  $("#contracts-table tbody").replaceChildren(...rows.map((row) => {
    const tr = el("tr", { tabindex: "0" },
      el("td", { class: "q" }, el("b", { text: row.question }), el("small", { text: row.question_id })),
      el("td", { class: "wide" }, el("div", { class: "clamp", text: row.expected_behavior })),
      el("td", {}, criteriaCell(criteriaOf(row))));
    tr.addEventListener("click", () => openDrawer(row));
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDrawer(row); } });
    return tr;
  }));
}

function criteriaCell(criteria) {
  const sheet = cheatSheetItems(criteria);
  return list([
    ...(criteria.general || []),
    ...metricCriteria(criteria),
    ...(sheet.length ? [`Cheat sheet: ${sheet.length} item(s) · see details`] : []),
  ]);
}

// --- report -----------------------------------------------------------------------------------
function applyRun(body) {
  state.run = body.state;
  state.rows = body.rows || [];
  state.scenarios = body.scenarios || [];
  state.packs = body.reference_packs || [];
  state.kpis = body.kpis || null;
  state.evaluations = body.evaluations || [];
  state.executions = body.executions || [];
  state.issues = body.issue_summary || null;
  state.questions = body.questions || [];
  state.metricConfig = body.metric_config || null;
  state.metricIds = body.metric_ids || [];
  state.guidance = body.evaluation_guidance || null;
  fillGuidance();
  if (state.run && state.run.metric_selection) state.selectedMetrics = new Set(state.run.metric_selection.selected_metrics);
  renderConnection();
  renderReport(body);
  renderContracts();
  renderEvaluation();
  updateUi();
}

function questionRows() { return state.rows.filter((r) => r.question_id); }
function verdictCounts() {
  const rows = questionRows();
  const by = (v) => rows.filter((r) => r.final_verdict === v).length;
  return { pass: by("pass"), fail: by("fail"), inconclusive: by("inconclusive"), recorded: by("recorded_only") };
}
function activeMetrics() {
  if (state.metricIds.length) return state.metricIds;
  if (state.kpis && state.kpis.metrics.length) return state.kpis.metrics.map((x) => x.metric);
  if (state.run && state.run.metric_selection) return state.run.metric_selection.selected_metrics;
  return [];
}

function renderReport(body) {
  renderMeta();
  renderNotice();
  renderSummary();
  renderMetrics();
  renderIssues();
  renderBugs();
  renderBreakdown();
  renderTable();
  const rejected = $("#rejected");
  rejected.replaceChildren();
  for (const r of body.rejected_scenarios || []) rejected.append(el("div", { class: "rejected-item" }, el("b", { text: `Scenario: ${r.payload.title || "(no title)"}` }), el("div", { text: r.reasons.join("; ") })));
  for (const r of body.rejected_questions || []) rejected.append(el("div", { class: "rejected-item" }, el("b", { text: `Question (${r.scenario_id}): ${r.payload.utterance || "(none)"}` }), el("div", { text: r.reasons.join("; ") })));
  if (!rejected.children.length) rejected.append(el("p", { class: "muted", text: "Nothing was rejected." }));
  $("#artifacts").replaceChildren(...(body.artifacts || []).map((name) => el("a", { class: "btn btn-sm", href: `/api/runs/${state.runId}/artifacts/${encodeURIComponent(name)}` }, icon("download"), name)));
}

function renderMeta() {
  const k = state.kpis;
  const run = state.run;
  const parts = [];
  if (k) parts.push(k.product_id, k.mode === "grounded" ? "Grounded (with reference data)" : "Ungrounded (no reference data)", k.selected_metrics.length ? plural(k.selected_metrics.length, "metric") : `legacy ${run && run.legacy_evaluation_profile ? run.legacy_evaluation_profile.toUpperCase() : ""} profile`, `${k.questions_asked} questions`, `Evaluated ${when(k.computed_at)}`);
  else if (run) parts.push(fileName(run.sources.product) || run.sources.product, `Started ${when(run.created_at)}`, "Results fill in as the steps complete.");
  $("#report-meta").textContent = parts.length ? parts.join("  ·  ") : "Run the tests to produce a report.";
}

function renderNotice() {
  const box = $("#report-notice");
  const k = state.kpis;
  const go = (view, text) => el("a", { class: "btn btn-sm", href: `#${view}`, text });
  let node = null;
  if (!state.runId) node = callout("info", "No report yet.", "Set up and run the tests to produce a report.", go("setup", "Go to test setup"));
  else if (!k) node = callout("info", "Evaluation has not run yet.", "Scenarios, questions and answers appear below as each step completes.", go("run", "Go to run tests"));
  else if (k.questions_asked && k.judged === 0) {
    const outcomes = Object.entries(k.outcomes).filter(([, n]) => n).map(([o, n]) => `${n} ${OUTCOME_LABELS[o] || o}`).join(", ");
    const example = questionRows().find((r) => r.product_outcome === "error" && r.product_message);
    node = callout("warn", "No answers could be judged.",
      el("div", {}, `Of ${k.questions_asked} question(s): ${outcomes}. Answers and substantive refusals are scored; unresolved clarifications, errors and timeouts are recorded only.`,
        example ? el("div", { class: "quote", text: `Product message: ${example.product_message}` }) : null),
      go("connect", "Check connection"));
  } else {
    const failed = (k.outcomes.error || 0) + (k.outcomes.timeout || 0);
    if (failed) node = callout("warn", `${failed} of ${k.questions_asked} question(s) got no answer.`, "Errors and timeouts lower availability and are not scored.");
  }
  const extra = k && k.compiler_failures && k.compiler_failures.length
    ? callout("warn", `${plural(k.compiler_failures.length, "custom metric")} could not be compiled.`, list(k.compiler_failures.map((f) => `${f.metric_id} (${f.provenance}): ${f.reason}`)))
    : null;
  box.replaceChildren(...[node, extra].filter(Boolean));
}

function tile(label, ...content) { return el("div", { class: "tile" }, el("div", { class: "tile-label", text: label }), ...content); }

function renderSummary() {
  const k = state.kpis;
  const c = verdictCounts();
  const total = c.pass + c.fail + c.inconclusive + c.recorded;
  const judged = c.pass + c.fail + c.inconclusive;
  const rate = k ? (k.accuracy != null ? k.accuracy : k.task_success_rate) : null;
  const ring = el("div", { class: `ring${rate == null ? " empty" : ""}`, style: `--p:${rate == null ? 0 : Math.round(rate * 100)}` }, el("span", { text: pct(rate) }));
  const accuracy = tile("Accuracy", el("div", { class: "tile-row" }, ring,
    el("div", {}, el("div", { class: "tile-value small", text: k ? `${c.pass} passed` : "—" }),
      el("div", { class: "tile-sub", text: k ? `${c.pass} of ${judged} evaluated answers` : "available after evaluation" }))),
  el("div", { class: "tile-sub", text: `Provisional verdict policy: passes only when every scored selected metric is perfect${k && k.verdict_policy ? ` (${k.verdict_policy})` : ""}` }));

  const stack = el("div", { class: "stack" }, ...[["pass", c.pass], ["fail", c.fail], ["inconclusive", c.inconclusive], ["recorded", c.recorded]]
    .filter(([, n]) => n && total).map(([cls, n]) => el("div", { class: `c-${cls}`, style: `width:${(n / total) * 100}%`, title: `${cls}: ${n}` })));
  const legendItem = (cls, label, n) => el("span", {}, el("i", { class: `c-${cls}` }), label, el("b", { text: String(n) }));
  const verdicts = tile("Verdicts", stack, el("div", { class: "legend" },
    legendItem("pass", "Pass", c.pass), legendItem("fail", "Fail", c.fail),
    legendItem("inconclusive", "Inconclusive", c.inconclusive), legendItem("recorded", "Not judged", c.recorded)),
  el("div", { class: "tile-sub", text: `${questionRows().length} question(s) · ${new Set(state.rows.map((r) => r.scenario_id)).size} scenario(s)` }));

  const health = tile("Product availability", el("div", { class: "tile-value", text: k ? pct(k.availability) : "—" }),
    el("div", { class: "stat-pair" },
      el("div", {}, el("b", { text: k ? ms(k.latency_p50_ms) : "—" }), el("small", { text: "Median latency (p50)" })),
      el("div", {}, el("b", { text: k ? ms(k.latency_p95_ms) : "—" }), el("small", { text: "Slowest 5% (p95)" }))));

  const severities = k ? Object.entries(k.new_bugs_by_severity).filter(([, n]) => n).sort((a, b) => SEVERITY_RANK[b[0]] - SEVERITY_RANK[a[0]]) : [];
  const bugs = tile("New bug candidates", el("div", { class: "tile-value", text: k ? String(k.new_bugs) : "—" }),
    severities.length ? el("div", { class: "sev-row" }, severities.map(([sev, n]) => pill(`${n} ${sev}`, sev))) : null,
    el("div", { class: "tile-sub", text: k ? `${k.known_gap_bugs} more explained by known gaps` : "available after evaluation" }));
  $("#summary").replaceChildren(accuracy, verdicts, health, bugs);
}

function renderMetrics() {
  const k = state.kpis;
  const metrics = activeMetrics();
  $("#metric-cards").className = `metrics metrics-${Math.min(Math.max(metrics.length, 1), 4)}`;
  $("#metric-cards").replaceChildren(...metrics.map((name) => {
    const info = metricDef(name);
    const m = k && k.metrics.find((x) => x.metric === name);
    const head = el("div", { class: "metric-name", text: info.label });
    const desc = el("div", { class: "metric-desc", text: info.help });
    const statusCounts = m ? [["not applicable", m.not_applicable], ["cannot generate", m.cannot_generate], ["evidence unavailable", m.evidence_unavailable], ["inconclusive", m.inconclusive], ["error", m.evaluation_error], ["unavailable", m.unavailable || 0]]
      .filter(([, n]) => n).map(([label, n]) => `${n} ${label}`).join(" · ") : "";
    const notes = m && m.notes.length ? el("div", { class: "metric-desc", text: m.notes.join(" · ") }) : null;
    const mrr = name === "mrr" && k && k.mrr ? el("div", { class: "metric-desc", text: `Run MRR over ${plural(k.mrr.denominator, "question")}; ${k.mrr.excluded_missing_rank} excluded for missing rank, ${k.mrr.excluded_other} other exclusions` }) : null;
    if (!m || m.mean == null) {
      const reasons = m ? Object.keys(m.reasons || {}).slice(0, 2) : [];
      return el("div", { class: "metric na" }, head, desc, el("div", { class: "metric-value", text: "—" }), el("div", { class: "meter" }),
        el("div", { class: "metric-foot" }, el("span", { text: !k ? "Available after evaluation" : statusCounts || "No scored answers" })),
        reasons.length ? el("div", { class: "metric-desc", text: reasons.join(" | ") }) : null, notes, mrr);
    }
    const meterCls = m.mean >= 1 ? "good" : m.mean <= 0 ? "bad" : "";
    return el("div", { class: "metric" }, head, desc,
      el("div", { class: "metric-value" }, pct(m.mean), el("small", { text: `mean ${m.mean.toFixed(2)}` })),
      el("div", { class: `meter ${meterCls}` }, el("div", { style: `width:${Math.max(m.mean * 100, m.mean > 0 ? 2 : 0)}%` })),
      el("div", { class: "metric-foot" },
        el("span", { text: `${pct(m.pass_rate)} at full score` }),
        el("span", { text: `n = ${m.n}${statusCounts ? ` · ${statusCounts}` : ""}` })), notes, mrr);
  }));
}

function renderIssues() {
  const block = $("#issues-block");
  const summary = state.issues;
  block.classList.toggle("hidden", !state.kpis);
  if (!summary || !summary.issues.length) {
    $("#issues-sub").textContent = "";
    $("#issues").replaceChildren(el("div", { class: "empty-card", text: state.kpis ? "No issues were found in this run." : "" }));
    return;
  }
  const known = summary.issues.filter((i) => i.known_gap).length;
  $("#issues-sub").replaceChildren(...[el("b", { text: plural(summary.issues.length, "consolidated issue") }),
    el("span", { text: `${plural(summary.total_findings, "finding")} across ${plural(summary.affected_questions, "answer")}` }),
    known ? el("span", { text: plural(known, "known limitation") }) : null].filter(Boolean));
  $("#issues").replaceChildren(...summary.issues.map((issue) => {
    const view = el("button", { class: "link-btn", type: "button", text: `View ${plural(issue.affected_question_ids.length, "failing question")}` });
    view.addEventListener("click", () => filterToQuestions(issue.affected_question_ids, `${issue.issue_id} · ${issue.title}`));
    return el("div", { class: `issue sev-${issue.severity}` },
      el("div", { class: "issue-head" },
        el("span", { class: "issue-id", text: `${issue.issue_id} · ${human(issue.theme)}` }),
        el("span", { class: "issue-pills" }, ...issue.categories.map((cat) => pill(metricLabel(cat).toLowerCase(), "blue")),
          issue.known_gap ? pill("known limitation", "neutral") : null, pill(issue.severity, issue.severity))),
      el("div", { class: "issue-title", text: issue.title }),
      el("div", { class: "issue-action" }, el("b", { text: "Recommended action: " }), issue.recommended_action),
      issue.examples.length ? el("ul", { class: "issue-examples" }, issue.examples.map((x) => el("li", { text: x }))) : null,
      el("div", { class: "issue-foot" }, el("span", { text: `${plural(issue.occurrences, "finding")} · ${plural(issue.affected_question_ids.length, "affected answer")}` })),
      view);
  }));
}

function filterToQuestions(ids, label) {
  state.filterIds = new Set(ids);
  state.filterLabel = label;
  state.filter = "all";
  for (const x of $$("#filter button")) x.classList.toggle("on", x.dataset.filter === "all");
  renderTable();
  $("#results-table").scrollIntoView({ behavior: "smooth", block: "start" });
}

function bugEntries() {
  const out = [];
  for (const ev of state.evaluations) {
    for (const bug of ev.bug_candidates) {
      const findings = ev.findings.filter((f) => bug.finding_ids.includes(f.finding_id));
      const severity = findings.reduce((best, f) => (SEVERITY_RANK[f.severity] > SEVERITY_RANK[best] ? f.severity : best), "low");
      out.push({ bug, ev, severity, known: findings.length > 0 && findings.every((f) => f.matches_known_gap) });
    }
  }
  return out.sort((a, b) => SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity] || a.bug.bug_id.localeCompare(b.bug.bug_id));
}

function renderBugs() {
  const block = $("#bugs-block");
  block.classList.toggle("hidden", !state.kpis);
  const entries = bugEntries();
  if (!entries.length) { $("#bugs").replaceChildren(el("div", { class: "empty-card", text: "No bug candidates in this run." })); return; }
  $("#bugs").replaceChildren(el("div", { class: "bug-list" }, entries.map(({ bug, severity, known }) => {
    const row = state.rows.find((r) => r.question_id === bug.question_id);
    const button = el("button", { class: `bug sev-${severity}`, type: "button" },
      el("span", { class: "bug-ic" }, icon("bug")),
      el("span", {}, el("div", { class: "bug-title", text: bug.title }),
        el("div", { class: "bug-meta", text: `${bug.bug_id} · ${bug.question_id} · Reproduce: “${bug.reproduction_question}”` })),
      el("span", { class: "bug-side" }, known ? pill("known gap", "neutral") : null, pill(severity, severity)));
    if (row) button.addEventListener("click", () => openDrawer(row));
    return button;
  })));
}

function scoreChip(value, judged, status) {
  if (status && status !== "scored") return el("span", { class: `score ${status === "inconclusive" ? "mid" : "na"}`, title: human(status), text: NON_SCORED_LABELS[status] || human(status) });
  if (!value) return el("span", { class: "score na", text: judged ? "n/a" : "" });
  if (value === "inconclusive") return el("span", { class: "score mid", title: "inconclusive", text: "?" });
  const v = Number(value);
  if (Number.isNaN(v)) return el("span", { class: "score na", text: "n/a" });
  const cls = v >= 1 ? "good" : v <= 0 ? "bad" : "mid";
  return el("span", { class: `score ${cls}`, text: Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/0$/, "") });
}

function renderBreakdown() {
  const k = state.kpis;
  const groups = k ? k[state.breakdownBy] || [] : [];
  const metrics = activeMetrics();
  $("#breakdown-block").classList.toggle("hidden", !k || (!k.by_task_type.length && !k.by_capability.length));
  const head = el("tr", {}, el("th", { text: state.breakdownBy === "by_task_type" ? "Task type" : "Capability" }), el("th", { class: "num", text: "Questions" }),
    el("th", { class: "num", text: "Judged" }), el("th", { class: "num", text: "Pass rate" }), metrics.map((m) => el("th", { class: "num", text: metricLabel(m) })));
  $("#breakdown-table thead").replaceChildren(head);
  $("#breakdown-table tbody").replaceChildren(...(groups.length ? groups.map((g) => el("tr", {},
    el("td", { text: human(g.group) }), el("td", { class: "num", text: String(g.questions) }), el("td", { class: "num", text: String(g.judged) }),
    el("td", { class: "num", text: pct(g.pass_rate) }),
    metrics.map((m) => el("td", { class: "num" }, scoreChip(g.metric_means[m] == null ? "" : String(Math.round(g.metric_means[m] * 100) / 100), g.judged > 0)))))
    : [el("tr", {}, el("td", { class: "empty", colspan: String(4 + metrics.length), text: "No data." }))]));
}

function mainColumns() {
  return [["question_id", "ID"], ["question", "Question"], ["expected_behavior", "Expected behaviour"], ["product_outcome", "Outcome"], ["response_classification", "Response kind"], ["final_verdict", "Verdict"],
    ...activeMetrics().map((m) => [m, metricLabel(m)]), ["bug_ids", "Bug"]];
}

function matchesFilter(r) {
  if (state.filterIds && !state.filterIds.has(r.question_id)) return false;
  if (state.filter === "bugs") return Boolean(r.bug_ids);
  if (state.filter !== "all") return r.final_verdict === state.filter;
  return true;
}
function filteredRows() {
  const q = state.search.trim().toLowerCase();
  return state.rows.filter((r) => matchesFilter(r) && (!q || Object.values(r).some((v) => v && v.toLowerCase().includes(q))));
}

function renderFilterCounts() {
  const counts = {
    all: state.rows.length, fail: 0, inconclusive: 0, pass: 0, recorded_only: 0, bugs: state.rows.filter((r) => r.bug_ids).length,
  };
  for (const r of state.rows) if (r.final_verdict in counts) counts[r.final_verdict] += 1;
  for (const b of $$("#filter button")) {
    b.dataset.label = b.dataset.label || b.textContent;
    b.replaceChildren(b.dataset.label, el("span", { class: "count", text: String(counts[b.dataset.filter]) }));
  }
  const chip = $("#issue-filter");
  if (state.filterIds) {
    const clear = el("button", { class: "icon-btn small", type: "button", "aria-label": "Clear issue filter" }, icon("close"));
    clear.addEventListener("click", () => { state.filterIds = null; state.filterLabel = ""; renderTable(); });
    chip.replaceChildren(el("span", { text: `Showing ${state.filterLabel}` }), clear);
    chip.classList.remove("hidden");
  } else chip.classList.add("hidden");
}

function renderTable() {
  renderFilterCounts();
  const all = state.allColumns && state.rows.length;
  const columns = all
    ? Object.keys(state.rows[0]).filter((key) => state.rows.some((r) => r[key])).map((k) => [k, human(k)])
    : mainColumns();
  const metrics = activeMetrics();
  const rows = filteredRows();
  $("#results-table thead").replaceChildren(el("tr", {}, columns.map(([key, label]) => el("th", { class: metrics.includes(key) && !all ? "num" : null, text: label }))));
  const body = $("#results-table tbody");
  if (!rows.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: String(columns.length), class: "empty", text: state.rows.length ? "No rows match the filter." : "No results yet. Start a run and run the tests." })));
    return;
  }
  body.replaceChildren(...rows.map((row) => {
    const judged = row.final_verdict && row.final_verdict !== "recorded_only";
    const cells = columns.map(([key]) => {
      const value = row[key] || "";
      if (all) {
        if (key === "final_verdict" && value) return el("td", { class: "nowrap" }, verdictPill(value));
        if (value.length <= 40 && !value.includes("\n")) return el("td", { class: "nowrap", text: value });
        return el("td", { class: "wide" }, el("div", { class: "clamp", text: value }));
      }
      if (key === "question") {
        return el("td", { class: "q" }, el("b", { text: value || row.scenario_title }), el("small", { text: value ? row.scenario_title : "Scenario without a question yet" }));
      }
      if (key === "expected_behavior") return el("td", { class: "wide" }, el("div", { class: "clamp", text: value }));
      if (key === "response_classification") return el("td", { class: "nowrap", text: human(value) });
      if (key === "final_verdict") return el("td", { class: "nowrap" }, value ? verdictPill(value) : "");
      if (key === "product_outcome") return el("td", { class: "nowrap" }, value ? el("span", { class: `outcome ${value}`, text: OUTCOME_LABELS[value] || value }) : "");
      if (key === "question_id" || key === "bug_ids") return el("td", { class: "id", text: split(value).join(", ") });
      if (metrics.includes(key)) return el("td", { class: "num" }, scoreChip(value, judged, row[`${key}_status`]));
      return el("td", { text: value });
    });
    const tr = el("tr", { tabindex: "0" }, cells);
    tr.addEventListener("click", () => openDrawer(row));
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDrawer(row); } });
    return tr;
  }));
}

// --- recent runs ------------------------------------------------------------------------------
async function loadRecent() {
  try { state.recent = (await api("/api/runs")).runs || []; } catch (err) { state.recent = []; log(`Could not list runs: ${err.message}`, true); }
  renderRecent();
}

function renderRecent() {
  $("#recent-card").classList.toggle("hidden", !state.recent.length);
  $("#recent-table thead").replaceChildren(el("tr", {}, ["Run", "Started", "Product", "Metrics", "Complexity", "Mode", "Progress", "Accuracy", ""].map((h) => el("th", { text: h }))));
  $("#recent-table tbody").replaceChildren(...state.recent.map((r) => {
    const steps = r.steps_done.filter((s) => STEPS.includes(s)).length;
    const grounded = r.sources.reference && r.sources.reference !== "off";
    const href = `${location.pathname}?run=${encodeURIComponent(r.run_id)}#${r.steps_done.includes("evaluate") ? "report" : "run"}`;
    const accuracy = r.accuracy != null ? r.accuracy : r.task_success_rate;
    return el("tr", {},
      el("td", { class: "id" }, r.run_id, r.run_id === state.runId ? el("span", { class: "muted", text: "  (open)" }) : null),
      el("td", { class: "nowrap", text: when(r.created_at) }),
      el("td", { text: r.product_id || (r.sources.product === "sample" ? "default" : fileName(r.sources.product)) }),
      el("td", {}, r.legacy_evaluation_profile ? pill(`legacy ${r.legacy_evaluation_profile}`, "neutral") : el("span", { text: `${r.selected_metrics.length}${r.custom_metrics ? ` + ${r.custom_metrics} custom` : ""}` })),
      el("td", { text: r.question_complexity || "persona" }),
      el("td", {}, pill(grounded ? "grounded" : "ungrounded", grounded ? "blue" : "neutral")),
      el("td", {}, r.steps_done.includes("evaluate") ? pill("evaluated", "ok") : el("span", { class: "muted", text: `${steps} of ${STEPS.length} steps` })),
      el("td", { class: "nowrap", text: accuracy == null ? "—" : `${pct(accuracy)} of ${r.judged}` }),
      el("td", {}, el("a", { class: "btn btn-sm", href, text: "Open" })));
  }));
}

// --- drawer -----------------------------------------------------------------------------------
function section(title, ...content) { return el("div", { class: "dsec" }, el("h3", {}, ...[].concat(title)), ...content); }
const tagChips = (tags) => (tags && tags.length ? el("span", { class: "tags" }, tags.map((t) => el("code", { class: "tag", text: t }))) : null);

function openDrawer(row) {
  const scenario = state.scenarios.find((s) => s.id === row.scenario_id);
  const question = state.questions.find((q) => q.question_id === row.question_id);
  // A question carries the cheat sheet it was generated with (success_criteria.cheat_sheet).
  const pack = (question && question.expectations.success_criteria.cheat_sheet)
    || (scenario && state.packs.find((p) => p.pack_id === scenario.reference_pack_id));
  const evaluation = state.evaluations.find((e) => e.question_id === row.question_id);
  const execution = state.executions.find((e) => e.question_id === row.question_id);
  $("#drawer-eyebrow").textContent = [row.question_id, row.scenario_id, row.mode, evaluation && evaluation.legacy_evaluation_profile ? `legacy ${evaluation.legacy_evaluation_profile}` : ""].filter(Boolean).join("  ·  ");
  $("#drawer-title").textContent = row.question || row.scenario_title;
  $("#drawer-verdict").replaceChildren(...(row.final_verdict ? [verdictPill(row.final_verdict)] : []));
  const parts = [];
  if (evaluation) parts.push(...evaluationSections(evaluation));
  if (row.question_id) {
    const structured = criteriaOf(row);
    const criteria = (structured.general || []).map((c, i) => el("span", {}, el("code", { class: "tag", text: `CRIT-${String(i + 1).padStart(2, "0")}` }), " ", c));
    parts.push(section("Question and expected behaviour", kvList([
      ["Asked", row.question], ["Expected behaviour", row.expected_behavior], ["Success criteria", criteria],
      ["Metric criteria", metricCriteria(structured)],
    ])));
  }
  if (execution && execution.turns && execution.turns.length > 2) {
    parts.push(section(["Conversation", el("span", { class: "section-note", text: `session ${execution.session_id || "—"}` })],
      el("div", { class: "turns" }, execution.turns.map((t) => el("div", { class: `turn ${t.role}` },
        el("div", { class: "turn-meta", text: `${t.role} · ${human(t.kind)}${t.latency_ms != null ? ` · ${ms(t.latency_ms)}` : ""}` }),
        el("div", { class: "turn-msg", text: t.message }),
        t.options && t.options.length ? el("div", { class: "turn-meta", text: `Options: ${t.options.join(" · ")}` }) : null)))));
  }
  if (row.product_outcome) {
    const content = [kvList([["Outcome", OUTCOME_LABELS[row.product_outcome] || row.product_outcome], ["Response kind", human(row.response_classification)],
      ["Answer", row.answer_text], ["Message", row.product_message], ["Tool calls", split(row.tool_calls)], ["Retrieved context", row.retrieved_context],
      ["Latency", row.latency_ms ? ms(Number(row.latency_ms)) : ""]])];
    for (const artifact of split(row.artifacts)) content.push(el("pre", { text: artifact }));
    const context = execution && execution.context;
    if (context && context.chunks && context.chunks.length) {
      content.push(el("details", { class: "raw" }, el("summary", { text: `Retrieved chunks (${context.chunks.length}${context.rank_exposed ? ", ranked" : ""})` }),
        el("pre", { text: context.chunks.map((c) => `${c.rank != null ? `#${c.rank} ` : ""}[${c.chunk_id}]${c.source_id ? ` (${c.source_id})` : ""} ${c.text}`).join("\n\n") })));
    }
    parts.push(section("Product response", ...content));
  }
  if (evaluation) parts.push(...detailSections(evaluation));
  parts.push(section(["Scenario", pill(row.mode, row.mode === "grounded" ? "blue" : "neutral")], kvList([
    ["Title", row.scenario_title], ["Business goal", row.business_goal],
    ["Persona task", scenario ? scenario.typical_task || "" : ""],
    ["Task / capability", scenario ? `${human(scenario.task_type)} · ${human(scenario.capability)}` : ""],
    ["Rationale", scenario ? scenario.capability_rationale : ""], ["Question intent", scenario ? scenario.question_intent : ""],
    ["Criteria guidance", scenario ? scenario.success_criteria : []],
  ])));
  if (pack) parts.push(packSection(pack));
  showDrawer(parts);
}

function showDrawer(parts) {
  $("#drawer-body").replaceChildren(...parts);
  $("#drawer-body").scrollTop = 0;
  $("#drawer").classList.remove("hidden"); $("#scrim").classList.remove("hidden");
  $("#drawer-close").focus();
}

function metricDetails(m) {
  const d = m.details || {};
  const out = [];
  if (d.factual_verification) out.push(`Factual verification: ${human(d.factual_verification)}${d.limitation ? ` - ${d.limitation}` : ""}`);
  if (d.reciprocal_rank != null) out.push(`Reciprocal rank ${d.reciprocal_rank}${d.first_relevant_rank ? ` (first relevant chunk at rank ${d.first_relevant_rank})` : " (no relevant chunk)"}`);
  if (d.required_information) out.push(...d.required_information.map((i) => `Required: ${i.item} - ${human(i.status)}${i.chunk_ids.length ? ` [${i.chunk_ids.join(", ")}]` : ""}`));
  if (d.disclosure) out.push(d.disclosure);
  if (d.citation_checks) out.push(...d.citation_checks.map((c) => `Citation ${c.citation}${c.chunk_id ? ` -> ${c.chunk_id}` : ""}: ${human(c.support)}`));
  if (m.missing_requirement && m.status !== "scored") out.push(`Missing: ${human(m.missing_requirement)}`);
  return out;
}

function evaluationSections(ev) {
  const names = activeMetrics().filter((m) => ev.metrics[m]).concat(Object.keys(ev.metrics).filter((m) => !activeMetrics().includes(m)));
  if (!names.length) {
    return [section(["Evaluation", verdictPill(ev.verdict || "recorded_only")],
      ev.verdict ? callout("warn", "No scores.", "No selected metric produced a result; see the notes.")
        : callout("info", "Not judged.", `The product ${OUTCOME_LABELS[ev.product_outcome] || ev.product_outcome}. Unresolved clarifications, errors and timeouts are recorded but not scored.`),
      ev.notes.length ? kvList([["Notes", ev.notes]]) : null)];
  }
  const minis = el("div", { class: "mini-metrics" }, names.map((name) => {
    const m = ev.metrics[name];
    return el("div", { class: "mini" }, el("b", { text: metricLabel(name) }), scoreChip(m.score == null ? "" : String(m.score), true, m.status));
  }));
  const why = el("div", { class: "why" });
  for (const name of names) {
    const m = ev.metrics[name];
    const details = metricDetails(m);
    why.append(el("div", { class: "why-row" },
      el("div", { class: "why-head" }, el("b", { text: metricLabel(name) }),
        m.status === "scored" ? scoreChip(String(m.score), true) : pill(m.status, "neutral"), m.judge_group ? el("small", { class: "muted", text: ` ${human(m.judge_group)} judge` }) : null),
      el("div", { text: m.rationale }), m.evidence ? el("span", { class: "quote", text: m.evidence }) : null,
      (m.what_worked || []).length ? el("ul", { class: "worked" }, m.what_worked.map((x) => el("li", { text: x }))) : null,
      (m.what_didnt || []).length ? el("ul", { class: "didnt" }, m.what_didnt.map((x) => el("li", { text: x }))) : null,
      details.length ? list(details) : null));
  }
  return [section(["Evaluation", verdictPill(ev.verdict || "recorded_only")], minis,
    el("h4", { text: "Why these scores" }), why,
    kvList([["Product outcome", human(ev.product_outcome)], ["Response kind", ev.response_classification ? `${human(ev.response_classification)} (${human(ev.classification_source)})` : ""],
      ["Judge's own opinion", ev.llm_verdict ? verdictPill(ev.llm_verdict) : ""], ["Verdict policy", ev.verdict_policy], ["Notes", ev.notes]]))];
}

function detailSections(ev) {
  const out = [];
  if (ev.criteria_results.length) {
    const table = el("table", { class: "dtable" });
    for (const c of ev.criteria_results) {
      table.append(el("tr", {}, el("td", { class: "fit" }, c.tag ? el("code", { class: "tag", text: c.tag }) : null, " ", pill(c.status, c.status)),
        el("td", {}, el("div", { text: c.criterion }), el("div", { class: "muted", text: c.rationale }), c.evidence ? el("span", { class: "quote", text: c.evidence }) : null)));
    }
    out.push(section(["Success criteria", el("span", { class: "section-note", text: "shown for traceability; each feeds its matching metric" })], table));
  }
  if (ev.findings.length) {
    out.push(section([`Findings (${ev.findings.length})`, el("span", { class: "section-note", text: "verdict reason" })], ...ev.findings.map((f) => el("div", { class: `finding sev-${f.severity}` },
      el("div", { class: "finding-meta" }, pill(f.severity, f.severity), el("span", { text: `${f.finding_id} · ${metricLabel(f.category)} · evidence ${f.evidence_status}` }), f.matches_known_gap ? pill("known gap", "neutral") : null),
      el("div", { text: f.description }),
      el("div", { class: "exp" }, el("b", { text: "Expected: " }), f.expected_behavior),
      el("div", { class: "exp" }, el("b", { text: "Observed: " }), f.observed_behavior),
      f.evidence ? el("span", { class: "quote", text: f.evidence }) : null,
      tagChips(f.related_tags)))));
  }
  for (const bug of ev.bug_candidates) {
    out.push(section(["Bug candidate", pill(bug.status, "fail")], kvList([
      ["Id", bug.bug_id], ["Title", bug.title], ["Description", bug.description], ["Reproduce with", bug.reproduction_question],
      ["Product", bug.product_id], ["Findings", bug.finding_ids.join(", ")],
    ])));
  }
  return out;
}

function packDraft(pack) {
  if (pack.kind === "document_collection") return { documents: pack.documents.map((d) => ({ name: d.name, why: d.why })) };
  return {
    tables: pack.tables.map((t) => ({ table: t.name, columns: t.columns.map((c) => c.name), why: t.why })),
    joins: pack.joins.map((j) => ({ from: j.from, to: j.to })),
    rules: pack.rules.map((r) => r.id),
  };
}

function columnLine(c) {
  return `${c.name}${c.type ? ` (${c.type})` : ""}${c.description ? `: ${c.description}` : ""}${c.values && c.values.length ? ` [values: ${c.values.join(", ")}]` : ""}`;
}

function packSection(pack) {
  const content = [el("p", { class: "muted", text: "The answer key for this scenario, copied from your reference. Question generation and evaluation see this, not the whole reference." })];
  if (pack.kind === "document_collection") {
    content.push(...pack.documents.map((d) => el("div", { class: "pack-item" }, el("div", { class: "name", text: d.name }), el("div", { class: "muted", text: `Why: ${d.why}` }), el("span", { class: "quote", text: d.summary }))));
  } else {
    content.push(...pack.tables.map((t, i) => el("div", { class: "pack-item" },
      el("div", { class: "name" }, el("code", { class: "tag", text: `TABLE-${String(i + 1).padStart(2, "0")}` }), " ", t.name),
      el("div", { text: t.description || "(no description)" }), el("div", { class: "muted", text: `Why: ${t.why}` }),
      t.rules.length ? list(t.rules.map((r) => `Rule: ${r}`)) : null,
      t.columns.length ? el("span", { class: "quote", text: t.columns.map(columnLine).join("\n") }) : null)));
    if (pack.joins.length) content.push(el("h4", { text: "Joins" }), list(pack.joins.map((j, i) => `JOIN-${String(i + 1).padStart(2, "0")}  ${j.from} -> ${j.to} (${j.verified ? "verified" : "NOT verified"})`)));
    if (pack.rules.length) content.push(el("h4", { text: "Rules" }), list(pack.rules.map((r) => `RULE-${r.id}: ${r.rule}`)));
  }
  if (pack.rejected_items.length) content.push(el("h4", { text: "Dropped (not in the reference)" }), list(pack.rejected_items.map((d) => d.reason)));
  const editor = el("textarea", { rows: "14", class: "hidden", spellcheck: "false", "aria-label": "Cheat sheet picks (JSON)" });
  editor.value = JSON.stringify(packDraft(pack), null, 2);
  const save = el("button", { class: "btn btn-sm btn-primary hidden", type: "button", text: "Save cheat sheet" });
  const edit = el("button", { class: "btn btn-sm", type: "button", text: "Edit cheat sheet" });
  edit.addEventListener("click", () => { editor.classList.toggle("hidden"); save.classList.toggle("hidden"); edit.textContent = editor.classList.contains("hidden") ? "Edit cheat sheet" : "Cancel"; });
  save.addEventListener("click", () => withBusy(`Saving ${pack.pack_id}`, async () => {
    let draft;
    try { draft = JSON.parse(editor.value); } catch (err) { throw new Error(`Not valid JSON: ${err.message}`); }
    const body = await putJson(`/api/runs/${state.runId}/reference_packs/${pack.pack_id}`, draft);
    for (const later of STEPS.slice(1)) $(".step-result", $(`.step[data-step="${later}"]`)).textContent = "";
    applyRun(body);
    log(`${pack.pack_id} saved - questions, execute and evaluate must be run again`);
    toast(`${pack.pack_id} saved. Run questions, ask and evaluate again.`);
    closeDrawer();
  }));
  content.push(el("div", { class: "row-actions" }, edit, save), editor);
  return section(["Cheat sheet", el("span", { class: `pill pill-${pack.edited ? "warn" : "neutral"}`, text: pack.edited ? "edited" : pack.pack_id })], ...content);
}

function closeDrawer() { $("#drawer").classList.add("hidden"); $("#scrim").classList.add("hidden"); }

// --- boot -------------------------------------------------------------------------------------
async function boot() {
  try {
    state.config = await api("/api/config");
    $("#env").textContent = `${state.config.llm_provider} · ${state.config.llm_model} · adapter ${state.config.product_adapter}`;
    $("#env").title = `Judge / generator LLM: ${state.config.llm_provider} ${state.config.llm_model}. Default product adapter: ${state.config.product_adapter}.`;
    $("#scenario-count").value = state.config.scenario_count;
    $("#question-count").value = state.config.question_count;
    $("#scenario-count").max = String(maxStep());
    $("#question-count").max = String(maxStep());
    state.selectedMetrics = new Set(state.config.default_metrics || []);
    state.samples = await api("/api/samples");
  } catch (err) {
    log(`Could not load configuration: ${err.message}`, true);
    toast(`Could not load configuration: ${err.message}`, "error");
  }
  for (const box of $$("[data-preview-for]")) {
    const empty = $(".preview-empty", box);
    if (empty) box.dataset.empty = empty.textContent;
  }
  for (const kind of ["persona", "product", "reference", "reference_guide", "custom_metrics"]) for (const zone of $$("[data-drop]", panel(kind))) wireDrop(zone, (file) => onFile(kind, file));
  for (const b of $$("#persona-source button")) b.addEventListener("click", () => setPersonaSource(b.dataset.source));
  for (const b of $$("#instructions-source button")) b.addEventListener("click", () => setInstructionsSource(b.dataset.source));
  $("#use-reference").addEventListener("change", (e) => setUseReference(e.target.checked));
  $("#product-clear").addEventListener("click", () => clearUpload("product"));
  $("#reference-clear").addEventListener("click", () => clearUpload("reference"));
  $("#custom-clear").addEventListener("click", () => {
    if (!state.uploads.custom_metrics) state.removeSavedWorkbook = true;
    clearUpload("custom_metrics");
  });
  $("#lock-evaluation").addEventListener("click", lockEvaluation);
  $("#unlock-evaluation").addEventListener("click", unlockEvaluation);
  $("#custom-metrics-text").addEventListener("input", () => { renderEvaluation(); updateUi(); });
  $("#guidance-text").addEventListener("input", renderGuidance);
  $("#guidance-file").addEventListener("change", renderGuidance);
  $("#save-guidance").addEventListener("click", () => saveGuidance(false));
  $("#remove-guidance").addEventListener("click", () => saveGuidance(true));
  for (const b of $$("#context-source button")) b.addEventListener("click", () => setContextSource(b.dataset.source));
  setContextSource("response_field");
  $("#question-complexity").addEventListener("change", (e) => { state.questionComplexity = e.target.value; });
  $("#create-run").addEventListener("click", createRun);
  wireDrop($("[data-contract-drop]"), setContractFile);
  for (const b of $$("#contract-source button")) b.addEventListener("click", () => setContractSource(b.dataset.source));
  $("#product-url").addEventListener("input", updateUi);
  $("#connect").addEventListener("click", connect);
  $("#disconnect").addEventListener("click", disconnect);
  $("#run-all").addEventListener("click", runAll);
  for (const node of $$(".step")) $(".step-btn", node).addEventListener("click", () => withBusy(`Running ${node.dataset.step}`, () => runStep(node.dataset.step)));
  for (const b of $$("#filter button")) b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    for (const x of $$("#filter button")) x.classList.toggle("on", x === b);
    renderTable();
  });
  for (const b of $$("#breakdown-by button")) b.addEventListener("click", () => {
    state.breakdownBy = b.dataset.by;
    for (const x of $$("#breakdown-by button")) x.classList.toggle("on", x === b);
    renderBreakdown();
  });
  $("#search").addEventListener("input", (e) => { state.search = e.target.value; renderTable(); });
  $("#all-columns").addEventListener("change", (e) => { state.allColumns = e.target.checked; renderTable(); });
  $("#print").addEventListener("click", () => window.print());
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
  $("#new-run").href = `${location.pathname}#setup`;
  setPersonaSource(sample("persona") ? "sample" : "upload");
  setUseReference(Boolean(sample("reference")));
  setInstructionsSource("none");
  renderInput("product");
  renderInput("custom_metrics");

  const fromUrl = new URLSearchParams(location.search).get("run");
  let initial = location.hash.slice(1);
  if (fromUrl) {
    state.runId = fromUrl;
    try {
      applyRun(await api(`/api/runs/${encodeURIComponent(fromUrl)}`));
      $("#persona-queries").value = state.run.persona_queries.join(String.fromCharCode(10));
      loadEvaluationDraft();
      log(`Reopened run ${fromUrl}`);
      if (!VIEWS.includes(initial)) initial = done().has("evaluate") ? "report" : "run";
    } catch (err) {
      state.runId = null;
      log(`Could not reopen ${fromUrl}: ${err.message}`, true);
      toast(`Could not reopen ${fromUrl}: ${err.message}`, "error");
    }
  }
  if (!state.run) renderReport({});
  renderEvaluation();
  updateUi();
  showView(initial, { scroll: false });
  loadRecent();
}

boot();
