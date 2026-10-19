"use strict";
// Synthetic User web UI (D199, D201): workflow pages + test report over the JSON API.
// All data is inserted with textContent / DOM nodes (never innerHTML): inputs are untrusted.

const STEPS = ["scenarios", "questions", "execute", "evaluate"];
const VIEWS = ["setup", "connect", "run", "report"];
const METRICS = ["relevancy", "correctness", "groundedness", "hallucination_rate", "completeness"];
const METRIC_LABELS = {
  relevancy: "Answer relevancy", correctness: "Correctness", groundedness: "Groundedness",
  hallucination_rate: "Hallucination rate", completeness: "Completeness",
};
const METRIC_SHORT = {
  relevancy: "Relevancy", correctness: "Correctness", groundedness: "Groundedness",
  hallucination_rate: "Hallucination", completeness: "Completeness",
};
const METRIC_HELP = {
  relevancy: "Does the answer address the question that was asked?",
  correctness: "Are the key facts right according to the reference data?",
  groundedness: "Share of the answer's statements backed by the reference data.",
  hallucination_rate: "Share of statements that contradict the reference or are invented.",
  completeness: "Share of the question's success criteria the answer meets.",
};
const NEEDS_REFERENCE = new Set(["correctness", "groundedness"]);
const SEVERITY_RANK = { low: 1, medium: 2, high: 3, critical: 4 };
const OUTCOME_LABELS = { answered: "answered", clarification: "asked to clarify", denied: "declined", error: "error", timeout: "timed out" };
const SOURCE_LABELS = { persona: "Persona", product: "Product description", reference: "Reference", reference_guide: "Usage instructions" };

const state = {
  config: null, samplePersona: null, personaSource: "sample", useReference: false, instructionsSource: "none", uploads: {},
  runId: null, run: null, rows: [], scenarios: [], packs: [], evaluations: [], kpis: null,
  filter: "all", search: "", allColumns: false, breakdownBy: "by_task_type", busy: false,
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
const list = (items) => (items && items.length ? el("ul", {}, items.map((i) => el("li", { text: i }))) : el("span", { class: "muted", text: "—" }));
// CSV cells list several items as "• item" lines (items may span lines, e.g. SQL).
const split = (value) => {
  const text = (value || "").trim();
  if (!text) return [];
  if (!text.startsWith("• ")) return [text];
  return text.slice(2).split(/\n• /).map((item) => item.trim()).filter(Boolean);
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
  finally { state.busy = false; $("#busy-line").classList.remove("on"); updateUi(); }
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
function previewOf(kind, d, raw) {
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
      ? `${d.tables} table(s), ${d.relationships} relationship(s), ${d.rules} rule(s)`
      : `${d.documents} document(s)`;
    return kvList([["Format", human(d.kind)], ["Name", d.name], ["Contents", counts], ["Reference id", d.reference_id], ["Size", `${d.chars.toLocaleString()} characters`]]);
  }
  return kvList([["Instructions", d.name], ["Size", `${d.chars.toLocaleString()} characters`]]);
}

function panel(kind) { return $(`.input[data-kind="${kind}"]`); }
function currentInput(kind) {
  if (kind === "persona" && state.personaSource === "sample") return state.samplePersona ? { preview: state.samplePersona, label: "Sample persona" } : null;
  if ((kind === "reference" || kind === "reference_guide") && !state.useReference) return null;
  if (kind === "reference_guide" && state.instructionsSource !== "upload") return null;
  const upload = state.uploads[kind];
  return upload ? { preview: upload.preview, label: upload.file.name } : null;
}

function renderInput(kind) {
  const root = panel(kind);
  const input = currentInput(kind);
  const box = $(".preview", root);
  const source = $(".source", root);
  box.replaceChildren();
  source.className = input ? "source ok" : "source";
  source.replaceChildren(...(input ? [icon("check"), input.label] : []));
  if (input) {
    const rawLabel = kind === "persona" || kind === "product" ? "View YAML" : "View file";
    box.append(previewOf(kind, input.preview.parsed, input.preview.raw), el("details", { class: "raw" }, el("summary", { text: rawLabel }), el("pre", { text: input.preview.raw })));
  }
  if (kind === "product") {
    const status = $("[data-status-for=product]");
    status.className = `pill ${input ? "pill-ok" : "pill-req"}`;
    status.textContent = input ? "Ready" : "Required";
  }
  updateReadiness();
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
    $(".preview", root).before(el("div", { class: "callout callout-error" }, icon("warn"), el("div", { text: `${file.name} was not accepted: ${err.message}` })));
  }
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
  const ref = state.uploads.reference;
  const kind = ref ? ref.preview.parsed.kind : "data_catalog";
  $("#instructions-text").placeholder = defaults[kind] || "";
}

function readiness() {
  const persona = Boolean(currentInput("persona"));
  const product = Boolean(currentInput("product"));
  const reference = !state.useReference || Boolean(currentInput("reference"));
  return { persona, product, reference, ok: persona && product && reference };
}

function updateReadiness() {
  const r = readiness();
  $("#chk-persona").className = r.persona ? "ok" : "";
  $("#chk-product").className = r.product ? "ok" : "";
  $("#chk-reference").className = r.reference ? "ok" : "";
  $("#chk-reference").textContent = state.useReference ? "Reference" : "Reference (off)";
  $("#start-hint").textContent = r.ok
    ? (state.runId ? "Starting again creates a new run." : "Ready to start.")
    : state.runId ? "To start a new run, complete the inputs above." : "Complete the required inputs to start.";
  $("#create-run").disabled = !r.ok || state.busy;
}

async function createRun() {
  const form = new FormData();
  if (state.personaSource === "upload" && state.uploads.persona) form.append("persona", state.uploads.persona.file);
  form.append("product", state.uploads.product.file);
  if (state.useReference) {
    form.append("reference", state.uploads.reference.file);
    if (state.instructionsSource === "upload" && state.uploads.reference_guide) form.append("reference_guide", state.uploads.reference_guide.file);
    if (state.instructionsSource === "write") form.append("reference_instructions", $("#instructions-text").value);
  }
  form.append("use_reference", state.useReference ? "true" : "false");
  form.append("persona_queries", $("#persona-queries").value);
  await withBusy("Starting run", async () => {
    const body = await api("/api/runs", { method: "POST", body: form });
    state.runId = body.state.run_id;
    history.replaceState(null, "", `${location.pathname}?run=${encodeURIComponent(state.runId)}#connect`);
    for (const node of $$(".step")) { node.classList.remove("done", "error", "running"); $(".step-result", node).textContent = ""; }
    applyRun(body);
    log(`Run ${state.runId} started (${state.useReference ? "grounded" : "ungrounded"})`);
    toast(`Run ${state.runId} started`);
    showView("connect");
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
    box.append(
      el("div", { class: "facts" },
        fact("Endpoint", `${c.method} ${c.product_url}`), fact("Contract", c.contract_file),
        fact("Headers", c.header_names.length ? `${c.header_names.join(", ")} (values hidden)` : "none"),
        fact("Recognised outcomes", outcomes.join(", "))),
      el("details", { class: "raw" }, el("summary", { text: "View request template and response mapping" }),
        el("pre", { text: JSON.stringify({ timeout: c.timeout, request_template: c.request_template, response_mapping: mapping }, null, 2) })),
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
    const ready = Boolean(state.runId) && (i === 0 || d.has(STEPS[i - 1]));
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
  $("#run-all").disabled = !state.runId || state.busy || completed === STEPS.length;
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
    ? "Business goals plus a reference pack per scenario, quoted from your reference data."
    : "Business goals this persona would pursue with the product.";
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
    for (const [kind, label] of Object.entries(SOURCE_LABELS)) {
      const value = state.run.sources[kind];
      sources.append(el("dt", { text: label }), el("dd", { text: value === "sample" ? "Sample" : value === "off" ? "Off" : fileName(value) || "—" }));
    }
    sources.append(el("dt", { text: "Answers from" }), el("dd", { text: connection ? connection.product_url : mock ? "Mock product" : `Configured adapter (${adapter})` }));
  }
  $("#crumb").replaceChildren(...(state.runId ? [el("span", { class: "sep", text: "/" }), el("code", { text: state.runId })] : []));

  // navigation status
  const k = state.kpis;
  const navStatus = {
    setup: state.runId ? "Run created" : readiness().ok ? "Ready to start" : "Persona, product, reference",
    connect: connection ? "Connected" : mock ? "Mock product" : "Not connected",
    run: `${completed} of ${STEPS.length} steps`,
    report: k ? `Task success ${pct(k.task_success_rate)}` : "Not available yet",
  };
  const navDone = { setup: Boolean(state.runId), connect: Boolean(connection || mock), run: completed === STEPS.length, report: Boolean(k) };
  for (const node of $$("[data-nav]")) {
    $("[data-nav-status]", node).textContent = navStatus[node.dataset.nav];
    node.classList.toggle("done", navDone[node.dataset.nav]);
  }
}

function renderRunCallout(connection, mock, d) {
  const box = $("#run-callout");
  const go = (view, text) => { const a = el("a", { class: "btn btn-sm", href: `#${view}`, text }); return a; };
  let node;
  if (!state.runId) node = callout("info", "No run yet.", "Complete the test setup and start a run first.", go("setup", "Go to test setup"));
  else if (connection) node = callout("info", "Connected.", `Questions go to ${connection.product_url}.`);
  else if (mock) node = callout("warn", "Mock product (demo).", "Answers come from an AI stand-in, not from the real product.", go("connect", "Change"));
  else if (!d.has("execute")) node = callout("warn", "No product connected.", `Questions will go to the configured "${adapterName()}" adapter (test data), not to your product.`, go("connect", "Connect product"));
  else node = callout("warn", "No product connected.", `The answers in this run came from the configured "${adapterName()}" adapter (test data), not from your product. Connect the product and run "Ask the product" again.`, go("connect", "Connect product"));
  box.replaceChildren(...(node ? [node] : []));
}

function describe(step, detail) {
  if (step === "scenarios") return `${detail.accepted} accepted, ${detail.rejected} rejected${detail.reference_packs ? ` · ${detail.reference_packs} reference pack(s)` : ""}`;
  if (step === "questions") return `${detail.accepted} questions, ${detail.rejected} rejected`;
  if (step === "execute") return `${detail.executions} answers via ${detail.product}`;
  return `${detail.evaluations} evaluated · ${detail.bug_candidates} bug candidate(s)${detail.task_success_rate != null ? ` · task success ${pct(detail.task_success_rate)}` : ""}`;
}

async function runStep(step) {
  const node = $(`.step[data-step="${step}"]`);
  node.classList.remove("done", "error", "ready"); node.classList.add("running");
  $(".step-status", node).replaceChildren(el("span", { class: "spinner" }), "Running");
  $(".step-result", node).textContent = "";
  const count = step === "scenarios" ? $("#scenario-count").value : step === "questions" ? $("#question-count").value : "";
  const started = performance.now();
  try {
    const body = await postJson(`/api/runs/${state.runId}/steps/${step}`, count ? { count: Number(count) } : {});
    node.classList.remove("running");
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    $(".step-result", node).textContent = `${describe(step, body.detail)} · ${secs}s`;
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

// --- report -----------------------------------------------------------------------------------
function applyRun(body) {
  state.run = body.state;
  state.rows = body.rows || [];
  state.scenarios = body.scenarios || [];
  state.packs = body.reference_packs || [];
  state.kpis = body.kpis || null;
  state.evaluations = body.evaluations || [];
  renderConnection();
  renderReport(body);
  updateUi();
}

function questionRows() { return state.rows.filter((r) => r.question_id); }
function verdictCounts() {
  const rows = questionRows();
  const by = (v) => rows.filter((r) => r.final_verdict === v).length;
  return { pass: by("pass"), fail: by("fail"), inconclusive: by("inconclusive"), recorded: by("recorded_only") };
}

function renderReport(body) {
  renderMeta();
  renderNotice();
  renderSummary();
  renderMetrics();
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
  if (k) parts.push(k.product_id, k.mode === "grounded" ? "Grounded (with reference data)" : "Ungrounded (no reference data)", `${k.questions_asked} questions`, `Evaluated ${when(k.computed_at)}`);
  else if (run) parts.push(fileName(run.sources.product), `Started ${when(run.created_at)}`, "Results fill in as the steps complete.");
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
    const sample = questionRows().find((r) => r.product_outcome === "error" && r.product_message);
    node = callout("warn", "No answers could be judged.",
      el("div", {}, `Of ${k.questions_asked} question(s): ${outcomes}. Only answered questions are scored; clarifications, refusals, errors and timeouts are recorded only.`,
        sample ? el("div", { class: "quote", text: `Product message: ${sample.product_message}` }) : null),
      go("connect", "Check connection"));
  } else {
    const failed = (k.outcomes.error || 0) + (k.outcomes.timeout || 0);
    if (failed) node = callout("warn", `${failed} of ${k.questions_asked} question(s) got no answer.`, "Errors and timeouts lower availability and are not scored.");
  }
  box.replaceChildren(...(node ? [node] : []));
}

function tile(label, ...content) { return el("div", { class: "tile" }, el("div", { class: "tile-label", text: label }), ...content); }

function renderSummary() {
  const k = state.kpis;
  const c = verdictCounts();
  const total = c.pass + c.fail + c.inconclusive + c.recorded;
  const judged = c.pass + c.fail + c.inconclusive;
  const rate = k ? k.task_success_rate : null;
  const ring = el("div", { class: `ring${rate == null ? " empty" : ""}`, style: `--p:${rate == null ? 0 : Math.round(rate * 100)}` }, el("span", { text: pct(rate) }));
  const success = tile("Task success", el("div", { class: "tile-row" }, ring,
    el("div", {}, el("div", { class: "tile-value small", text: k ? `${c.pass} of ${judged}` : "—" }),
      el("div", { class: "tile-sub", text: k ? "judged answers passed" : "available after evaluation" }))));

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
  $("#summary").replaceChildren(success, verdicts, health, bugs);
}

function renderMetrics() {
  const k = state.kpis;
  $("#metric-cards").replaceChildren(...METRICS.map((name) => {
    const m = k && k.metrics.find((x) => x.metric === name);
    const lower = name === "hallucination_rate";
    const head = el("div", { class: "metric-name", text: METRIC_LABELS[name] });
    const desc = el("div", { class: "metric-desc", text: lower ? `${METRIC_HELP[name]} Lower is better.` : METRIC_HELP[name] });
    if (!m || m.mean == null) {
      const why = !k ? "Available after evaluation"
        : k.mode === "ungrounded" && NEEDS_REFERENCE.has(name) ? "Needs reference data (ungrounded run)"
          : m && m.inconclusive ? `${m.inconclusive} inconclusive` : "No judged answers";
      return el("div", { class: "metric na" }, head, desc, el("div", { class: "metric-value", text: "—" }), el("div", { class: "meter" }), el("div", { class: "metric-foot" }, el("span", { text: why })));
    }
    const meterCls = lower ? (m.mean > 0 ? "bad" : "good") : "";
    const extras = [m.inconclusive ? `${m.inconclusive} inconclusive` : "", m.not_applicable ? `${m.not_applicable} n/a` : ""].filter(Boolean).join(" · ");
    return el("div", { class: "metric" }, head, desc,
      el("div", { class: "metric-value" }, pct(m.mean), el("small", { text: `mean ${m.mean.toFixed(2)}` })),
      el("div", { class: `meter ${meterCls}` }, el("div", { style: `width:${Math.max(m.mean * 100, m.mean > 0 ? 2 : 0)}%` })),
      el("div", { class: "metric-foot" },
        el("span", { text: `${pct(m.pass_rate)} ${lower ? "hallucination-free" : "at full score"}` }),
        el("span", { text: `n = ${m.n}${extras ? ` · ${extras}` : ""}` })));
  }));
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

function scoreChip(name, value, judged) {
  if (!value) return el("span", { class: "score na", text: judged ? "n/a" : "" });
  if (value === "inconclusive") return el("span", { class: "score mid", title: "inconclusive", text: "?" });
  const v = Number(value);
  const cls = name === "hallucination_rate" ? (v === 0 ? "good" : "bad") : v >= 1 ? "good" : v <= 0 ? "bad" : "mid";
  return el("span", { class: `score ${cls}`, text: Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/0$/, "") });
}

function renderBreakdown() {
  const k = state.kpis;
  const groups = k ? k[state.breakdownBy] || [] : [];
  $("#breakdown-block").classList.toggle("hidden", !k || (!k.by_task_type.length && !k.by_capability.length));
  const head = el("tr", {}, el("th", { text: state.breakdownBy === "by_task_type" ? "Task type" : "Capability" }), el("th", { class: "num", text: "Questions" }),
    el("th", { class: "num", text: "Judged" }), el("th", { class: "num", text: "Pass rate" }), METRICS.map((m) => el("th", { class: "num", text: METRIC_SHORT[m] })));
  $("#breakdown-table thead").replaceChildren(head);
  $("#breakdown-table tbody").replaceChildren(...(groups.length ? groups.map((g) => el("tr", {},
    el("td", { text: human(g.group) }), el("td", { class: "num", text: String(g.questions) }), el("td", { class: "num", text: String(g.judged) }),
    el("td", { class: "num", text: pct(g.pass_rate) }),
    METRICS.map((m) => el("td", { class: "num" }, scoreChip(m, g.metric_means[m] == null ? "" : String(Math.round(g.metric_means[m] * 100) / 100), g.judged > 0)))))
    : [el("tr", {}, el("td", { class: "empty", colspan: "9", text: "No data." }))]));
}

const MAIN_COLUMNS = [
  ["question_id", "ID"], ["question", "Question"], ["product_outcome", "Outcome"], ["final_verdict", "Verdict"],
  ...METRICS.map((m) => [m, METRIC_SHORT[m]]), ["bug_ids", "Bug"],
];

function matchesFilter(r) {
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
}

function renderTable() {
  renderFilterCounts();
  const all = state.allColumns && state.rows.length;
  const columns = all ? Object.keys(state.rows[0]).map((k) => [k, human(k)]) : MAIN_COLUMNS;
  const rows = filteredRows();
  $("#results-table thead").replaceChildren(el("tr", {}, columns.map(([key, label]) => el("th", { class: METRICS.includes(key) && !all ? "num" : null, text: label }))));
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
      if (key === "final_verdict") return el("td", { class: "nowrap" }, value ? verdictPill(value) : "");
      if (key === "product_outcome") return el("td", { class: "nowrap" }, value ? el("span", { class: `outcome ${value}`, text: OUTCOME_LABELS[value] || value }) : "");
      if (key === "question_id" || key === "bug_ids") return el("td", { class: "id", text: split(value).join(", ") });
      if (METRICS.includes(key)) return el("td", { class: "num" }, scoreChip(key, value, judged));
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
  $("#recent-table thead").replaceChildren(el("tr", {}, ["Run", "Started", "Product", "Mode", "Progress", "Task success", ""].map((h) => el("th", { text: h }))));
  $("#recent-table tbody").replaceChildren(...state.recent.map((r) => {
    const steps = r.steps_done.filter((s) => STEPS.includes(s)).length;
    const grounded = r.sources.reference && r.sources.reference !== "off";
    const href = `${location.pathname}?run=${encodeURIComponent(r.run_id)}#${r.steps_done.includes("evaluate") ? "report" : "run"}`;
    return el("tr", {},
      el("td", { class: "id" }, r.run_id, r.run_id === state.runId ? el("span", { class: "muted", text: "  (open)" }) : null),
      el("td", { class: "nowrap", text: when(r.created_at) }),
      el("td", { text: r.product_id || fileName(r.sources.product) }),
      el("td", {}, pill(grounded ? "grounded" : "ungrounded", grounded ? "blue" : "neutral")),
      el("td", {}, r.steps_done.includes("evaluate") ? pill("evaluated", "ok") : el("span", { class: "muted", text: `${steps} of ${STEPS.length} steps` })),
      el("td", { class: "nowrap", text: r.task_success_rate == null ? "—" : `${pct(r.task_success_rate)} of ${r.judged}` }),
      el("td", {}, el("a", { class: "btn btn-sm", href, text: "Open" })));
  }));
}

// --- drawer -----------------------------------------------------------------------------------
function section(title, ...content) { return el("div", { class: "dsec" }, el("h3", {}, ...[].concat(title)), ...content); }

function openDrawer(row) {
  const scenario = state.scenarios.find((s) => s.id === row.scenario_id);
  const pack = scenario && state.packs.find((p) => p.pack_id === scenario.reference_pack_id);
  const evaluation = state.evaluations.find((e) => e.question_id === row.question_id);
  $("#drawer-eyebrow").textContent = [row.question_id, row.scenario_id, row.mode].filter(Boolean).join("  ·  ");
  $("#drawer-title").textContent = row.question || row.scenario_title;
  $("#drawer-verdict").replaceChildren(...(row.final_verdict ? [verdictPill(row.final_verdict)] : []));
  const parts = [];
  if (evaluation) parts.push(...evaluationSections(evaluation));
  if (row.question_id) {
    parts.push(section("Question and expectations", kvList([
      ["Asked", row.question], ["Expected behaviour", row.expected_behavior],
      ["Success criteria", split(row.success_criteria)],
    ])));
  }
  if (row.product_outcome) {
    const content = [kvList([["Outcome", OUTCOME_LABELS[row.product_outcome] || row.product_outcome], ["Answer", row.answer_text], ["Message", row.product_message], ["Latency", row.latency_ms ? ms(Number(row.latency_ms)) : ""]])];
    for (const artifact of split(row.artifacts)) content.push(el("pre", { text: artifact }));
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

function evaluationSections(ev) {
  if (!ev.metrics) {
    return [section(["Evaluation", verdictPill("recorded_only")],
      callout("info", "Not judged.", `The product ${OUTCOME_LABELS[ev.product_outcome] || ev.product_outcome}. Clarifications, refusals, errors and timeouts are recorded but not scored.`),
      ev.notes.length ? kvList([["Notes", ev.notes]]) : null)];
  }
  const minis = el("div", { class: "mini-metrics" }, METRICS.map((name) => {
    const m = ev.metrics[name];
    return el("div", { class: "mini" }, el("b", { text: METRIC_SHORT[name] }),
      m.status === "scored" ? scoreChip(name, String(m.score), true) : el("span", { class: "na", text: human(m.status) }));
  }));
  const why = el("table", { class: "dtable" });
  for (const name of METRICS) {
    const m = ev.metrics[name];
    why.append(el("tr", {}, el("td", { class: "fit" }, el("b", { text: METRIC_LABELS[name] })),
      el("td", {}, m.rationale, m.evidence ? el("span", { class: "quote", text: m.evidence }) : null)));
  }
  return [section(["Evaluation", verdictPill(ev.verdict)], minis,
    el("h4", { text: "Why these scores" }), why,
    kvList([["Judge's own opinion", ev.llm_verdict ? verdictPill(ev.llm_verdict) : ""], ["Notes", ev.notes]]))];
}

function detailSections(ev) {
  const out = [];
  if (ev.criteria_results.length) {
    const table = el("table", { class: "dtable" });
    for (const c of ev.criteria_results) {
      table.append(el("tr", {}, el("td", { class: "fit" }, pill(c.status, c.status)),
        el("td", {}, el("div", { text: c.criterion }), el("div", { class: "muted", text: c.rationale }), c.evidence ? el("span", { class: "quote", text: c.evidence }) : null)));
    }
    out.push(section(["Success criteria", el("span", { class: "section-note", text: "basis of completeness" })], table));
  }
  if (ev.claims.length) {
    const table = el("table", { class: "dtable" });
    for (const c of ev.claims) {
      table.append(el("tr", {}, el("td", { class: "fit" }, pill(c.status, c.status)),
        el("td", {}, el("span", { class: "quote", text: c.text }), el("div", { class: "muted", text: c.reason }),
          c.reference_quote ? el("div", { class: "muted", text: `Reference: “${c.reference_quote}”` }) : null)));
    }
    out.push(section([`Statements checked (${ev.claims.length})`, el("span", { class: "section-note", text: "basis of groundedness and hallucination rate" })], table));
  }
  if (ev.findings.length) {
    out.push(section(`Findings (${ev.findings.length})`, ...ev.findings.map((f) => el("div", { class: `finding sev-${f.severity}` },
      el("div", { class: "finding-meta" }, pill(f.severity, f.severity), el("span", { text: `${f.finding_id} · ${METRIC_LABELS[f.category] || human(f.category)} · evidence ${f.evidence_status}` }), f.matches_known_gap ? pill("known gap", "neutral") : null),
      el("div", { text: f.description }),
      el("div", { class: "exp" }, el("b", { text: "Expected: " }), f.expected_behavior),
      el("div", { class: "exp" }, el("b", { text: "Observed: " }), f.observed_behavior),
      f.evidence ? el("span", { class: "quote", text: f.evidence }) : null))));
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
    content.push(...pack.tables.map((t) => el("div", { class: "pack-item" },
      el("div", { class: "name", text: t.name }), el("div", { text: t.description || "(no description)" }), el("div", { class: "muted", text: `Why: ${t.why}` }),
      t.rules.length ? list(t.rules.map((r) => `Rule: ${r}`)) : null,
      t.columns.length ? el("span", { class: "quote", text: t.columns.map(columnLine).join("\n") }) : null)));
    if (pack.joins.length) content.push(el("h4", { text: "Joins" }), list(pack.joins.map((j) => `${j.from} -> ${j.to} (${j.verified ? "verified" : "NOT verified"})`)));
    if (pack.rules.length) content.push(el("h4", { text: "Rules" }), list(pack.rules.map((r) => `${r.id}: ${r.rule}`)));
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
    const samples = await api("/api/samples");
    state.samplePersona = samples.persona;
  } catch (err) {
    log(`Could not load configuration: ${err.message}`, true);
    toast(`Could not load configuration: ${err.message}`, "error");
  }
  for (const kind of ["persona", "product", "reference", "reference_guide"]) for (const zone of $$("[data-drop]", panel(kind))) wireDrop(zone, (file) => onFile(kind, file));
  for (const b of $$("#persona-source button")) b.addEventListener("click", () => setPersonaSource(b.dataset.source));
  for (const b of $$("#instructions-source button")) b.addEventListener("click", () => setInstructionsSource(b.dataset.source));
  $("#use-reference").addEventListener("change", (e) => setUseReference(e.target.checked));
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
  setPersonaSource(state.samplePersona ? "sample" : "upload");
  setUseReference(false);
  setInstructionsSource("none");
  renderInput("product");

  const fromUrl = new URLSearchParams(location.search).get("run");
  let initial = location.hash.slice(1);
  if (fromUrl) {
    state.runId = fromUrl;
    try {
      applyRun(await api(`/api/runs/${encodeURIComponent(fromUrl)}`));
      $("#persona-queries").value = state.run.persona_queries.join(String.fromCharCode(10));
      log(`Reopened run ${fromUrl}`);
      if (!VIEWS.includes(initial)) initial = done().has("evaluate") ? "report" : "run";
    } catch (err) {
      state.runId = null;
      log(`Could not reopen ${fromUrl}: ${err.message}`, true);
      toast(`Could not reopen ${fromUrl}: ${err.message}`, "error");
    }
  }
  if (!state.run) renderReport({});
  updateUi();
  showView(initial, { scroll: false });
  loadRecent();
}

boot();
