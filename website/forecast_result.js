const account = document.querySelector("#workspaceAccount");
const resultQuestion = document.querySelector("#resultQuestion");
const primaryResult = document.querySelector("#primaryResult");
const resultContext = document.querySelector("#resultContext");
const methodComparisonMeta = document.querySelector("#methodComparisonMeta");
const methodTable = document.querySelector("#methodTable");

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayName(user) {
  if (user.email === "premium@example.test" || user.email === "free@example.test") {
    return "Will";
  }
  const name = String(user.name ?? "").trim();
  return name.split(/\s+/)[0] || "Will";
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return "n/a";
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function formatMoney(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
    style: "currency",
    currency: "USD",
  }).format(number);
}

function methodLabel(method) {
  const labels = {
    top_similarity: "Nearest retrieved market",
    unweighted_mean: "Simple average",
    median: "Median retrieved market",
    similarity_weighted_mean: "Relevance-weighted average",
    liquidity_weighted_mean: "Liquidity-weighted average",
    similarity_volume_weighted: "Relevance and volume weighted",
    top5_similarity_weighted_mean: "Top-five relevance weighted",
    min_volume_similarity_volume_weighted: "Liquid markets weighted",
    same_category_similarity_weighted: "Same-category weighted",
    same_category_top5_similarity_weighted: "Same-category top five",
    event_grouped_similarity_mean: "Event-family weighted",
  };
  if (labels[method]) return labels[method];
  return String(method || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatPercentagePointDelta(value) {
  if (!Number.isFinite(Number(value))) return "n/a";
  return `${Math.round(Math.abs(Number(value)) * 1000) / 10} percentage points`;
}

function selectedForecastSentence(probability, method) {
  if (!Number.isFinite(Number(probability))) {
    return "The selected method did not return a usable probability for this evidence set.";
  }
  return `Using ${methodLabel(method).toLowerCase()}, the retrieved evidence implies about ${formatPercent(probability)} for the target question.`;
}

async function signOut() {
  await requestJson("/api/logout", { method: "POST", body: "{}" });
  window.location.assign("/");
}

function readRun() {
  try {
    return JSON.parse(sessionStorage.getItem("forecastEvidenceRun") || "null");
  } catch {
    return null;
  }
}

function renderEmptyState() {
  resultQuestion.textContent = "No forecast result is available yet. Start with a question and review the evidence first.";
  primaryResult.innerHTML = `
    <span>No result</span>
    <strong>Start a forecast first.</strong>
    <p>The result page needs a retrieved evidence set before it can compare forecast signals.</p>
    <a class="card-action" href="/forecast">Start a forecast</a>
  `;
  resultContext.innerHTML = "";
  methodComparisonMeta.textContent = "";
  methodTable.innerHTML = "";
}

function renderResult(run) {
  const { payload, question } = run;
  const target = payload.target;
  const forecast = payload.forecast;
  const summary = target.summary;
  const forecastSummary = forecast.summary;
  const methods = forecast.aggregationMethods ?? [];
  const selected = methods.find((method) => method.method === forecastSummary.selectedMethod);
  const nearest = methods.find((method) => method.method === "top_similarity");
  const selectedProbability = forecastSummary.selectedForecastProbability;
  const nearestProbability = nearest?.forecast_probability;
  const delta = Number.isFinite(Number(selectedProbability)) && Number.isFinite(Number(nearestProbability))
    ? selectedProbability - nearestProbability
    : null;

  resultQuestion.innerHTML = `Question: <strong>${escapeHtml(question)}</strong>`;
  primaryResult.innerHTML = `
    <span>Selected forecast</span>
    <strong>${escapeHtml(formatPercent(selectedProbability))}</strong>
    <p>${escapeHtml(methodLabel(forecastSummary.selectedMethod))}</p>
    <small>${escapeHtml(selectedForecastSentence(selectedProbability, forecastSummary.selectedMethod))}</small>
    <div class="result-bars">
      <div>
        <span><b>Selected method</b><em>${escapeHtml(formatPercent(selectedProbability))}</em></span>
        <i style="--bar-value: ${Math.max(0, Math.min(100, Number(selectedProbability) * 100 || 0))}%"></i>
      </div>
      <div>
        <span><b>Nearest market</b><em>${escapeHtml(formatPercent(nearestProbability))}</em></span>
        <i style="--bar-value: ${Math.max(0, Math.min(100, Number(nearestProbability) * 100 || 0))}%"></i>
      </div>
    </div>
  `;
  resultContext.innerHTML = `
    <div class="context-card">
      <span>Nearest market</span>
      <strong>${escapeHtml(formatPercent(nearestProbability))}</strong>
      <p>${delta === null ? "No direct comparison is available." : `Selected forecast is ${escapeHtml(formatPercentagePointDelta(delta))} ${delta >= 0 ? "above" : "below"} the nearest retrieved market.`}</p>
    </div>
    <div class="context-card">
      <span>Evidence base</span>
      <strong>${escapeHtml(summary.marketCount)} markets</strong>
      <p>${escapeHtml(summary.eventCount)} event families · ${escapeHtml(formatMoney(summary.totalVolume))} volume.</p>
    </div>
    <div class="context-card">
      <span>Interpretation</span>
      <p>${escapeHtml(target.interpretation)}</p>
    </div>
    <a class="card-action secondary" href="/forecast/evidence">Back to evidence</a>
    <a class="card-action secondary" href="/forecast">Ask another question</a>
  `;
  methodComparisonMeta.textContent = `${methods.length} aggregation methods compared.`;
  methodTable.innerHTML = methods.map((method) => `
    <article class="method-row ${method.method === forecastSummary.selectedMethod ? "selected" : ""}">
      <div>
        <strong>${escapeHtml(methodLabel(method.method))}</strong>
        <span>${method.method === forecastSummary.selectedMethod ? "Selected method" : `${escapeHtml(method.markets_used)} markets · ${escapeHtml(method.events_used)} events`}</span>
      </div>
      <strong>${escapeHtml(formatPercent(method.forecast_probability))}</strong>
    </article>
  `).join("");
}

async function initResult() {
  try {
    const session = await requestJson("/api/session", { method: "GET" });
    if (!session.authenticated || !session.user) {
      window.location.replace("/");
      return;
    }
    const user = session.user;
    account.innerHTML = `
      <div>
        <span>${escapeHtml(user.role)} account</span>
        <strong>${escapeHtml(displayName(user))}</strong>
      </div>
      <button type="button" id="workspaceSignOut">Sign out</button>
    `;
    document.querySelector("#workspaceSignOut").addEventListener("click", signOut);
    const run = readRun();
    if (!run?.payload) {
      renderEmptyState();
      return;
    }
    renderResult(run);
  } catch {
    window.location.replace("/");
  }
}

initResult();
