const account = document.querySelector("#workspaceAccount");
const evidenceQuestion = document.querySelector("#evidenceQuestion");
const evidenceSummary = document.querySelector("#evidenceSummary");
const methodNote = document.querySelector("#methodNote");
const marketReviewMeta = document.querySelector("#marketReviewMeta");
const eventList = document.querySelector("#eventList");
const marketList = document.querySelector("#marketList");
const resultLink = document.querySelector("#resultLink");

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
  return String(method || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
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
  evidenceQuestion.textContent = "No retrieved evidence is available yet. Start with a forecast question first.";
  evidenceSummary.innerHTML = "";
  methodNote.innerHTML = `
    <span>No forecast run</span>
    <p>Use the forecast page to run retrieval before reviewing markets.</p>
  `;
  eventList.innerHTML = "";
  marketList.innerHTML = `<a class="card-action" href="/forecast">Start a forecast</a>`;
  resultLink.setAttribute("aria-disabled", "true");
}

function renderEvidence(run) {
  const { payload, question } = run;
  const target = payload.target;
  const forecast = payload.forecast;
  const summary = target.summary;
  const events = target.events ?? [];
  const markets = forecast.evidenceMarkets ?? [];
  const selected = forecast.summary;
  const nearest = (forecast.aggregationMethods ?? []).find((method) => method.method === "top_similarity");
  resultLink.removeAttribute("aria-disabled");

  evidenceQuestion.innerHTML = `Question: <strong>${escapeHtml(question)}</strong>`;
  evidenceSummary.innerHTML = `
    <div><span>Events</span><strong>${escapeHtml(summary.eventCount)}</strong></div>
    <div><span>Markets</span><strong>${escapeHtml(summary.marketCount)}</strong></div>
    <div><span>Volume</span><strong>${escapeHtml(formatMoney(summary.totalVolume))}</strong></div>
    <div><span>Selected signal</span><strong>${escapeHtml(formatPercent(selected.selectedForecastProbability))}</strong></div>
    <div><span>Nearest market</span><strong>${escapeHtml(formatPercent(nearest?.forecast_probability))}</strong></div>
    <div><span>History</span><strong>${escapeHtml(formatPercent(summary.priceHistoryCoveragePercent / 100))}</strong></div>
  `;
  methodNote.innerHTML = `
    <span>Method trace</span>
    <p>${escapeHtml(target.interpretation)}</p>
    <dl>
      <div><dt>Retrieval</dt><dd>${escapeHtml(methodLabel(payload.retrieval.method))}</dd></div>
      <div><dt>Forecast</dt><dd>${escapeHtml(methodLabel(selected.selectedMethod))}</dd></div>
      <div><dt>Status</dt><dd>${selected.accuracyEvaluated ? "Accuracy evaluated" : "Current snapshot only"}</dd></div>
    </dl>
  `;
  marketReviewMeta.textContent = `${markets.length} markets shown from ${summary.marketCount} retrieved.`;
  eventList.innerHTML = events.map((event) => `
    <article class="event-row">
      <div>
        <span>Rank ${escapeHtml(event.rank)} · ${escapeHtml(event.category)}</span>
        <strong>${escapeHtml(event.title)}</strong>
      </div>
      <div>
        <strong>${escapeHtml(event.markets)}</strong>
        <span>markets</span>
      </div>
    </article>
  `).join("");
  marketList.innerHTML = markets.map((market) => `
    <article class="market-row">
      <div>
        <span>${escapeHtml(market.eventTitle)}</span>
        <strong>${escapeHtml(market.question)}</strong>
      </div>
      <div class="market-row-meta">
        <span>${escapeHtml(formatPercent(market.probability))}</span>
        <span>${escapeHtml(formatMoney(market.volume))}</span>
        <span>${market.hasPriceHistory ? "History" : "No history"}</span>
      </div>
    </article>
  `).join("");
}

async function initEvidence() {
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
    renderEvidence(run);
  } catch {
    window.location.replace("/");
  }
}

initEvidence();
