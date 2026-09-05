const account = document.querySelector("#workspaceAccount");
const coverage = document.querySelector("#backtestCoverage");
const methodTable = document.querySelector("#backtestMethodTable");
const slopeChart = document.querySelector("#backtestSlopeChart");
const horizonTable = document.querySelector("#horizonDetailTable");

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
  if (!user) return "";
  if (user.email === "premium@example.test" || user.email === "free@example.test") {
    return "Will";
  }
  const name = String(user.name ?? "").trim();
  return name.split(/\s+/)[0] || "Will";
}

function formatCount(value) {
  const number = Number(value || 0);
  if (number >= 1000000) return `${(number / 1000000).toFixed(2).replace(/\.0+$/, "")}M`;
  if (number >= 1000) return `${(number / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  return new Intl.NumberFormat("en-GB").format(number);
}

function percent(value, digits = 1) {
  return `${Number(value || 0).toFixed(digits).replace(/\.0$/, "")}%`;
}

function score(value) {
  return Number(value || 0).toFixed(3).replace(/^0/, "");
}

function horizonLabel(group) {
  return String(group || "").replace("_days", "d").toUpperCase();
}

async function signOut() {
  await requestJson("/api/logout", { method: "POST", body: "{}" });
  window.location.assign("/");
}

function renderAccount(session) {
  if (session.authenticated && session.user) {
    account.innerHTML = `
      <div>
        <span>${escapeHtml(session.user.role)} account</span>
        <strong>${escapeHtml(displayName(session.user))}</strong>
      </div>
      <button type="button" id="workspaceSignOut">Sign out</button>
    `;
    document.querySelector("#workspaceSignOut").addEventListener("click", signOut);
    return;
  }
  account.innerHTML = '<a class="card-action secondary compact-action" href="/">Sign in</a>';
}

function renderCoverage(results, overview) {
  const query = results.queryLed;
  const event = results.eventStructure.overall;
  const items = [
    ["Query rows", formatCount(query.evaluatedRows), "historical aggregation forecasts"],
    ["Target markets", formatCount(query.targetMarkets), "resolved target questions"],
    ["Event rows", formatCount(event.predictionRows), "normalisation predictions"],
    ["Event families", formatCount(event.events), "eligible coherent families"],
    ["Price history", percent(overview.overview.priceHistoryPercent), "markets with usable snapshots"],
    ["Leakage rows", formatCount(query.leakageRows), "flagged in query-led tests"],
  ];
  coverage.innerHTML = items.map(([label, value, note]) => `
    <article>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <p>${escapeHtml(note)}</p>
    </article>
  `).join("");
}

function renderMethods(methods) {
  const sorted = [...methods].sort((a, b) => Number(a.brier || 0) - Number(b.brier || 0));
  methodTable.innerHTML = `
    <div class="backtest-method-head" aria-hidden="true">
      <span>Method</span>
      <span>Rows</span>
      <span>Markets used</span>
      <span>Brier</span>
      <span>Log loss</span>
    </div>
    ${sorted.map((method, index) => `
      <article class="${index === 0 ? "best" : ""}">
        <div>
          <small>${index + 1}</small>
          <strong>${escapeHtml(method.label)}</strong>
        </div>
        <span>${escapeHtml(formatCount(method.forecastRows))}</span>
        <span>${escapeHtml(method.meanMarketsUsed)}</span>
        <b>${escapeHtml(score(method.brier))}</b>
        <b>${escapeHtml(score(method.logLoss))}</b>
      </article>
    `).join("")}
  `;
}

function renderSlope(event) {
  const metrics = [
    {
      label: "Brier score",
      raw: Number(event.rawBrier || 0),
      normalised: Number(event.normalisedBrier || 0),
      improvement: event.brierImprovementPercent,
      y: 102,
    },
    {
      label: "Log loss",
      raw: Number(event.rawLogLoss || 0),
      normalised: Number(event.normalisedLogLoss || 0),
      improvement: event.logLossImprovementPercent,
      y: 222,
    },
  ];
  const xRaw = 250;
  const xNormalised = 545;
  slopeChart.innerHTML = `
    <svg viewBox="0 0 620 290" role="img" aria-label="Raw scores decrease after normalisation">
      <text x="${xRaw}" y="24" text-anchor="middle" class="chart-label">Raw</text>
      <text x="${xNormalised}" y="24" text-anchor="middle" class="chart-label">Normalised</text>
      <line x1="24" y1="146" x2="596" y2="146" class="chart-rule"></line>
      ${metrics.map((metric) => `
        <g>
          <text x="24" y="${metric.y - 22}" class="chart-title">${escapeHtml(metric.label)}</text>
          <text x="24" y="${metric.y + 4}" class="chart-note">${escapeHtml(percent(metric.improvement))} improvement</text>
          <line x1="${xRaw}" y1="${metric.y - 24}" x2="${xNormalised}" y2="${metric.y + 24}" class="slope-line"></line>
          <circle cx="${xRaw}" cy="${metric.y - 24}" r="7" class="slope-dot raw"></circle>
          <circle cx="${xNormalised}" cy="${metric.y + 24}" r="7" class="slope-dot"></circle>
          <text x="${xRaw}" y="${metric.y - 40}" text-anchor="middle" class="chart-value">${escapeHtml(score(metric.raw))}</text>
          <text x="${xNormalised}" y="${metric.y + 8}" text-anchor="middle" class="chart-value">${escapeHtml(score(metric.normalised))}</text>
        </g>
      `).join("")}
    </svg>
  `;
}

function renderHorizons(horizons) {
  const sorted = [...horizons].sort((a, b) => Number.parseInt(a.group, 10) - Number.parseInt(b.group, 10));
  horizonTable.innerHTML = `
    <div class="horizon-detail-head" aria-hidden="true">
      <span>Horizon</span>
      <span>Rows</span>
      <span>Raw Brier</span>
      <span>Normalised</span>
      <span>Brier delta</span>
      <span>Log-loss delta</span>
    </div>
    ${sorted.map((horizon) => `
      <article>
        <strong>${escapeHtml(horizonLabel(horizon.group))}</strong>
        <span>${escapeHtml(formatCount(horizon.predictionRows))}</span>
        <span>${escapeHtml(score(horizon.rawBrier))}</span>
        <span>${escapeHtml(score(horizon.normalisedBrier))}</span>
        <b>${escapeHtml(percent(horizon.brierImprovementPercent))}</b>
        <b>${escapeHtml(percent(horizon.logLossImprovementPercent))}</b>
      </article>
    `).join("")}
  `;
}

async function initBacktesting() {
  try {
    const [session, results, overview] = await Promise.all([
      requestJson("/api/session", { method: "GET" }).catch(() => ({ authenticated: false, user: null })),
      requestJson("/api/results", { method: "GET" }),
      requestJson("/api/overview", { method: "GET" }),
    ]);
    renderAccount(session);
    renderCoverage(results, overview);
    renderMethods(results.queryLed.methods);
    renderSlope(results.eventStructure.overall);
    renderHorizons(results.eventStructure.horizons);
  } catch (error) {
    methodTable.innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

initBacktesting();
