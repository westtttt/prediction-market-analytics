const account = document.querySelector("#workspaceAccount");
const findingSummary = document.querySelector("#findingSummary");
const queryMethodList = document.querySelector("#queryMethodList");
const eventSlopeChart = document.querySelector("#eventSlopeChart");
const eventHorizonChart = document.querySelector("#eventHorizonChart");

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
  return String(group || "").replace("_days", "d");
}

function svgPoint(x, y) {
  return `${Number(x).toFixed(1)},${Number(y).toFixed(1)}`;
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

function renderSummary(data) {
  const query = data.queryLed;
  const event = data.eventStructure.overall;
  findingSummary.innerHTML = `
    <article>
      <span>Query-led aggregation</span>
      <strong>Direct markets are hard to beat.</strong>
      <p>
        Best method: ${escapeHtml(query.bestMethod.label)} at
        ${escapeHtml(score(query.bestMethod.brier))} Brier across
        ${escapeHtml(query.bestMethod.targetMarkets)} available direct-market targets.
      </p>
    </article>
    <article class="accent">
      <span>Event-structure correction</span>
      <strong>${escapeHtml(percent(event.brierImprovementPercent))} lower Brier after normalisation.</strong>
      <p>
        Coherent sibling markets improve from ${escapeHtml(score(event.rawBrier))}
        to ${escapeHtml(score(event.normalisedBrier))} Brier.
      </p>
    </article>
  `;
}

function renderQueryMethods(methods) {
  const sorted = [...methods].sort((a, b) => Number(a.brier || 0) - Number(b.brier || 0)).slice(0, 4);
  queryMethodList.innerHTML = sorted.map((method, index) => {
    return `
      <article class="result-method-row ${index === 0 ? "best" : ""}">
        <small>${index + 1}</small>
        <div>
          <strong>${escapeHtml(method.label)}</strong>
          <span>${escapeHtml(formatCount(method.forecastRows))} rows · ${escapeHtml(method.meanMarketsUsed)} markets used on average</span>
        </div>
        <dl>
          <div><dt>Brier</dt><dd>${escapeHtml(score(method.brier))}</dd></div>
          <div><dt>Log loss</dt><dd>${escapeHtml(score(method.logLoss))}</dd></div>
        </dl>
      </article>
    `;
  }).join("");
}

function renderSlopeChart(event) {
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
  const chart = metrics.map((metric) => {
    const yRaw = metric.y - 24;
    const yNormalised = metric.y + 24;
    return `
      <g>
        <text x="24" y="${metric.y - 22}" class="chart-title">${escapeHtml(metric.label)}</text>
        <text x="24" y="${metric.y + 4}" class="chart-note">${escapeHtml(percent(metric.improvement))} improvement</text>
        <line x1="${xRaw}" y1="${yRaw}" x2="${xNormalised}" y2="${yNormalised}" class="slope-line"></line>
        <circle cx="${xRaw}" cy="${yRaw}" r="7" class="slope-dot raw"></circle>
        <circle cx="${xNormalised}" cy="${yNormalised}" r="7" class="slope-dot"></circle>
        <text x="${xRaw}" y="${yRaw - 16}" text-anchor="middle" class="chart-value">${escapeHtml(score(metric.raw))}</text>
        <text x="${xNormalised}" y="${yNormalised - 16}" text-anchor="middle" class="chart-value">${escapeHtml(score(metric.normalised))}</text>
      </g>
    `;
  }).join("");
  eventSlopeChart.innerHTML = `
    <svg viewBox="0 0 620 290" role="img" aria-label="Raw scores slope down after normalisation">
      <text x="${xRaw}" y="24" text-anchor="middle" class="chart-label">Raw</text>
      <text x="${xNormalised}" y="24" text-anchor="middle" class="chart-label">Normalised</text>
      <line x1="24" y1="146" x2="596" y2="146" class="chart-rule"></line>
      ${chart}
    </svg>
  `;
}

function renderHorizonChart(horizons) {
  const sorted = [...horizons].sort((a, b) => Number.parseInt(a.group, 10) - Number.parseInt(b.group, 10));
  const values = sorted.map((horizon) => Number(horizon.brierImprovementPercent || 0));
  const max = Math.max(...values, 1) * 1.15;
  const width = 690;
  const height = 300;
  const left = 62;
  const right = width - 28;
  const top = 42;
  const bottom = height - 54;
  const x = (index) => left + ((right - left) * index) / Math.max(sorted.length - 1, 1);
  const y = (value) => bottom - ((bottom - top) * value) / max;
  const points = sorted.map((horizon, index) => [x(index), y(Number(horizon.brierImprovementPercent || 0))]);
  const linePath = points.map((point) => svgPoint(point[0], point[1])).join(" ");
  const yTicks = [0, Math.round(max / 2), Math.round(max)];
  eventHorizonChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Brier improvement by forecast horizon">
      ${yTicks.map((tick) => `
        <line x1="${left}" y1="${y(tick)}" x2="${right}" y2="${y(tick)}" class="chart-grid"></line>
        <text x="${left - 12}" y="${y(tick) + 4}" text-anchor="end" class="chart-label">${escapeHtml(percent(tick, 0))}</text>
      `).join("")}
      <polyline points="${linePath}" class="horizon-line"></polyline>
      ${sorted.map((horizon, index) => {
        const value = Number(horizon.brierImprovementPercent || 0);
        return `
          <g>
            <circle cx="${x(index)}" cy="${y(value)}" r="7" class="horizon-dot"></circle>
            <text x="${x(index)}" y="${y(value) - 16}" text-anchor="middle" class="chart-value">${escapeHtml(percent(value))}</text>
            <text x="${x(index)}" y="${bottom + 28}" text-anchor="middle" class="chart-label">${escapeHtml(horizonLabel(horizon.group))}</text>
          </g>
        `;
      }).join("")}
    </svg>
  `;
}

async function initResults() {
  try {
    const [session, data] = await Promise.all([
      requestJson("/api/session", { method: "GET" }).catch(() => ({ authenticated: false, user: null })),
      requestJson("/api/results", { method: "GET" }),
    ]);
    renderAccount(session);
    renderSummary(data);
    renderQueryMethods(data.queryLed.methods);
    renderSlopeChart(data.eventStructure.overall);
    renderHorizonChart(data.eventStructure.horizons);
  } catch (error) {
    queryMethodList.innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

initResults();
