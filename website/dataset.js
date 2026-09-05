const account = document.querySelector("#workspaceAccount");
const categoryChart = document.querySelector("#categoryChart");
const categoryMetric = document.querySelector("#categoryMetric");
const categoryIntro = document.querySelector("#categoryIntro");
const categoryPicker = document.querySelector("#categoryPicker");
const timeChart = document.querySelector("#timeChart");
const readinessGrid = document.querySelector("#readinessGrid");
const lorenzChart = document.querySelector("#lorenzChart");

let dataset = null;
let selectedMetric = "volume";
let selectedCategories = ["All"];

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

function formatMoney(value) {
  const number = Number(value || 0);
  if (number >= 1000000000) return `$${(number / 1000000000).toFixed(1)}B`;
  if (number >= 1000000) return `$${(number / 1000000).toFixed(1)}M`;
  if (number >= 1000) return `$${(number / 1000).toFixed(1)}K`;
  return `$${number.toFixed(0)}`;
}

function formatMetric(value, metric) {
  return metric === "volume" ? formatMoney(value) : formatCount(value);
}

function percent(value, digits = 1) {
  return `${Number(value || 0).toFixed(digits).replace(/\.0$/, "")}%`;
}

function metricLabel(metric) {
  return {
    volume: "traded volume",
    markets: "markets",
    events: "event families",
  }[metric];
}

function categoryShare(row, metric) {
  const total = dataset.categories.reduce((sum, item) => sum + Number(item[metric] || 0), 0);
  return total > 0 ? (Number(row[metric] || 0) / total) * 100 : 0;
}

async function signOut() {
  await requestJson("/api/logout", { method: "POST", body: "{}" });
  window.location.assign("/");
}

function renderAccount(session) {
  if (session.authenticated && session.user) {
    const firstName = displayName(session.user);
    account.innerHTML = `
      <div>
        <span>${escapeHtml(session.user.role)} account</span>
        <strong>${escapeHtml(firstName)}</strong>
      </div>
      <button type="button" id="workspaceSignOut">Sign out</button>
    `;
    document.querySelector("#workspaceSignOut").addEventListener("click", signOut);
    return;
  }
  account.innerHTML = '<a class="card-action secondary compact-action" href="/">Sign in</a>';
}

function renderCategoryChart() {
  const rows = [...dataset.categories]
    .sort((a, b) => Number(b[selectedMetric] || 0) - Number(a[selectedMetric] || 0))
    .slice(0, 10);
  const max = Math.max(...rows.map((row) => Number(row[selectedMetric] || 0)), 1);
  const top = rows[0];
  categoryIntro.textContent = `${top.category} is the largest category by ${metricLabel(selectedMetric)}. The table keeps the comparison readable while still showing relative scale.`;
  categoryChart.innerHTML = rows.map((row) => {
    const value = Number(row[selectedMetric] || 0);
    const width = Math.max((value / max) * 100, 1.5);
    const share = categoryShare(row, selectedMetric);
    return `
      <div class="category-row">
        <strong>${escapeHtml(row.category)}</strong>
        <span>${escapeHtml(percent(share))}</span>
        <span>${escapeHtml(formatMetric(value, selectedMetric))}</span>
        <i style="--bar-value: ${width}%"></i>
      </div>
    `;
  }).join("");
}

function monthlySeries(category) {
  return dataset.monthlyCreation.rows
    .filter((row) => row.category === category)
    .map((row) => ({ month: row.month, value: Number(row.markets || 0) }));
}

function renderCategoryPicker() {
  const options = dataset.monthlyCreation.categories.slice(0, 9);
  categoryPicker.innerHTML = options.map((category) => `
    <button
      type="button"
      class="${selectedCategories.includes(category) ? "active" : ""}"
      data-category="${escapeHtml(category)}"
    >${escapeHtml(category)}</button>
  `).join("");
}

function toggleCategory(category) {
  if (category === "All") {
    selectedCategories = ["All"];
  } else {
    selectedCategories = selectedCategories.filter((item) => item !== "All");
    if (selectedCategories.includes(category)) {
      selectedCategories = selectedCategories.filter((item) => item !== category);
    } else {
      selectedCategories = [...selectedCategories, category].slice(-3);
    }
    if (selectedCategories.length === 0) selectedCategories = ["All"];
  }
  renderCategoryPicker();
  renderTimeChart();
}

function renderTimeChart() {
  const colors = ["#12675f", "#2f6f9f", "#9a6d1b"];
  const series = selectedCategories.map((category, index) => ({
    category,
    color: category === "All" ? "#12675f" : colors[index % colors.length],
    values: monthlySeries(category),
  })).filter((item) => item.values.length);
  const allMonths = [...new Set(series.flatMap((item) => item.values.map((point) => point.month)))].sort();
  const maxValue = Math.max(...series.flatMap((item) => item.values.map((point) => point.value)), 1);
  const width = 940;
  const height = 370;
  const padding = { top: 18, right: 20, bottom: 42, left: 54 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const monthIndex = new Map(allMonths.map((month, index) => [month, index]));
  const xForMonth = (month) => padding.left + (monthIndex.get(month) / Math.max(allMonths.length - 1, 1)) * innerWidth;
  const yForValue = (value) => padding.top + innerHeight - (value / maxValue) * innerHeight;
  const paths = series.map((item) => {
    const points = item.values
      .filter((point) => monthIndex.has(point.month))
      .map((point) => `${xForMonth(point.month).toFixed(2)},${yForValue(point.value).toFixed(2)}`)
      .join(" ");
    return `<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>`;
  }).join("");
  const tickIndexes = [0, .25, .5, .75, 1].map((fraction) => Math.round(fraction * Math.max(allMonths.length - 1, 0)));
  const ticks = [...new Set(tickIndexes)].map((index) => {
    const month = allMonths[index] || "";
    const x = padding.left + (index / Math.max(allMonths.length - 1, 1)) * innerWidth;
    return `<text x="${x.toFixed(1)}" y="${height - 12}" text-anchor="middle">${escapeHtml(month)}</text>`;
  }).join("");
  const yTicks = [0, .5, 1].map((fraction) => {
    const y = padding.top + innerHeight - fraction * innerHeight;
    const value = Math.round(maxValue * fraction);
    return `
      <line x1="${padding.left}" x2="${width - padding.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"></line>
      <text x="${padding.left - 12}" y="${(y + 4).toFixed(1)}" text-anchor="end">${formatCount(value)}</text>
    `;
  }).join("");
  const legend = series.length > 1 ? `
    <div class="chart-legend">
      ${series.map((item) => `<span><i style="background: ${item.color}"></i>${escapeHtml(item.category)}</span>`).join("")}
    </div>
  ` : "";
  timeChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Monthly market creation line chart">
      <g class="grid">${yTicks}</g>
      <line class="axis" x1="${padding.left}" x2="${width - padding.right}" y1="${height - padding.bottom}" y2="${height - padding.bottom}"></line>
      <g class="series">${paths}</g>
      <g class="x-ticks">${ticks}</g>
    </svg>
    ${legend}
  `;
}

function renderReadiness() {
  const overview = dataset.overview;
  const items = [
    ["Resolved", percent(overview.resolvedPercent), "usable outcome labels"],
    ["Price history", percent(overview.priceHistoryPercent), "usable historical snapshots"],
    ["Positive volume", percent(overview.positiveVolumePercent), "markets with traded volume"],
    ["Positive liquidity", percent(overview.positiveLiquidityPercent), "markets with liquidity"],
  ];
  readinessGrid.innerHTML = items.map(([label, value, note]) => `
    <article>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <p>${escapeHtml(note)}</p>
    </article>
  `).join("");
}

function renderLorenzCurve() {
  const sourcePoints = dataset.volumeConcentration?.points ?? [];
  const bottomWeighted = sourcePoints
    .map((point) => ({
      marketShare: 100 - Number(point.marketShare || 0),
      volumeShare: 100 - Number(point.volumeShare || 0),
    }))
    .filter((point) => point.marketShare >= 0 && point.marketShare <= 100)
    .sort((a, b) => a.marketShare - b.marketShare);
  const points = [{ marketShare: 0, volumeShare: 0 }, ...bottomWeighted, { marketShare: 100, volumeShare: 100 }]
    .filter((point, index, array) => index === 0 || point.marketShare !== array[index - 1].marketShare);
  const width = 620;
  const height = 430;
  const padding = { top: 22, right: 24, bottom: 54, left: 56 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const xFor = (value) => padding.left + (value / 100) * innerWidth;
  const yFor = (value) => padding.top + innerHeight - (value / 100) * innerHeight;
  const path = points.map((point) => `${xFor(point.marketShare).toFixed(2)},${yFor(point.volumeShare).toFixed(2)}`).join(" ");
  const ticks = [0, 25, 50, 75, 100].map((value) => {
    const x = xFor(value);
    const y = yFor(value);
    return `
      <line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${padding.top}" y2="${height - padding.bottom}"></line>
      <line x1="${padding.left}" x2="${width - padding.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"></line>
      <text x="${x.toFixed(1)}" y="${height - 22}" text-anchor="middle">${value}%</text>
      <text x="${padding.left - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end">${value}%</text>
    `;
  }).join("");
  const topOne = dataset.concentration?.top_1_percent;
  const topTen = dataset.concentration?.top_10_percent;
  lorenzChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Lorenz curve showing cumulative market volume concentration">
      <g class="grid">${ticks}</g>
      <line class="equality" x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${padding.top}"></line>
      <polyline class="lorenz-line" points="${path}" fill="none"></polyline>
      <text x="${padding.left + innerWidth / 2}" y="${height - 4}" text-anchor="middle">Share of markets</text>
      <text x="14" y="${padding.top + innerHeight / 2}" transform="rotate(-90 14 ${padding.top + innerHeight / 2})" text-anchor="middle">Share of volume</text>
    </svg>
    <div class="lorenz-notes">
      <span>Top 1%: <strong>${escapeHtml(percent(topOne))}</strong></span>
      <span>Top 10%: <strong>${escapeHtml(percent(topTen))}</strong></span>
    </div>
  `;
}

function bindControls() {
  categoryMetric.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-metric]");
    if (!button) return;
    selectedMetric = button.dataset.metric;
    categoryMetric.querySelectorAll("button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    renderCategoryChart();
  });
  categoryPicker.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-category]");
    if (button) toggleCategory(button.dataset.category);
  });
}

async function initDataset() {
  try {
    const [session, data] = await Promise.all([
      requestJson("/api/session", { method: "GET" }).catch(() => ({ authenticated: false, user: null })),
      requestJson("/api/overview", { method: "GET" }),
    ]);
    dataset = data;
    renderAccount(session);
    renderCategoryChart();
    renderCategoryPicker();
    renderTimeChart();
    renderReadiness();
    renderLorenzCurve();
    bindControls();
  } catch (error) {
    categoryChart.innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

initDataset();
