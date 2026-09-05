const account = document.querySelector("#workspaceAccount");
const workspaceTitle = document.querySelector("#workspaceTitle");
const workspaceIntro = document.querySelector("#workspaceIntro");
const workspaceMetrics = document.querySelector("#workspaceMetrics");
const workspaceResult = document.querySelector("#workspaceResult");

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

function formatCount(value) {
  const number = Number(value || 0);
  if (number >= 1000000) return `${(number / 1000000).toFixed(2).replace(/\.0+$/, "")}M`;
  if (number >= 1000) return `${(number / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  return new Intl.NumberFormat("en-GB").format(number);
}

function percent(value, digits = 1) {
  return `${Number(value || 0).toFixed(digits).replace(/\.0$/, "")}%`;
}

async function signOut() {
  await requestJson("/api/logout", { method: "POST", body: "{}" });
  window.location.assign("/");
}

async function initWorkspace() {
  try {
    const [session, data] = await Promise.all([
      requestJson("/api/session", { method: "GET" }),
      requestJson("/api/overview", { method: "GET" }),
    ]);
    if (!session.authenticated || !session.user) {
      window.location.replace("/");
      return;
    }
    const user = session.user;
    const firstName = displayName(user);
    workspaceTitle.textContent = `Welcome back, ${firstName}.`;
    workspaceIntro.textContent = `${firstName}, start a forecast or review the evidence behind the project.`;
    workspaceResult.innerHTML = `
      <span>Headline result</span>
      <strong>${escapeHtml(percent(data.eventStructure?.normalisation?.overall?.brierImprovementPercent))}</strong>
      <p>Lower Brier score after normalising coherent event-family probabilities.</p>
    `;
    workspaceMetrics.innerHTML = [
      ["Markets", formatCount(data.overview.markets), "Polymarket contracts in the cleaned corpus."],
      ["Events", formatCount(data.overview.events), "Grouped event-family rows for analysis."],
      ["Price history", percent(data.overview.priceHistoryPercent), "Markets with usable historical snapshots."],
      ["Resolved", percent(data.overview.resolvedPercent), "Markets with outcome labels for evaluation."],
    ].map(([label, value, note]) => `
      <article class="metric-card">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <p>${escapeHtml(note)}</p>
      </article>
    `).join("");
    account.innerHTML = `
      <div>
        <span>${escapeHtml(user.role)} account</span>
        <strong>${escapeHtml(firstName)}</strong>
      </div>
      <button type="button" id="workspaceSignOut">Sign out</button>
    `;
    document.querySelector("#workspaceSignOut").addEventListener("click", signOut);
  } catch {
    window.location.replace("/");
  }
}

initWorkspace();
