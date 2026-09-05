const account = document.querySelector("#workspaceAccount");
const questionForm = document.querySelector("#questionForm");
const forecastQuestion = document.querySelector("#forecastQuestion");
const findEvidenceButton = document.querySelector("#findEvidenceButton");
const questionStatus = document.querySelector("#questionStatus");
const evidenceState = document.querySelector("#evidenceState");
const resultState = document.querySelector("#resultState");
const methodState = document.querySelector("#methodState");
const evidenceStep = document.querySelector("#evidenceStep");

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

function setPremiumAccess(user) {
  const premium = user.role === "premium";
  forecastQuestion.disabled = !premium;
  findEvidenceButton.disabled = !premium;
  questionStatus.textContent = premium
    ? "Custom forecasting is available. Ask a question to retrieve market evidence."
    : "This is a premium workflow. Use a premium account to run custom forecast questions.";
}

async function signOut() {
  await requestJson("/api/logout", { method: "POST", body: "{}" });
  window.location.assign("/");
}

function handleQuestionSubmit(event) {
  event.preventDefault();
  const question = forecastQuestion.value.trim();
  if (!question) {
    questionStatus.textContent = "Enter a forecasting question first.";
    forecastQuestion.focus();
    return;
  }
  runForecast(question);
}

async function runForecast(question) {
  findEvidenceButton.disabled = true;
  findEvidenceButton.textContent = "Finding";
  questionStatus.textContent = "Running event-aware retrieval and forecast aggregation...";
  evidenceState.textContent = "Searching the market corpus. The retrieved set will open on the evidence page.";
  resultState.textContent = "Waiting until evidence has been reviewed.";
  methodState.textContent = "Using the tested query-led forecast service.";
  evidenceStep?.classList.add("active");
  try {
    const payload = await requestJson("/api/forecast/query", {
      method: "POST",
      body: JSON.stringify({ query: question }),
    });
    sessionStorage.setItem("forecastEvidenceRun", JSON.stringify({
      createdAt: new Date().toISOString(),
      question,
      payload,
    }));
    questionStatus.textContent = "Forecast evidence loaded. Opening retrieved markets.";
    window.location.assign("/forecast/evidence");
  } catch (error) {
    questionStatus.textContent = error.message;
    evidenceState.textContent = "Forecast retrieval did not complete.";
    resultState.textContent = "No result available.";
    methodState.textContent = "Check the generated retrieval artifacts and local database before retrying.";
  } finally {
    findEvidenceButton.disabled = false;
    findEvidenceButton.textContent = "Find evidence";
  }
}

async function initForecast() {
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
    setPremiumAccess(user);
    questionForm.addEventListener("submit", handleQuestionSubmit);
  } catch {
    window.location.replace("/");
  }
}

initForecast();
