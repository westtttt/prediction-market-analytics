const tabs = {
  signIn: document.querySelector("#signInTab"),
  create: document.querySelector("#createTab"),
};

const panels = {
  signIn: document.querySelector("#signInPanel"),
  create: document.querySelector("#createPanel"),
  session: document.querySelector("#sessionPanel"),
};

const pageTitle = document.querySelector("#pageTitle");
const pageSubtitle = document.querySelector("#pageSubtitle");
const message = document.querySelector("#message");

const modeCopy = {
  signIn: {
    title: "Sign in to continue.",
    subtitle: "Access the forecasting workspace to ask questions, review supporting market evidence, and compare forecast signals.",
    message: "Use a demo account or your local account.",
  },
  create: {
    title: "Create your workspace.",
    subtitle: "Set up a local account for saving access state before the forecast, evidence, and results pages are added.",
    message: "Create a local account for this prototype.",
  },
};

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setMode(mode) {
  const isCreate = mode === "create";
  tabs.signIn.classList.toggle("active", !isCreate);
  tabs.create.classList.toggle("active", isCreate);
  tabs.signIn.setAttribute("aria-selected", String(!isCreate));
  tabs.create.setAttribute("aria-selected", String(isCreate));
  panels.signIn.hidden = isCreate;
  panels.create.hidden = !isCreate;
  pageTitle.textContent = modeCopy[mode].title;
  pageSubtitle.textContent = modeCopy[mode].subtitle;
  setMessage(modeCopy[mode].message);
}

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

function renderSession(payload) {
  const user = payload.user;
  if (!payload.authenticated || !user) {
    panels.session.hidden = true;
    panels.signIn.hidden = tabs.create.classList.contains("active");
    panels.create.hidden = !tabs.create.classList.contains("active");
    return;
  }

  panels.signIn.hidden = true;
  panels.create.hidden = true;
  panels.session.hidden = false;
  panels.session.innerHTML = `
    <div>
      <span>Signed in</span>
      <strong>${escapeHtml(user.name)}</strong>
      <strong>${escapeHtml(user.email)}</strong>
    </div>
    <button type="button" id="signOutButton">Sign out</button>
  `;
  document.querySelector("#signOutButton").addEventListener("click", async () => {
    await requestJson("/api/logout", { method: "POST", body: "{}" });
    renderSession({ authenticated: false, user: null });
    setMode("signIn");
    setMessage("Signed out.");
  });
  setMessage("Account ready. The forecasting workspace will be added next.");
  pageTitle.textContent = "Account ready.";
  pageSubtitle.textContent = "You are signed in. The next page we add will become the entry point to the forecasting workspace.";
}

function goToWorkspace() {
  window.location.assign("/workspace");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

tabs.signIn.addEventListener("click", () => setMode("signIn"));
tabs.create.addEventListener("click", () => setMode("create"));

panels.signIn.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("Signing in...");
  const form = new FormData(event.currentTarget);
  try {
    const payload = await requestJson("/api/login", {
      method: "POST",
      body: JSON.stringify({
        email: form.get("email"),
        password: form.get("password"),
      }),
    });
    renderSession(payload);
    goToWorkspace();
  } catch (error) {
    setMessage(error.message, true);
  }
});

panels.create.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("Creating account...");
  const form = new FormData(event.currentTarget);
  try {
    const payload = await requestJson("/api/register", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        email: form.get("email"),
        password: form.get("password"),
      }),
    });
    event.currentTarget.reset();
    renderSession(payload);
    goToWorkspace();
  } catch (error) {
    setMessage(error.message, true);
  }
});

requestJson("/api/session", { method: "GET" })
  .then(renderSession)
  .catch(() => setMessage("Start the local final website server to sign in.", true));
