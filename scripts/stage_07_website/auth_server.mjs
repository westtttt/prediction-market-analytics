import { createHmac, pbkdf2Sync, randomBytes, timingSafeEqual } from "node:crypto";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const siteDir = path.join(root, "website");
const localDir = path.join(root, ".local");
const userStorePath = path.join(localDir, "website_users.json");
const demoUsersPath = path.join(root, "config/demo_users.json");
const forecastBridgePath = path.join(root, "scripts", "stage_07_website", "query_forecast_api.py");
const dashboardDataPath = path.join(root, "website", "data", "dashboard_data.js");
const backtestDataPath = path.join(root, "website", "data", "backtest_data.js");
const sessionSecret = process.env.FINAL_WEBSITE_SESSION_SECRET || randomBytes(32).toString("base64url");
const sessionCookie = "forecast_evidence_session";
const sessionSeconds = 8 * 60 * 60;

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};

function parseArgs() {
  const args = new Map();
  for (let index = 2; index < process.argv.length; index += 1) {
    if (process.argv[index].startsWith("--")) {
      args.set(process.argv[index].slice(2), process.argv[index + 1]);
      index += 1;
    }
  }
  return {
    host: args.get("host") || "127.0.0.1",
    port: Number(args.get("port") || 4285),
  };
}

async function ensureUserStore() {
  await mkdir(localDir, { recursive: true });
  try {
    await readFile(userStorePath, "utf8");
  } catch {
    const demoUsers = JSON.parse(await readFile(demoUsersPath, "utf8")).users;
    await writeFile(userStorePath, JSON.stringify({ users: demoUsers }, null, 2));
  }
}

async function readUsers() {
  await ensureUserStore();
  return JSON.parse(await readFile(userStorePath, "utf8")).users;
}

async function writeUsers(users) {
  await mkdir(localDir, { recursive: true });
  await writeFile(userStorePath, JSON.stringify({ users }, null, 2));
}

function hashPassword(password) {
  const salt = randomBytes(16);
  const iterations = 210000;
  const hash = pbkdf2Sync(password, salt, iterations, 32, "sha256");
  return {
    algorithm: "pbkdf2_sha256",
    iterations,
    salt: salt.toString("base64"),
    password_hash: hash.toString("base64"),
  };
}

function verifyPassword(password, user) {
  if (user.algorithm !== "pbkdf2_sha256") {
    return false;
  }
  const salt = Buffer.from(user.salt, "base64");
  const expected = Buffer.from(user.password_hash, "base64");
  const actual = pbkdf2Sync(password, salt, user.iterations, expected.length, "sha256");
  return expected.length === actual.length && timingSafeEqual(expected, actual);
}

function sign(value) {
  return createHmac("sha256", sessionSecret).update(value).digest("base64url");
}

function createSession(user) {
  const payload = Buffer.from(JSON.stringify({
    email: user.email,
    exp: Math.floor(Date.now() / 1000) + sessionSeconds,
  })).toString("base64url");
  return `${payload}.${sign(payload)}`;
}

async function currentUser(request) {
  const cookies = Object.fromEntries((request.headers.cookie || "")
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [name, ...value] = part.split("=");
      return [name, value.join("=")];
    }));
  const raw = cookies[sessionCookie] || "";
  const [payload, signature] = raw.split(".");
  if (!payload || !signature || sign(payload) !== signature) {
    return null;
  }
  const decoded = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  if (!decoded.exp || decoded.exp <= Math.floor(Date.now() / 1000)) {
    return null;
  }
  const users = await readUsers();
  const user = users.find((candidate) => candidate.email === decoded.email);
  return user ? publicUser(user) : null;
}

function publicUser(user) {
  return { email: user.email, name: user.name, role: user.role };
}

async function requestBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

function jsonResponse(response, payload, status = 200, headers = {}) {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    ...headers,
  });
  response.end(JSON.stringify(payload));
}

async function readWindowData(filePath, globalName) {
  const source = await readFile(filePath, "utf8");
  const match = source.match(new RegExp(`window\\.${globalName}\\s*=\\s*([\\s\\S]*);\\s*$`));
  if (!match) {
    throw new Error(`${globalName} assignment was not found`);
  }
  return JSON.parse(match[1]);
}

async function readDashboardData() {
  return readWindowData(dashboardDataPath, "DASHBOARD_DATA");
}

async function readBacktestData() {
  return readWindowData(backtestDataPath, "BACKTEST_DATA");
}

function compactOverviewData(data) {
  const monthlyCreation = data.charts?.monthlyCreation ?? { categories: [], rows: [] };
  const eventNormalisation = data.eventStructure?.normalisation ?? null;
  return {
    overview: data.overview,
    concentration: data.concentration,
    volumeConcentration: data.charts?.volumeConcentration ?? null,
    descriptive: data.descriptive,
    categories: data.categories,
    monthlyCreation,
    eventStructure: {
      coverage: data.eventStructure?.coverage ?? null,
      labels: data.eventStructure?.labels ?? [],
      normalisation: eventNormalisation ? {
        overall: eventNormalisation.overall,
        horizons: eventNormalisation.horizons,
      } : null,
    },
  };
}

function compactResultsData(dashboardData, backtestData) {
  const methodLabels = {
    direct_market: "Direct target market",
    top_similarity: "Nearest retrieved market",
    same_category_top5_similarity_weighted: "Same-category top five",
    top5_similarity_weighted_mean: "Top-five relevance weighted",
    liquidity_weighted_mean: "Liquidity-weighted average",
    same_category_similarity_weighted: "Same-category weighted",
  };
  const methods = (backtestData.methods ?? []).map((method) => ({
    ...method,
    label: methodLabels[method.method] ?? method.method.replaceAll("_", " "),
  }));
  const bestMethod = [...methods].sort((a, b) => Number(a.brier || 0) - Number(b.brier || 0))[0] ?? null;
  const eventNormalisation = dashboardData.eventStructure?.normalisation ?? null;
  return {
    queryLed: {
      available: Boolean(backtestData.available),
      forecastRows: backtestData.forecastRows,
      evaluatedRows: backtestData.evaluatedRows,
      targetMarkets: backtestData.targetMarkets,
      horizons: backtestData.horizons,
      leakageRows: backtestData.leakageRows,
      bestMethod,
      methods,
    },
    eventStructure: eventNormalisation ? {
      overall: eventNormalisation.overall,
      horizons: eventNormalisation.horizons,
    } : null,
  };
}

async function runForecastQuery(query) {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", [forecastBridgePath], {
      cwd: root,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(Object.assign(new Error("Forecast service timed out"), { code: "TIMEOUT" }));
    }, 120000);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve(JSON.parse(stdout));
        return;
      }
      reject(Object.assign(new Error(stderr || "Forecast service failed"), { code, stderr }));
    });
    child.stdin.end(JSON.stringify({ query }));
  });
}

async function handleApi(request, response, pathname) {
  if (pathname === "/api/overview" && request.method === "GET") {
    if (!(await currentUser(request))) {
      return jsonResponse(response, { error: "Sign in required" }, 401);
    }
    try {
      return jsonResponse(response, compactOverviewData(await readDashboardData()));
    } catch {
      return jsonResponse(response, { error: "Overview data unavailable" }, 500);
    }
  }

  if (pathname === "/api/results" && request.method === "GET") {
    if (!(await currentUser(request))) {
      return jsonResponse(response, { error: "Sign in required" }, 401);
    }
    try {
      return jsonResponse(response, compactResultsData(await readDashboardData(), await readBacktestData()));
    } catch {
      return jsonResponse(response, { error: "Results data unavailable" }, 500);
    }
  }

  if (pathname === "/api/session" && request.method === "GET") {
    const user = await currentUser(request);
    return jsonResponse(response, { authenticated: Boolean(user), user });
  }

  if (pathname === "/api/login" && request.method === "POST") {
    const payload = await requestBody(request);
    const email = String(payload.email || "").trim().toLowerCase();
    const password = String(payload.password || "");
    const users = await readUsers();
    const user = users.find((candidate) => candidate.email === email);
    if (!user || !verifyPassword(password, user)) {
      return jsonResponse(response, { error: "Invalid email or password" }, 401);
    }
    return jsonResponse(response, { authenticated: true, user: publicUser(user) }, 200, {
      "set-cookie": `${sessionCookie}=${createSession(user)}; Path=/; Max-Age=${sessionSeconds}; HttpOnly; SameSite=Lax`,
    });
  }

  if (pathname === "/api/register" && request.method === "POST") {
    const payload = await requestBody(request);
    const name = String(payload.name || "").trim();
    const email = String(payload.email || "").trim().toLowerCase();
    const password = String(payload.password || "");
    if (!name) {
      return jsonResponse(response, { error: "Name is required" }, 400);
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return jsonResponse(response, { error: "Valid email is required" }, 400);
    }
    if (password.length < 8) {
      return jsonResponse(response, { error: "Password must be at least 8 characters" }, 400);
    }
    const users = await readUsers();
    if (users.some((candidate) => candidate.email === email)) {
      return jsonResponse(response, { error: "An account already exists for that email" }, 409);
    }
    const user = {
      email,
      name,
      role: "free",
      ...hashPassword(password),
    };
    users.push(user);
    await writeUsers(users);
    return jsonResponse(response, { authenticated: true, user: publicUser(user) }, 201, {
      "set-cookie": `${sessionCookie}=${createSession(user)}; Path=/; Max-Age=${sessionSeconds}; HttpOnly; SameSite=Lax`,
    });
  }

  if (pathname === "/api/logout" && request.method === "POST") {
    return jsonResponse(response, { authenticated: false, user: null }, 200, {
      "set-cookie": `${sessionCookie}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax`,
    });
  }

  if (pathname === "/api/forecast/query" && request.method === "POST") {
    const user = await currentUser(request);
    if (!user) {
      return jsonResponse(response, { error: "Sign in required" }, 401);
    }
    if (user.role !== "premium") {
      return jsonResponse(response, { error: "Premium role required" }, 403);
    }
    const payload = await requestBody(request);
    const query = String(payload.query || "").trim();
    try {
      return jsonResponse(response, await runForecastQuery(query));
    } catch (error) {
      const stderr = String(error.stderr || "").trim();
      if (error.code === 2 && stderr) {
        try {
          return jsonResponse(response, JSON.parse(stderr), 400);
        } catch {
          return jsonResponse(response, { error: stderr }, 400);
        }
      }
      return jsonResponse(response, { error: "Forecast service failed" }, 500);
    }
  }

  return jsonResponse(response, { error: "Not found" }, 404);
}

async function serveStatic(response, pathname) {
  const routeFiles = {
    "/": "/index.html",
    "/workspace": "/workspace.html",
    "/data-analysis": "/dataset.html",
    "/dataset": "/dataset.html",
    "/results": "/results.html",
    "/backtesting": "/backtesting.html",
    "/forecast": "/forecast.html",
    "/forecast/evidence": "/forecast_evidence.html",
    "/forecast/result": "/forecast_result.html",
  };
  const safePath = routeFiles[pathname] || pathname;
  const absolutePath = path.normalize(path.join(siteDir, safePath));
  if (!absolutePath.startsWith(siteDir)) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }
  try {
    const body = await readFile(absolutePath);
    response.writeHead(200, {
      "content-type": mimeTypes[path.extname(absolutePath)] || "application/octet-stream",
    });
    response.end(body);
  } catch {
    response.writeHead(404);
    response.end("Not found");
  }
}

const { host, port } = parseArgs();
await ensureUserStore();

createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host || `${host}:${port}`}`);
    if (url.pathname.startsWith("/api/")) {
      await handleApi(request, response, url.pathname);
      return;
    }
    const protectedPages = new Set([
      "/workspace",
      "/workspace.html",
      "/data-analysis",
      "/dataset",
      "/dataset.html",
      "/results",
      "/results.html",
      "/backtesting",
      "/backtesting.html",
      "/forecast",
      "/forecast.html",
      "/forecast/evidence",
      "/forecast_evidence.html",
      "/forecast/result",
      "/forecast_result.html",
    ]);
    if (protectedPages.has(url.pathname) && !(await currentUser(request))) {
      response.writeHead(302, { location: "/" });
      response.end();
      return;
    }
    await serveStatic(response, url.pathname);
  } catch (error) {
    jsonResponse(response, { error: "Server error" }, 500);
  }
}).listen(port, host, () => {
  console.log(`Serving website at http://${host}:${port}/`);
});
