const { spawn } = require("child_process");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const API_PATH = "/__audit_api__/delete-audit";
const BODY_LIMIT_BYTES = 64 * 1024;

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Content-Length", Buffer.byteLength(body));
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;

    req.on("data", chunk => {
      size += chunk.length;
      if (size > BODY_LIMIT_BYTES) {
        reject(new Error("request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", reject);
  });
}

function runAuditDelete(href) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      "uv",
      ["run", "python", "tools/audits/audit_nav.py", "delete", "--href", href],
      {
        cwd: PROJECT_ROOT,
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",
          PYTHONPATH: process.env.PYTHONPATH || "src",
          UV_CACHE_DIR: process.env.UV_CACHE_DIR || "agents/temp/uv-cache",
        },
        windowsHide: true,
      },
    );
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", chunk => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", chunk => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", code => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || stdout.trim() || `audit delete failed with exit code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`audit delete returned invalid JSON: ${error.message}`));
      }
    });
  });
}

module.exports = async function auditMiddleware(req, res, next) {
  const requestPath = req.url ? req.url.split("?")[0] : "";
  if (requestPath !== API_PATH) {
    next();
    return;
  }

  if (req.method === "OPTIONS") {
    sendJson(res, 204, {});
    return;
  }

  if (req.method !== "POST") {
    sendJson(res, 405, { ok: false, error: "method not allowed" });
    return;
  }

  try {
    const rawBody = await readBody(req);
    const payload = rawBody ? JSON.parse(rawBody) : {};
    const href = typeof payload.href === "string" ? payload.href : "";
    if (!href) {
      sendJson(res, 400, { ok: false, error: "missing href" });
      return;
    }
    const result = await runAuditDelete(href);
    sendJson(res, 200, { ok: true, ...result });
  } catch (error) {
    sendJson(res, 500, {
      ok: false,
      error: error && error.message ? error.message : String(error),
    });
  }
};
