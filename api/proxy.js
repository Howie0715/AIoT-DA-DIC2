// Vercel Serverless Function: GET /api/proxy?url=<csv url>
// Fetches a public CSV on behalf of the browser when the source does not send CORS headers.
const dns = require("node:dns").promises;
const net = require("node:net");

const MAX_BYTES = 10 * 1024 * 1024;

function isPrivateAddress(ip) {
  if (net.isIPv4(ip)) {
    const [a, b] = ip.split(".").map(Number);
    return a === 10 || a === 127 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127);
  }
  const v6 = ip.toLowerCase();
  return v6 === "::1" || v6 === "::" || v6.startsWith("fc") || v6.startsWith("fd") || v6.startsWith("fe80") || v6.startsWith("::ffff:");
}

module.exports = async (req, res) => {
  const fail = (code, msg) => res.status(code).setHeader("Content-Type", "text/plain; charset=utf-8").send(msg);

  let target;
  try {
    target = new URL(String(req.query.url || ""));
  } catch {
    return fail(400, "Invalid url parameter.");
  }

  try {
    // Follow redirects manually so every hop is checked against private addresses.
    let upstream;
    for (let hop = 0; ; hop++) {
      if (!["http:", "https:"].includes(target.protocol)) return fail(400, "Only http(s) URLs are allowed.");
      const addrs = await dns.lookup(target.hostname, { all: true }).catch(() => []);
      if (!addrs.length) return fail(400, `Cannot resolve host ${target.hostname}.`);
      if (addrs.some((a) => isPrivateAddress(a.address))) return fail(400, "Private or local addresses are not allowed.");
      upstream = await fetch(target, { redirect: "manual", headers: { "User-Agent": "regression-lab-proxy" }, signal: AbortSignal.timeout(15000) });
      const location = upstream.headers.get("location");
      if (upstream.status < 300 || upstream.status >= 400 || !location) break;
      if (hop >= 4) return fail(502, "Too many redirects.");
      target = new URL(location, target);
    }
    if (!upstream.ok) return fail(502, `Upstream responded HTTP ${upstream.status}.`);
    const declared = Number(upstream.headers.get("content-length") || 0);
    if (declared > MAX_BYTES) return fail(413, "File is larger than 10 MB.");
    const buf = Buffer.from(await upstream.arrayBuffer());
    if (buf.length > MAX_BYTES) return fail(413, "File is larger than 10 MB.");
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader("Cache-Control", "public, s-maxage=3600, max-age=300");
    return res.status(200).send(buf);
  } catch (e) {
    return fail(502, `Fetch failed: ${e.message}`);
  }
};
