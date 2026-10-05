// Gateway overhead: the same small chat request against the mock LLM directly and through the
// gateway. Env: BASE_URL, KEY, VUS, DURATION. Run via scripts/overhead.py.
import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    steady: { executor: "constant-vus", vus: __ENV.VUS ? Number(__ENV.VUS) : 10, duration: __ENV.DURATION || "30s" },
  },
  summaryTrendStats: ["avg", "p(50)", "p(90)", "p(99)", "max"],
};

const body = JSON.stringify({
  model: "mock-small",
  messages: [{ role: "user", content: "hello" }],
  max_tokens: 16,
});

export default function () {
  const res = http.post(`${__ENV.BASE_URL}/v1/chat/completions`, body, {
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${__ENV.KEY || "x"}` },
  });
  check(res, { "status 200": (r) => r.status === 200 });
}
