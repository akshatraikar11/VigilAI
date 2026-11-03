# OrchestAI — Multi-Agent Task Automation Platform using MCP (Model Context Protocol)

**Final Year B.E. Information Technology — Major Project Synopsis**
**Author:** Akshat | Atharva College of Engineering, Mumbai
**Project Name:** OrchestAI

---

## 1. Problem Statement

Existing workflow automation tools (Zapier, n8n, Make) operate on rigid trigger→action rules and cannot handle tasks requiring judgment, multi-step reasoning, or adaptive decision-making. Meanwhile, most AI-powered "assistant" projects are stateless chatbots that simply call an LLM API and return text — they don't autonomously execute actions across tools, validate their own outputs, or expose their reasoning process.

There is no lightweight, transparent system that lets a user define a task in natural language and have a set of coordinated AI agents plan, execute, and validate multi-step actions across real tools — with safety checks before any consequential action is taken.

## 2. Objectives

1. Design a multi-agent architecture (Planner → Executor → Validator) that decomposes a natural-language task into tool-executable steps
2. Integrate MCP (Model Context Protocol) to standardize how agents discover and call external tools
3. Implement a human-approval gate for sensitive/irreversible actions
4. Provide a real-time execution trace showing agent reasoning, tool calls, and validation outcomes
5. Ensure reliability via iteration limits, retry logic, and self-correction (re-planning on validation failure)

## 3. Scope (Trimmed / Buildable)

**In scope:**
- Fixed set of 4–6 MCP tool servers (mocked email + calendar, real GitHub API, free web search)
- Config/form-based workflow definition (no drag-drop canvas)
- Hand-rolled agent loop (not a black-box framework) for full explainability
- Execution trace viewer + approval queue + basic analytics

**Explicitly out of scope (mention as "future work" in report):**
- Public tool marketplace / billing
- Drag-and-drop visual agent builder
- Multi-tenant/org-level auth
- Docker-sandboxed arbitrary code execution

## 4. Core Features

| Feature | Description |
|---|---|
| Natural-language task input | User types a goal ("follow up on unpaid invoices"); Planner agent decomposes it into steps |
| Planner–Executor–Validator loop | Planner creates a plan → Executor calls tools via MCP → Validator checks output correctness before proceeding |
| Self-correction | If Validator rejects a step, Planner re-plans instead of failing outright |
| Human-approval gate | Sensitive actions (e.g., sending an email) pause for explicit user confirmation |
| Execution trace viewer | Step-by-step visibility into agent reasoning, tool calls, inputs/outputs, and validation verdicts |
| Tool connection manager | Add/remove MCP tool servers, manage auth tokens |
| Analytics dashboard | Success/failure rate per workflow, avg steps per task, most-used tools *(lower priority — build last)* |
| Retry & iteration limits | Prevents infinite loops / runaway API usage |

## 5. Modules / Pages

1. **Auth** (login/signup)
2. **Dashboard** — overview of workflows, recent runs, quick stats
3. **Workflow Config** — define a task, select allowed tools, set approval rules
4. **Execution Trace Viewer** *(highest demo value — invest most UI effort here)* — live/replay view of agent steps, tool calls, validator decisions
5. **Approval Queue** — pending sensitive actions awaiting user confirmation
6. **Tool Connections** — manage MCP tool servers and credentials
7. **Analytics** — success/failure rates, latency, cost/token usage per run *(build last; a simple run history table is sufficient if time runs short)*
8. **Logs/History** — past workflow runs, searchable

## 6. Agent Architecture

```
User Input (natural language task)
        │
        ▼
   ┌─────────┐
   │ Planner │  → decomposes task into ordered steps, selects tools
   └────┬────┘
        ▼
   ┌──────────┐
   │ Executor │  → calls MCP tool server, gets result
   └────┬─────┘
        ▼
   ┌───────────┐
   │ Validator │  → checks result correctness/safety
   └────┬──────┘
        │
   pass │ fail → re-plan (back to Planner, max N retries)
        ▼
 Sensitive action? → Approval Queue (human confirms) → Execute
        ▼
   Log + Trace stored → next step or task complete
```

### 6.1 LLM Structured Output Strategy

Reliable JSON output from the LLM is the #1 integration risk. The following strategy is used:

| Technique | Purpose |
|---|---|
| **JSON schema mode (strict)** | Use Groq's `response_format: { type: "json_schema", json_schema: {...}, strict: true }` to guarantee valid, schema-compliant JSON output |
| **Strict schema validation** | Every LLM response is validated against a Zod/JSON schema before processing; malformed responses trigger a retry |
| **Retry with error feedback** | On parse failure, re-prompt the LLM with the validation error message (max 3 retries per step) |
| **Fallback templates** | If retries exhaust, use a safe default action (e.g., "skip step and log error") rather than crashing |

## 7. Tech Stack (Fully Free)

| Layer | Choice | Why |
|---|---|---|
| Frontend | React (MERN) | Existing strength |
| Backend | Node.js + Express | Existing strength |
| Database | MongoDB Atlas (free 512MB tier) | Sufficient for this scope |
| LLM (primary) | **Groq API** (free tier) | Fast inference — critical for live agent-loop demo |
| LLM (fallback/offline) | **Ollama** (local, Llama 3.1 8B / Qwen2.5) | Zero rate-limit risk during viva demo |
| LLM (alt free option) | NVIDIA NIM (build.nvidia.com, 40 RPM free) | Backup provider |
| Tool protocol | MCP SDK | Core novelty of the project |
| Mocked tools | Simulated email/calendar (own DB) | Avoids Gmail/Calendar OAuth verification overhead |
| Real tool (kept live) | GitHub REST API (free personal token) | Easy setup, real integration for demo credibility |
| Web search tool | Tavily API (free tier, 1000 credits/month) | Free, reliable, agent-friendly JSON search |
| Hosting (frontend) | Vercel (free Hobby tier) | Always-on, optimized for React |
| Hosting (backend) | Render (free tier) | Persistent Node.js service; use UptimeRobot (free) cron ping every 14 min to prevent cold-start spin-down during demo |

**Total cost: ₹0**

> **⚠️ Ollama hardware note:** Llama 3.1 8B (Q4 quantized) needs ~5 GB disk + 8 GB RAM minimum (16 GB recommended). If your demo laptop has only 8 GB total RAM, use **Qwen2.5 3B** or **Phi-3 Mini** instead — they run on 4 GB RAM. **Test on the actual demo machine by Week 3.**

## 8. Novelty / Differentiation (for viva)

> "This isn't a new product category — it's an applied systems project demonstrating agentic orchestration patterns (Planner-Executor-Validator) using MCP, with a focus on reliability and observability, which off-the-shelf no-code tools (Zapier/n8n) don't provide and most framework-based student agent projects don't bother implementing."

| Compared to | Differentiation |
|---|---|
| **Zapier / n8n** | Rule-based automation vs. **reasoning-based** automation |
| **LangChain / CrewAI / AutoGPT** | Hand-rolled orchestration logic (fully explainable) instead of an imported black box |
| **Typical student agent projects** | Observability layer (execution trace) — rare even in hobbyist projects |
| **All of the above** | Failure-handling as first-class feature (validator + self-correction + approval gate) — the "responsible AI" angle |

## 9. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| LLM makes wrong tool call / hallucinates | High | Validator agent + strict JSON schema for tool calls + retry cap (max 3 per step) |
| LLM returns malformed / unparseable output | High | JSON mode enforcement + Zod schema validation + retry with error feedback + fallback templates |
| Free-tier rate limits stall live demo | Medium | Groq (high throughput) primary, Ollama local fallback — test both before demo |
| Infinite agent loop | Medium | Hard iteration cap (max 6 steps per task) + per-step timeout |
| Ollama too slow / doesn't run on demo machine | Medium | Test on actual demo hardware by Week 3; keep Groq + NVIDIA NIM as online alternatives |
| OAuth setup overhead (Gmail/Calendar) | Low | Mock these two tools; keep GitHub + search real |
| Scope creep (analytics, extra UI polish) | Medium | Analytics is the last module built; a simple run history table is the minimum viable version |

## 10. Timeline (12 weeks — revised)

| Week | Milestone | Notes |
|---|---|---|
| 1–2 | Finalize scope, synopsis approval, repo setup, MERN boilerplate | — |
| 3 | Auth, DB schema, basic dashboard UI | **Also:** test Ollama on demo machine this week |
| 4 | MCP SDK setup + first mocked tool server (email) | MCP moved earlier — it's the core novelty |
| 5–6 | Agent loop v1 (Planner + Executor + MCP) — **one end-to-end happy path working** | Target demo: "Star a GitHub repo + search for its README" fully working |
| 7 | Validator + self-correction + retry logic | Add the fail→re-plan loop |
| 8 | Approval gate + remaining MCP tool servers (calendar, search) | Human-in-the-loop flow complete |
| 9–10 | **Execution trace viewer** (live + replay) | This is the most demo-worthy feature — allocate two full weeks |
| 11 | Analytics dashboard + logs/history + deployment | Analytics is nice-to-have; a simple table of past runs is sufficient |
| 12 | Testing, guardrails, demo hardening, report writing, viva prep | Prepare 3 canned demo workflows |

> **Key change from original timeline:** MCP integration moved from Week 9 to Week 4. The headline feature of the project should not be left to the last third of the timeline.

## 11. Demo Script (prepare 3 scenarios)

| # | Scenario | What it demonstrates |
|---|---|---|
| 1 | **Happy path:** "Find my top-starred GitHub repo and search the web for its latest issues" | Full Planner → Executor → Validator loop with real tools (GitHub API + web search) |
| 2 | **Validator rejection → re-plan:** Give a task with an intentionally ambiguous step that the Validator rejects; show the Planner re-planning and succeeding on retry | Self-correction, reliability, observability |
| 3 | **Approval gate:** "Send a follow-up email to unpaid invoices" — execution pauses at the email-send step, user approves in the Approval Queue, then execution continues | Human-in-the-loop safety, mocked email tool via MCP |

> **Tip for viva:** Walk the examiner through the Execution Trace Viewer during each demo. The trace is the visual proof that this is not a black-box chatbot.

## 12. Expected Outcome

A working, deployed multi-agent task automation system where a user submits a natural-language goal, watches agents plan/execute/validate steps in real time across real and mocked tools, approves sensitive actions, and reviews analytics on run performance — built entirely on free-tier infrastructure.

## 13. Prepared Viva Q&A

| Question | Answer |
|---|---|
| "What's novel about this?" | "The observability layer. Most student agent projects call an LLM and print the response. This one shows you every intermediate reasoning step, tool call, and validation verdict in a real-time trace viewer." |
| "Why not use LangChain / CrewAI?" | "Hand-rolled loop gives full explainability — every decision is traceable in the execution log. Framework-based agents are black boxes; I can't explain their internal routing to an examiner." |
| "How do you handle LLM errors?" | "Three layers: (1) strict JSON schema validation on every LLM response, (2) Validator agent that checks tool call correctness, (3) re-planning on failure with a hard retry cap to prevent infinite loops." |
| "Why mock email/calendar?" | "Gmail and Google Calendar require OAuth app verification (weeks of review). Mocking lets me demonstrate the same MCP integration pattern without external approval delays. GitHub and web search are kept real to prove live API integration." |
