```markdown
# Customer-Facing Resolution Agent — Airline Disruption

Assignment 3, AGENTIC AI FACTORY | AIONOS. A support agent that resolves airline
disruption cases from the supplied Data Pack, enforces company policy limits
deterministically, and escalates to human agents the moment a request exceeds its
authority.

## One-command run

```bash
git clone <your-repo-url> && cd resolution_agent
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
add GEMINI_API_KEY="your_api_key_here"         # create a .env file
streamlit run app.py

```

Policy correctness is computationally provable with **zero LLM calls** and no API key:

```bash
pytest test_policy_and_escalation.py -v     # 21 passed, 0 LLM calls, <1s

```

---

## 1. Central Design Decision: The LLM Never Touches a Number

The failure mode this assignment tests is an agent that *sounds* right about money
while hallucinating compensation or violating operational boundaries. Large Language
Models are probabilistic text engines; they cannot reliably perform inequality arithmetic
or strictly respect policy ceilings.

To eliminate financial hallucination, the model is architecturally forbidden from deciding
entitlements, calculating amounts, or judging escalation thresholds:

| Layer | Decides | LLM? |
| --- | --- | --- |
| `data.py` | Ground-truth records & rule constants | No ($O(1)$ dict lookups) |
| `policy.py` | Entitlements, compensation tiers & downward bounds | No (pure Python) |
| `escalation.py` | Authority boundary validation (5 Prohibited checks) | No (pure Python) |
| `llm_nodes.py` | Intent classification & final natural-language phrasing | Yes (2 calls) |
| `graph.py` | Workflow orchestration & typed state isolation | No (LangGraph) |

By the time the model writes a response, entitlements and escalation decisions are
already frozen in state. The model's permitted knowledge is strictly limited to the
`policy_facts` injected into its context; the prompt explicitly mandates that any
detail absent from those facts must be answered with "I'll have that looked into."

**Why No Database / Vector Search (RAG)?**
Vector retrieval introduces semantic ambiguity and retrieval jitter. The Data Pack
consists of exactly 3 customers, 4 flight segments, and 5 service rules. Loading this
into an in-memory dictionary provides deterministic $O(1)$ lookup latency, zero vector
drift, and complete protection against retrieval hallucinations.

---

## 2. Architecture and Process Flow

```
                       ┌──────────────────┐
  customer message ───▶│ identify_customer│  PNR → customer + affected segment
                       └────────┬─────────┘
                    unknown PNR │ known PNR
              ┌─────────────────┴──────────────┐
              ▼                                ▼
   ┌────────────────────┐           ┌────────────────────┐
   │ ask_for_booking_ref│           │   extract_intent   │  LLM #1, temp 0.0, JSON
   └─────────┬──────────┘           └─────────┬──────────┘
             │                                ▼
             │                      ┌────────────────────┐
             │                      │ run_policy_engine  │  pure functions
             │                      └─────────┬──────────┘
             │                                ▼
             │                      ┌────────────────────┐
             │                      │  escalation_gate   │  5 checks, pure Python
             │                      └────┬──────────┬────┘
             │                escalate   │          │ in policy
             │                           ▼          ▼
             │              ┌──────────────────┐ ┌──────────────┐
             │              │ escalate_to_human│ │ compose_reply│  LLM #2, temp 0.3
             │              └────────┬─────────┘ └──────┬───────┘
             └───────────────────────┴─────────────────┬┘
                                                       ▼
                                             ┌──────────────┐
                                             │  log_action  │──▶ END
                                             └──────────────┘

```

### Execution Steps

1. **Identity Resolution (`identify_customer`):** Scans the input for a valid 6-character PNR. If found, it fetches the customer profile and disrupted segment.
2. **Clarification Branch (`ask_for_booking_ref`):** If no PNR is found, the graph routes directly to a static clarification question, bypassing the LLM entirely to satisfy the "ask only necessary questions" requirement without wasting tokens.
3. **Intent Extraction (`extract_intent`):** Gemini classifies customer input at Temperature 0.0 with native JSON schema enforcement into an unambiguous taxonomy.
4. **Deterministic Policy Execution (`run_policy_engine`):** A pure Python function computes entitlements (meal vouchers, lounge passes, hotel hours) based strictly on flight status.
5. **Authority & Escalation Gate (`escalation_gate`):** Cross-checks extracted intents and requested amounts against the 5 prohibited actions in pure code.
6. **Reply Composition (`compose_reply` / `escalate_to_human`):** Synthesizes warm, human-like responses using pre-approved policy facts. Entitlements on delayed flights are treated as additive, while cancellations are presented as an "OR" choice.
7. **Audit Logging (`log_action`):** Appends turn telemetry, executed rules, and escalation reasons into an immutable session state ledger.

---


## 3. Inputs, Sources, and Assumptions Used

### Verified Inputs & Sources

* **Data Pack Only:** Strictly grounded in the supplied profiles (Priya, Arvind, Meher), transaction segments (SK-204, Return, SK-118, SK-305), and Service Rules.
* **Temporal Anchor:** Set on Wednesday, 23 September 2026.
* **Sample Conversations:** Used solely to inform empathetic tone and style; never treated as a source of policy or rules.

### Explicit Engineering Assumptions

1. **Downward Boundary Resolution (Gaps in Policy):**
The written policy specifies delay tiers as "under 3 hours", "more than 3 hours", and "more than 5 hours" ($< 3$, $> 3$, $> 5$). Exactly 3.0h and 5.0h are literal gaps in the text. The engine deliberately resolves these boundaries downward (e.g., 3.0h receives a meal voucher only; 5.0h receives meal + lounge, no hotel) to prevent unauthorized disbursements. This decision is explicitly tracked in `PolicyResult.assumptions`.
2. **Strict Waiver Ceiling:**
"Above ₹1,500" is treated as a strict mathematical inequality ($> 1500.0$). An amount of ₹1,500.00 is autonomously waivable; ₹1,500.01 triggers mandatory supervisor escalation.
3. **Multi-Segment Booking Integrity:**
PNR `SK4821X` carries two segments (outbound cancelled, return unaffected). The data layer models bookings as `PNR -> list[segment]` to prevent dropping Priya's return flight.
4. **Cancellation Choice vs. Delay Additivity:**
Cancellation entitlements are mutually exclusive (rebooking *or* refund; customer picks). Delay entitlements are additive (vouchers, lounge passes, and hotel hours are awarded together, never as a false trade-off).
5. **Non-Financial Loyalty Tier Privileges:**
Gold and Platinum tiers grant priority seat-inventory access, but never justify fare waivers above ₹1,500 or extra cash compensation.

---

## 54. Security & Invariant Defenses

* **Prompt Injection Isolation:** Customer messages are wrapped inside `<customer_message>` XML tags. System instructions are passed via the native SDK `system_instruction` parameter, establishing a strict barrier between operational rules and untrusted user input.
* **Deterministic Fail-Safe Fallback:** If the Gemini API experiences network timeouts or a 503 Overload error, `_deterministic_fallback_reply()` formats the pre-computed policy facts into a clean, bulleted summary. The app remains functional without crashing.
* **Intent Taxonomy Contract:** Entitled actions (`refund_request`) are strictly separated from unauthorized demands (`compensation_beyond_policy`). Frustrated or angry customers asking for what they are legally owed are never falsely escalated.
* **Audit Trail Preservation:** Every intent label, rule evaluation, and tool execution is recorded in an append-only telemetry log visible in the Streamlit sidebar.
* **Secrets Protection:** Keys are managed via environment variables (`GEMINI_API_KEY`) and `.env`, keeping credentials out of version control.

---

## 5. Failure Modes Handled

| Failure Mode | System Behavior |
| --- | --- |
| Model API 503 / High Demand | Catches exception and falls back to deterministic rule template (`_deterministic_fallback_reply`) |
| Malformed / Fenced LLM Output | Regex JSON salvaging extracts valid dictionary payload; non-canonical labels are dropped |
| Unrecognized or Missing PNR | Clarification node asks for booking reference in a single prompt |
| Upstream Node Bypass | All state reads use `state.get()` with explicit defaults to prevent graph `KeyErrors` |
| Streamlit Turn Rerun | Session state preserves only resolved identity (`booking_ref`), preventing stale responses from leaking across turns |

---

## 6. AI Tools Used and How

| Tool | Implementation Role | Justification |
| --- | --- | --- |
| **Google Gemini (`gemini-3.5-flash-lite`)** | Natural Language Processing | Used for intent extraction (Temp 0.0, strict JSON schema) and reply synthesis (Temp 0.3, fact-constrained). Never used for policy decisions. |
| **LangGraph** | Workflow Orchestration | Implements a deterministic cyclic state machine, enforcing execution boundaries between NLU, policy computation, and response generation. |
| **Pydantic (v2)** | Data Validation | Enforces strict typing on extracted intents, preventing unstructured text from polluting downstream nodes. |
| **Streamlit** | Presentation & Telemetry | Dual-viewport interface providing live customer chat alongside real-time state machine telemetry and action logging. |

---

## 7. Project Structure

```
airline-resolution-agent/
├── data.py                        # Ground-truth profiles, flight segments & constants
├── policy.py                      # Pure Python policy engine (zero LLM calls)
├── escalation.py                  # 5-point prohibited action validator
├── llm_nodes.py                   # GenAI SDK integration (NLU parsing & NLG phrasing)
├── graph.py                       # LangGraph wiring, conditional routing & state definitions
├── app.py                         # Streamlit UI with live action record
├── test_policy_and_escalation.py  # 21-test deterministic verification suite
├── requirements.txt               # Project dependencies
└── README.md                      # Architecture, setup & defense documentation

```

```

```
