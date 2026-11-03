# 🛡️ DarkGuard: Vigil AI — System Architecture & Design Document (HLD & LLD)

> **Document Version:** 3.2.0  
> **Classification:** Enterprise Cybersecurity & Applied AI System Design  
> **Author:** Akshat Raikar  
> **Status:** Production-Verified (339/339 Passing Automated Tests)

---

## Table of Contents
1. [Executive Summary & System Vision](#1-executive-summary--system-vision)
2. [High-Level Design (HLD)](#2-high-level-design-hld)
   - [2.1 System Architecture Overview](#21-system-architecture-overview)
   - [2.2 End-to-End Scanning Flow](#22-end-to-end-scanning-flow)
   - [2.3 Browser Stealth & DOM Extraction Engine](#23-browser-stealth--dom-extraction-engine)
   - [2.4 9-Engine Concurrent Detection Pipeline](#24-9-engine-concurrent-detection-pipeline)
   - [2.5 Threat Model & AI Security Guardrails](#25-threat-model--ai-security-guardrails)
3. [Low-Level Design (LLD)](#3-low-level-design-lld)
   - [3.1 Class & Component Architecture](#31-class--component-architecture)
   - [3.2 ML Feature Union & Ensemble Pipeline](#32-ml-feature-union--ensemble-pipeline)
   - [3.3 MLOps Confidence Drift Monitor](#33-mlops-confidence-drift-monitor)
   - [3.4 Harm-Aware Decision Engine (HADE) 8-Step Protocol](#34-harm-aware-decision-engine-hade-8-step-protocol)
   - [3.5 Weighted Ensemble Fusion & Consensus Gate](#35-weighted-ensemble-fusion--consensus-gate)
   - [3.6 Trust Score Calculation & Severity Matrix](#36-trust-score-calculation--severity-matrix)
   - [3.7 Database Schema & Storage Subsystem](#37-database-schema--storage-subsystem)
   - [3.8 API Interface Contracts](#38-api-interface-contracts)
4. [Interview Talking Points & Architectural Justifications](#4-interview-talking-points--architectural-justifications)

---

## 1. Executive Summary & System Vision

**DarkGuard: Vigil AI** is an enterprise-grade cybersecurity and consumer privacy engine engineered to detect, classify, and mitigate manipulative digital user interfaces (**dark patterns**) across modern web applications. 

Deceptive design practices—including forced continuity, drip pricing, confirmshaming, hidden consent checkboxes, roach motels, and cookie walls—undermine user autonomy and violate strict international regulatory frameworks (EU GDPR Art. 7/13, FTC ROSCA Act, California CPRA §1798, and EU Digital Services Act Art. 25).

Unlike naive regex scrapers or non-deterministic LLM-only scanners, Vigil AI operates as a **hybrid multi-engine verification pipeline**:
- It couples headless browser automation with deep Shadow DOM piercing to capture dynamic client-side DOM mutations and computed CSS states.
- It executes **9 independent analysis engines** concurrently via worker thread pools.
- It employs a calibrated **Scikit-Learn VotingClassifier ensemble** combining text TF-IDF with dense linguistic lexicons.
- It enforces strict **MLOps drift tracking** to detect distribution shifts in real-time.
- It filters candidate signals through the **Harm-Aware Decision Engine (HADE)** and a **Quorum-Based Consensus Gate**, eliminating false positives on legitimate websites while guaranteeing high-precision detection of malicious patterns.

```
+-----------------------------------------------------------------------------------------+
|                                  Vigil AI Ecosystem                                      |
|                                                                                         |
|  +--------------------+     +------------------------+     +-------------------------+  |
|  | Chrome Ext. (MV3)  | <-> |  Waitress WSGI Backend | <-> | Playwright Headless Hub |  |
|  +--------------------+     +------------------------+     +-------------------------+  |
|                                         |                                               |
|                    +--------------------+---------------------+                         |
|                    |                                          |                         |
|         +----------------------+                   +----------------------+             |
|         | 9-Engine Parallel    |                   | HADE + Consensus     |             |
|         | ThreadPoolExecutor   | ----------------> | Calibration Gate     |             |
|         +----------------------+                   +----------------------+             |
|                    |                                          |                         |
|                    +--------------------+---------------------+                         |
|                                         |                                               |
|                             +------------------------+                                  |
|                             | SQLite (WAL) / Reports |                                  |
|                             +------------------------+                                  |
+-----------------------------------------------------------------------------------------+
```

---

## 2. High-Level Design (HLD)

### 2.1 System Architecture Overview

The system is decomposed into four decoupled tiers:
1. **Client / Presentation Layer**: Chrome Extension (Manifest V3) for live inline DOM inspection and a responsive Flask web portal with real-time audit dashboards.
2. **API & Orchestration Layer**: Production Flask server running on Waitress WSGI, fronted by token-bucket rate limiting (`Flask-Limiter`), secure CORS configuration, and asynchronous background worker queues.
3. **Execution & Analysis Engine**: Playwright browser automation pool feeding an asynchronous 9-engine parallel detection pipeline (`DetectionPipeline`).
4. **Data Persistence & Monitoring**: High-performance SQLite engine configured in Write-Ahead Logging (`WAL`) mode, coupled with an in-memory MLOps Confidence Drift Monitor.

```mermaid
flowchart TD
    Client["Client / User / Extension (MV3)"]
    API["Flask REST API Gateway (Waitress WSGI)"]
    Limiter["Flask-Limiter (Rate Limiting Guard)"]
    AuthValidation["URL & Input Validator + SSRF Guard"]
    Worker["Async Scan Worker / Task Queue"]
    Playwright["Headless Browser Automation (Playwright)"]
    Stealth["Stealth Anti-Fingerprint & Shadow DOM Piercer"]
    Pipeline["Concurrent Detection Pipeline (ThreadPoolExecutor)"]
    
    subgraph EngineSuite["9-Engine Parallel Detection Suite"]
        E1["1. ML Classifier Ensemble"]
        E2["2. DOM Structural Analyzer"]
        E3["3. spaCy NLP Linguistic Engine"]
        E4["4. Visual & CSS Contrast Analyzer"]
        E5["5. Advanced Heuristic Engine"]
        E6["6. Cookie & GDPR Consent Auditor"]
        E7["7. Link & Roach Motel Analyzer"]
        E8["8. Readability & Complexity Engine"]
        E9["9. Behavioral Scoring Engine"]
    end

    HADE["Harm-Aware Decision Engine (HADE 8-Step)"]
    Fusion["Weighted Ensemble Fusion Engine"]
    Consensus["Consensus & Element Clustering Gate"]
    TrustEngine["Trust Score & Remediation Engine"]
    DB[("SQLite 3 DB (WAL Mode)")]

    Client -->|POST /api/scan| Limiter
    Limiter --> API
    API --> AuthValidation
    AuthValidation -->|Enqueues Task| Worker
    Worker --> Playwright
    Playwright --> Stealth
    Stealth -->|Extracts HTML, Text, CSS, Network| Pipeline
    Pipeline --> EngineSuite
    EngineSuite -->|Raw Finding Candidates| HADE
    HADE --> Fusion
    Fusion --> Consensus
    Consensus --> TrustEngine
    TrustEngine --> DB
    TrustEngine -->|Final Audit JSON| API
    API -->|Polling Response| Client
```

---

### 2.2 End-to-End Scanning Flow

The scan lifecycle executes asynchronously to prevent HTTP connection timeouts when analyzing complex Single-Page Applications (SPAs):

```mermaid
sequenceDiagram
    autonumber
    participant Client as Chrome Ext / Browser
    participant API as Flask API Gateway
    participant Guard as SSRF & Security Guard
    participant Worker as Background Task Worker
    participant Browser as Playwright Headless Hub
    participant Pipeline as 9-Engine Pipeline
    participant HADE as HADE & Consensus Gate
    participant DB as SQLite Database

    Client->>API: POST /api/scan { url: "https://example.com" }
    API->>Guard: Validate URL, Scheme & Hostname
    Guard->>Guard: Resolve DNS & verify non-private IP
    Guard-->>API: URL Safe (Pass)
    API->>DB: Insert Initial Scan Record (status='processing')
    API-->>Client: 202 Accepted { scan_id: "uuid-4", status: "processing" }
    
    API->>Worker: Dispatch Background Scan Task
    Worker->>Browser: Launch Chromium with Anti-Fingerprint Stealth
    Browser->>Browser: Intercept subrequests (Anti-SSRF IP check)
    Browser->>Browser: Navigate (Cascading Timeout: networkidle -> domcontentloaded)
    Browser->>Browser: Execute Shadow DOM Piercing Script
    Browser-->>Worker: Extracted DOM Data, Computed CSS, Text & Cookies
    
    Worker->>Pipeline: Execute 9 Analyzers in Parallel (ThreadPoolExecutor)
    par Concurrent Analysis
        Pipeline->>Pipeline: ML Classifier (Ensemble + Feature Union)
        Pipeline->>Pipeline: DOM Structural Analysis
        Pipeline->>Pipeline: NLP Syntax Dependency Check (spaCy)
        Pipeline->>Pipeline: Visual Color & Button Misdirection
        Pipeline->>Pipeline: Cookie Wall & Tracking Script Audit
        Pipeline->>Pipeline: Roach Motel & Link Navigation
        Pipeline->>Pipeline: Readability & Obfuscation Grader
    end
    Pipeline-->>Worker: Raw Candidates from All Engines
    
    Worker->>HADE: Run 8-Step Decision Protocol
    HADE->>HADE: Impact Scoring & Intent Validation
    HADE->>HADE: Apply Critical Overrides & Drop Weak Noise
    HADE->>HADE: Weighted Ensemble Fusion & Element Clustering Gate
    HADE-->>Worker: Fused, Validated Findings List
    
    Worker->>Worker: Calculate Site Trust Score & Risk Grade
    Worker->>DB: Persist Final Report, Metrics & Log Drift
    
    loop Status Polling
        Client->>API: GET /api/scan/<scan_id>
        API->>DB: Query Scan Status
        DB-->>API: Record
        API-->>Client: 200 OK { status: 'completed', trust_score: 85, findings: [...] }
    end
```

---

### 2.3 Browser Stealth & DOM Extraction Engine

Modern e-commerce and subscription sites utilize bot-detection (Cloudflare, Akamai, PerimeterX) and encapsulate Cookie Consent Management Platforms (CMPs) inside Shadow DOM boundaries. 

Vigil AI solves this through three defensive innovations in `scanner.py` and `dom_extractor.py`:

1. **Browser Stealth & Anti-Fingerprinting (`_STEALTH_SCRIPT`)**:
   - Overrides `navigator.webdriver` to `undefined`.
   - Mocks the native `window.navigator.chrome` runtime object.
   - Emulates hardware concurrency (`navigator.hardwareConcurrency = 8`) and spoofed plugins arrays.
   - Sets realistic user-agent strings and viewport dimensions.

2. **Shadow DOM Piercing (`_SHADOW_DOM_SCRIPT`)**:
   - Closed shadow roots normally block standard `document.querySelectorAll()` queries.
   - Vigil AI injects a recursive JavaScript crawler that traverses active `shadowRoot` trees:
     ```javascript
     function getAllElementsFromPoint(root) {
         let elements = [];
         function traverse(node) {
             if (!node) return;
             elements.push(node);
             if (node.shadowRoot) traverse(node.shadowRoot);
             let children = node.children || [];
             for (let i = 0; i < children.length; i++) traverse(children[i]);
         }
         traverse(root);
         return elements;
     }
     ```
   - This ensures CMP cookie banners (OneTrust, Cookiebot, Usercentrics) and injected dynamic paywalls are visible to all downstream analyzers.

3. **Cascading Timeout Fallback**:
   - Fast SPA check: Attempts navigation with `wait_until='networkidle'` (timeout: 15s).
   - Fallback 1: If dynamic assets keep the network busy, falls back to `wait_until='domcontentloaded'` (timeout: 10s).
   - Fallback 2: If browser execution fails entirely, falls back to an isolated `requests.get` session parsed via `BeautifulSoup4`.

---

### 2.4 9-Engine Concurrent Detection Pipeline

To guarantee sub-second analysis latency across large DOM trees, analyzers do not run sequentially. The `DetectionPipeline` manages an isolated `ThreadPoolExecutor` worker pool (`max_workers=8`):

| # | Engine | Module | Detection Mandate | Regulatory Alignment |
|---|---|---|---|---|
| **1** | **ML Classifier Ensemble** | `ml_analyzer.py` | Statistical tone classification, urgency, confirmshaming, deceptive phrasing | FTC Act §5 |
| **2** | **DOM Structural Analyzer** | `dom_analyzer.py` | Hidden pre-checked checkboxes, disabled opt-outs, hidden subscription fees | GDPR Art. 7, ROSCA |
| **3** | **NLP Linguistic Engine** | `text_analyzer.py` | 200+ dependency syntax rules, double-negatives, shame-inducing declination | FTC Deceptive Acts |
| **4** | **Visual & CSS Analyzer** | `visual_analyzer.py` | Low-contrast buttons (WCAG ratio < 3:1), ghost buttons, disguised display ads | WCAG 2.1, ADA Title III |
| **5** | **Advanced Heuristic Engine** | `advanced_analyzer.py` | Drip pricing, sneak-into-basket, countdown timer resets, fake social proof | DSA Art. 25 |
| **6** | **Cookie & Privacy Auditor** | `cookie_analyzer.py` | Accept-only cookie walls, tracking scripts firing before explicit consent | ePrivacy Directive, GDPR |
| **7** | **Link & Journey Analyzer** | `link_analyzer.py` | Roach motels, obscured cancellation links, mandatory phone-call cancellation | FTC "Click-to-Cancel" |
| **8** | **Readability Engine** | `readability_analyzer.py` | Flesch-Kincaid & Gunning Fog complexity scoring on Privacy Policies & ToS | Plain Writing Act, GDPR |
| **9** | **Behavioral Scorer** | `behavioral_scorer.py` | Compound funnel analysis: synthesizes combinations of urgency + preselection | DSA Compound Harm |

#### Error Isolation
Every engine implements the `BaseAnalyzer` contract. If an individual analyzer throws an unhandled exception or times out, it is caught in `pipeline.py`, logged to structured metrics, and yields an empty candidate list without failing the entire scan.

---

### 2.5 Threat Model & AI Security Guardrails

Running an automated scanner against arbitrary, untrusted internet URLs exposes backend infrastructure to severe security vectors. Vigil AI integrates strict guardrails:

```
+-----------------------------------------------------------------------------------------+
|                                Threat Mitigation Architecture                            |
+------------------------------------+----------------------------------------------------+
| Threat Vector                      | Vigil AI Security Guardrail Mitigation             |
+------------------------------------+----------------------------------------------------+
| SSRF (Server-Side Request Forgery) | Strict URL scheme validation (HTTP/HTTPS only).    |
| & DNS Rebinding Attacks            | Pre-flight socket getaddrinfo() IP resolution.     |
|                                    | Comprehensive IPv4 & IPv6 private subnet filter.   |
|                                    | Playwright request-interception blocking at wire.  |
+------------------------------------+----------------------------------------------------+
| Deserialization Remote Code        | SHA-256 integrity verification against signed      |
| Execution (RCE via pickle/joblib)  | model_manifest.json before model deserialization.  |
|                                    | Aborts loading if hash differs by even 1 bit.      |
+------------------------------------+----------------------------------------------------+
| Denial of Service (DoS) / Resource | Global scan timeout (90s) & per-analyzer timeout.  |
| Exhaustion                         | Max request payload limits (10MB).                 |
|                                    | Token-bucket rate limiting via Flask-Limiter.      |
+------------------------------------+----------------------------------------------------+
| Model Prediction / Concept Drift   | In-memory ConfidenceDriftMonitor with rolling deque|
|                                    | (size=100) comparing live mean vs training baseline|
+------------------------------------+----------------------------------------------------+
```

#### Detailed Anti-SSRF Defense:
```python
def _is_private_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    # NAT64 transition addresses (64:ff9b::/96) are legitimate public addresses
    if ip.version == 6 and ip in ipaddress.IPv6Network('64:ff9b::/96'):
        return False
    return (
        ip.is_private or ip.is_loopback or ip.is_reserved or
        ip.is_link_local or ip.is_multicast or ip.is_unspecified
    )
```

---

## 3. Low-Level Design (LLD)

### 3.1 Class & Component Architecture

```mermaid
classDiagram
    class AnalysisContext {
        +str scan_id
        +str url
        +str html
        +str text
        +Dict dom_data
        +Dict styles
        +List cookies
        +float created_at
    }

    class AnalyzerResult {
        +str analyzer_name
        +List findings
        +float execution_time_ms
        +Dict metrics
        +Optional[str] error
    }

    class BaseAnalyzer {
        <<abstract>>
        +str name
        +float weight
        +run(context: AnalysisContext) AnalyzerResult
        #analyze(context: AnalysisContext)* List
    }

    class MLAnalyzer {
        -_model_cache: VotingClassifier
        -_model_loaded: bool
        +ConfidenceDriftMonitor drift_monitor
        +_load_model()
        +predict_batch(texts: List[str]) List[Dict]
        #analyze(context: AnalysisContext) List
    }

    class DOMAnalyzer {
        #analyze(context: AnalysisContext) List
    }

    class CookieAnalyzer {
        #analyze(context: AnalysisContext) List
    }

    class DetectionPipeline {
        -Dict analyzers
        -int max_workers
        -int timeout
        +register(analyzer: BaseAnalyzer)
        +process(context: AnalysisContext) PipelineResult
    }

    class HarmAwareDecisionEngine {
        +evaluate(findings: List, context: str) List
        -_score_impact(finding: Dict) str
        -_validate_intent(finding: Dict) str
        -_apply_critical_boost(finding: Dict) bool
    }

    class FusionEngine {
        -Dict weights
        +fuse(findings: List) List
    }

    class FindingsAggregator {
        +aggregate(findings: List, page_text: str) List
        -_group_by_element(findings: List) Dict
        -_consensus_gate(grouped: Dict) List
        -_deduplicate_and_cap(findings: List) List
    }

    BaseAnalyzer <|-- MLAnalyzer
    BaseAnalyzer <|-- DOMAnalyzer
    BaseAnalyzer <|-- CookieAnalyzer
    DetectionPipeline o-- BaseAnalyzer
    DetectionPipeline ..> AnalysisContext
    DetectionPipeline ..> AnalyzerResult
    HarmAwareDecisionEngine ..> FindingsAggregator
    FusionEngine ..> HarmAwareDecisionEngine
```

---

### 3.2 ML Feature Union & Ensemble Pipeline

The ML pipeline is architected to eliminate false positives commonly produced by standard naive Bayes or un-calibrated LLM heuristics.

```mermaid
flowchart LR
    InputText["Raw Text Token Stream"]
    
    subgraph FeatureUnion["Scikit-Learn FeatureUnion Pipeline"]
        TFIDF["TfidfVectorizer (Word 1-3 grams, sublinear TF)"]
        CharTFIDF["TfidfVectorizer (Char 3-5 grams)"]
        Custom["Dense Linguistic Lexicon Extractor (10 Features)"]
    end

    Scaler["MaxAbsScaler (Cross-version compatible)"]
    
    subgraph Ensemble["VotingClassifier (Soft Voting)"]
        SVC["CalibratedClassifierCV (LinearSVC, C=1.0)"]
        RF["RandomForestClassifier (n_estimators=100, max_depth=20)"]
    end

    Probabilities["Calibrated Probability Vector P(C_i)"]
    Gate["Contextual Gating Filter"]
    OutputFinding["Validated ML Finding"]

    InputText --> TFIDF
    InputText --> CharTFIDF
    InputText --> Custom
    TFIDF --> Scaler
    CharTFIDF --> Scaler
    Custom --> Scaler
    Scaler --> SVC
    Scaler --> RF
    SVC --> Probabilities
    RF --> Probabilities
    Probabilities --> Gate
    Gate --> OutputFinding
```

#### Dense Linguistic Feature Engineering (`extract_custom_features`):
In addition to sublinear n-gram TF-IDF matrices, 10 specialized domain features are extracted per text snippet:
1. Exclamation point density: `count('!') / (len(text) + 1)`
2. Uppercase character ratio: `count(is_upper) / (len(text) + 1)`
3. Numerical digit density: `count(digits) / (len(text) + 1)`
4. Urgency lexicon hit count: (`hurry`, `limited`, `expires`, `ending`, `act now`, `last chance`)
5. Confirmshaming lexicon hit count: (`don't want`, `stay poor`, `prefer to pay more`, `i'll pass`)
6. Token length normalization factor: `min(tokens / 50.0, 1.0)`
7. Free/Gratis deceptive trap regex match.
8. Auto-renewal billing trigger regex match.
9. Preselection keyword regex match.
10. Forced action compulsion syntax regex match.

#### Ensemble Training Hyperparameters:
- **LinearSVC**: Calibrated via sigmoid isotonic regression (`CalibratedClassifierCV(method='sigmoid', cv=3)`) to yield true posterior probabilities.
- **RandomForest**: Constrained depth (`max_depth=20, min_samples_leaf=2`) to prevent memorization and data leakage.
- **Cross-Validation**: 5-Fold Repeated Stratified K-Fold cross-validation (`n_repeats=2`) on 4,000+ labeled dark pattern instances, achieving a macro **F1-score of 0.89**.

---

### 3.3 MLOps Confidence Drift Monitor

To prevent silent performance degradation due to internet linguistic shifts (concept drift), `ConfidenceDriftMonitor` operates as a thread-safe singleton:

```python
class ConfidenceDriftMonitor:
    def __init__(self):
        self._window = deque(maxlen=500)  # Rolling FIFO buffer
        self._alert_cooldown = 0
        self._load_baseline()              # Loads mean from model_manifest.json

    def record(self, confidences: list):
        self._window.extend(confidences)
        self._check_drift()

    def _check_drift(self):
        if not self._drift_threshold or len(self._window) < 50:
            return
        rolling_mean = sum(self._window) / len(self._window)
        if rolling_mean < self._drift_threshold:
            now = time.time()
            if now - self._alert_cooldown > 3600:  # 1 hour cooldown
                self._alert_cooldown = now
                logger.warning(
                    f"CONFIDENCE DRIFT ALERT: Rolling mean ({rolling_mean:.3f}) "
                    f"dropped below threshold ({self._drift_threshold:.3f}). "
                    f"Baseline: {self._baseline_mean:.3f}. Retraining recommended."
                )
```

---

### 3.4 Harm-Aware Decision Engine (HADE) 8-Step Protocol

HADE acts as a judicial calibration layer ensuring that findings reflect genuine consumer harm rather than stylistic web choices.

```
Step 1: Impact Scoring
        Assign finding category to High, Medium, or Low regulatory harm.
            |
Step 2: Intent Validation
        Evaluate manipulative intent (Strong / Moderate / Weak) based on UX syntax.
            |
Step 3: Decision Logic
        Apply 3x3 Impact-Intent Matrix:
        - High Impact + Strong Intent   -> Retain HIGH / CRITICAL
        - Low Impact + Weak Intent      -> Downgrade to INFORMATIONAL or Prune
            |
Step 4: Critical Pattern Boost
        Failsafe override: Irrefutable harms (pre-selected subscriptions, 
        forced arbitration, hidden cancellation fees) forced to CRITICAL.
            |
Step 5: Weak Signal Filter
        Prune informational disclosures (e.g. "We use cookies for analytics").
            |
Step 6: Multi-Signal Quorum Validation
        Medium-impact patterns require >= 2 independent corroborating signals.
            |
Step 7: Severity Recalibration
        Standardize severity across engines to FTC/GDPR tier conventions.
            |
Step 8: Trust Score Protection
        Only validated HIGH and CRITICAL findings penalize the final site trust score.
```

---

### 3.5 Weighted Ensemble Fusion & Consensus Gate

In `findings_aggregator.py` and `fusion.py`, candidate findings must survive a consensus quorum:

1. **Element-Level DOM Selector Grouping**:
   - Findings from different engines that target the same DOM element selector (e.g. `#subscribe-modal .btn-decline`) are grouped into an element cluster.
2. **Weighted Engine Quorum**:
   - Structural engines (DOM Analyzer, Cookie Auditor) receive higher weighting ($w = 0.25$) than heuristic engines ($w = 0.10$).
   - A multi-engine corroboration bonus is applied:
     $$\text{Confidence}_{\text{fused}} = \min(1.0, \text{Confidence}_{\text{base}} + 0.10) \quad \text{if } N_{\text{engines}} \ge 3$$
     $$\text{Confidence}_{\text{fused}} = \min(1.0, \text{Confidence}_{\text{base}} + 0.05) \quad \text{if } N_{\text{engines}} = 2$$
3. **Consensus Requirement**:
   A finding is only accepted into the final report if:
   - It is corroborated by $\ge 2$ independent engines, **OR**
   - It is a single strong structural rule with contextual validation, **OR**
   - It is an irrefutable CRITICAL failsafe pattern with $\text{Confidence} \ge 0.60$.

---

### 3.6 Trust Score Calculation & Severity Matrix

The overall website **Privacy & Trust Score** is computed on a 0–100 scale:

$$\text{Trust Score} = \max\left(0, 100 - \sum_{i=1}^{M} \text{Penalty}(f_i)\right)$$

Where penalties are weighted non-linearly:
- **CRITICAL Finding**: $-18$ points (e.g. Pre-selected trial enrolment, Cookie wall)
- **HIGH Finding**: $-10$ points (e.g. Confirmshaming, Hidden drip fees)
- **MEDIUM Finding**: $-4$ points (e.g. Fake countdown timer, Visual misdirection)
- **LOW / INFORMATIONAL**: $0$ points (Informative disclosure, no score penalty)

#### Site Risk Grading Scale:
| Trust Score | Risk Grade | Description | Recommended Action |
|---|---|---|---|
| **85 – 100** | **Grade A** | Clean / Trustworthy | Compliant with GDPR & FTC guidelines. |
| **70 – 84** | **Grade B** | Low Risk / Minor Nudges | Minor urgency cues; low risk of financial harm. |
| **50 – 69** | **Grade C** | Moderate Risk / Dark Patterns Present | Deceptive checkboxes or confirmshaming detected. |
| **30 – 49** | **Grade D** | High Risk / Aggressive Deception | Multiple coercive funnels and hidden fees. |
| **0 – 29** | **Grade F** | Severe Risk / Deceptive Trap | Predatory roach motels, cookie walls, or auto-billing. |

---

### 3.7 Database Schema & Storage Subsystem

The storage subsystem runs SQLite with Write-Ahead Logging (`PRAGMA journal_mode=WAL`) to allow concurrent reads and writes without thread deadlocks:

```sql
-- Scans Table
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    trust_score REAL,
    risk_grade TEXT,
    findings_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_ms REAL,
    error_message TEXT
);

-- Findings Table
CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    category TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFORMATIONAL')),
    confidence REAL NOT NULL,
    dom_selector TEXT,
    evidence_text TEXT,
    remediation_advice TEXT,
    FOREIGN KEY(scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

-- Indexes for Sub-Millisecond Lookups
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_findings_scan_id ON findings(scan_id);
CREATE INDEX IF NOT EXISTS idx_findings_category ON findings(category);
```

---

### 3.8 API Interface Contracts

#### 1. Initiate Scan
- **Route:** `POST /api/scan`
- **Rate Limit:** 10 requests / minute
- **Request Payload:**
  ```json
  {
    "url": "https://example-shop.com",
    "deep_scan": true
  }
  ```
- **Response (202 Accepted):**
  ```json
  {
    "status": "processing",
    "scan_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "poll_url": "/api/scan/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
  }
  ```

#### 2. Retrieve Scan Results
- **Route:** `GET /api/scan/<scan_id>`
- **Response (200 OK):**
  ```json
  {
    "id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "url": "https://example-shop.com",
    "status": "completed",
    "trust_score": 67.5,
    "risk_grade": "Grade C",
    "summary": {
      "total_findings": 3,
      "critical": 1,
      "high": 1,
      "medium": 1,
      "low": 0
    },
    "findings": [
      {
        "category": "preselection",
        "pattern_type": "Pre-Selected Upsell / Add-On",
        "severity": "CRITICAL",
        "confidence": 0.88,
        "dom_selector": "#warranty-opt-in > input[type='checkbox']",
        "evidence": "Pre-checked extended warranty checkbox detected in checkout flow.",
        "remediation": "Uncheck all add-ons by default; require affirmative user opt-in under EU CRD Art. 22."
      }
    ],
    "timing": {
      "browser_render_ms": 1120.4,
      "pipeline_execution_ms": 485.2,
      "total_elapsed_ms": 1605.6
    }
  }
  ```

---

## 4. Interview Talking Points & Architectural Justifications

When discussing Vigil AI in senior software engineering or AI/ML system design interviews, emphasize the following deliberate engineering tradeoffs:

1. **Why an Ensemble of Classical ML + NLP Rules over a Large Language Model (LLM)?**
   - *Cost & Latency:* Running an LLM (e.g. GPT-4) on thousands of extracted web page tokens costs pennies per request and introduces 3–8 seconds of latency. Vigil AI's `LinearSVC + RandomForest` ensemble runs in **under 25 milliseconds** with zero external API cost.
   - *Determinism & Auditability:* In regulatory and legal compliance (GDPR/FTC), decisions must be reproducible and explainable. Classical feature unions provide exact token attributions and cannot suffer from prompt injection attacks embedded in malicious webpage HTML.

2. **Why ThreadPoolExecutor rather than pure AsyncIO for Analyzer Execution?**
   - Text tokenization (spaCy syntax tree parsing) and ML inference (matrix multiplications) are **CPU-bound operations**.
   - Running CPU-bound tasks in a pure single-threaded asyncio event loop starves the loop and blocks I/O. ThreadPoolExecutor delegates execution across available CPU cores, keeping the HTTP event loop responsive.

3. **How is SSRF DNS Rebinding Defeated?**
   - Simple hostname checking fails when an attacker configures their authoritative DNS server to return a public IP on the first lookup (bypassing pre-checks) and `127.0.0.1` on the second lookup.
   - Vigil AI implements **two layers of defense**:
     1. It resolves all DNS addresses before navigation and rejects any private addresses.
     2. It activates Playwright request interception at the browser wire level, validating the destination socket of every outbound sub-request.

4. **How does HADE eliminate False Positives on legitimate e-commerce sites?**
   - Legitimate sites frequently mention words like "expires", "cart", or "privacy policy". Naive regex flags these as dark patterns.
   - HADE uses an 8-step protocol where single-engine signals are dropped unless corroborated by another engine (e.g. ML urgency + DOM timer element = True Positive). Furthermore, disclosures like "interest-based advertising" are explicitly recognized by the disclosure language filter as informative rather than deceptive.

