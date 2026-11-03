# Vigil AI - Complete Code Documentation

## Table of Contents
1. [Backend Architecture](#backend-architecture)
2. [Frontend Chrome Extension](#frontend-chrome-extension)
3. [Installation & Setup](#installation--setup)
4. [API Endpoints](#api-endpoints)
5. [Dark Pattern Detection](#dark-pattern-detection)

---

## Backend Architecture

### Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── config.py        # Configuration settings
│   │   ├── logger.py        # Logging configuration
│   │   └── auth.py          # Authentication
│   ├── routes/
│   │   └── scan.py          # API endpoints
│   ├── services/
│   │   ├── scanner.py       # Main scanning engine
│   │   ├── decision_engine.py # HADE decision system
│   │   ├── report_generator.py # Report generation
│   │   ├── dom_analyzer.py  # DOM analysis
│   │   ├── text_analyzer.py # Text analysis
│   │   ├── visual_analyzer.py # Visual analysis
│   │   ├── cookie_analyzer.py # Cookie analysis
│   │   ├── ml_analyzer.py   # Machine learning analysis
│   │   └── advanced_analyzer.py # Advanced patterns
│   └── analyzers/
│       └── adapters.py      # Analyzer adapters
├── run.py                   # Production entry point
├── requirements.txt         # Dependencies
└── .env.example            # Environment variables
```

### Core Backend Files

#### 1. Main Application (app/main.py)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import scan
from app.core.config import Config

def create_app():
    app = FastAPI(
        title="Vigil AI",
        description="Enterprise-grade dark pattern detection API",
        version=Config.VERSION
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(scan.router, prefix="/api")
    
    return app

app = create_app()
```

#### 2. Configuration (app/core/config.py)
```python
import os
from typing import Optional

class Config:
    VERSION = "3.0.0"
    DEBUG = os.getenv("VIGIL_DEBUG", "false").lower() == "true"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///vigil_scans.db")
    
    # Scanning settings
    SCAN_TIMEOUT = int(os.getenv("VIGIL_SCAN_TIMEOUT", "90"))
    MAX_CONCURRENT_SCANS = int(os.getenv("VIGIL_MAX_SCANS", "10"))
    
    # ML Model settings
    MODEL_PATH = os.getenv("VIGIL_MODEL_PATH", "models/vigil_model.pkl")
    
    # API Keys
    API_KEY = os.getenv("VIGIL_API_KEY")
    AUTH_ENABLED = bool(API_KEY)
```

#### 3. Scanner Service (app/services/scanner.py)
```python
import uuid
import time
import requests
from playwright.sync_api import sync_playwright
from app.services.dom_extractor import extract_dom_data
from app.core.logger import get_logger

logger = get_logger('vigil.scanner')

class WebsiteScanner:
    def __init__(self, screenshot_dir='static/screenshots'):
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def scan(self, url, session_cookies=None, local_storage=None):
        """Main scanning method with SSRF protection"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # SSRF protection
        if not self._is_safe_url(url):
            return self._error_result(url, 'URL blocked: internal/private addresses are not allowed.')

        scan_id = str(uuid.uuid4())[:8]
        result = self._init_result(scan_id, url, session_cookies)
        
        try:
            self._scan_with_playwright(url, result, session_cookies, local_storage)
            self._process_html(url, result)
        except Exception as e:
            logger.warning(f"Playwright failed for {url}. Falling back to requests...")
            self._scan_fallback(url, result)

        return result

    def _scan_with_playwright(self, url, result, session_cookies, local_storage):
        """Advanced scanning with Playwright browser automation"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1280, 'height': 800}
            )
            
            page = context.new_page()
            
            # Session injection
            if session_cookies:
                context.add_cookies(session_cookies)
            
            # Navigate and extract content
            page.goto(url, wait_until='networkidle')
            result['html_content'] = page.content()
            
            # Screenshot
            screenshot_path = os.path.join(self.screenshot_dir, f"{result['scan_id']}.png")
            page.screenshot(path=screenshot_path)
            result['screenshot_path'] = screenshot_path
            
            browser.close()

    def _init_result(self, scan_id, url, session_cookies):
        """Initialize scan result structure"""
        return {
            'scan_id': scan_id,
            'url': url,
            'domain': urlparse(url).netloc,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success',
            'html_content': '',
            'dom_data': {},
            'scan_state': 'authenticated' if session_cookies else 'unknown'
        }
```

#### 4. Decision Engine (app/services/decision_engine.py)
```python
from app.core.logger import get_logger

logger = get_logger('vigil.hade')

class HarmAwareDecisionEngine:
    """8-step decision protocol for dark pattern validation"""
    
    CONF_DEFAULT = 0.75
    CONF_HIGH_IMPACT = 0.65
    CONF_CRITICAL = 0.60

    # High impact categories
    _HIGH_IMPACT = frozenset({
        'privacy', 'cookie_wall', 'tracking', 'forced_continuity',
        'obstruction', 'hidden_costs', 'preselection', 'trick_question'
    })
    
    # Critical override triggers
    _CRITICAL_TYPE_TRIGGERS = (
        'pre-selected marketing checkbox',
        'cookie_manipulation',
        'basket sneaking',
        'forced arbitration',
        'compound dark pattern'
    )

    def evaluate(self, all_findings):
        """Complete 8-step evaluation pipeline"""
        if not all_findings:
            return []

        # Steps 1-2: Score findings
        evaluated = [self._score(f) for f in all_findings]
        
        # Steps 3-5: Apply decision logic
        accepted = [self._decide(f) for f in evaluated if f is not None]
        
        # Step 6: Multi-signal validation
        accepted = self._multi_signal_gate(accepted)
        
        # Step 7: Final severity recalibration
        accepted = [self._recalibrate(f) for f in accepted]
        
        return accepted

    def _score(self, finding):
        """Attach impact and intent strength scores"""
        f = dict(finding)
        cat = f.get('category', '')
        
        # Impact scoring
        if cat in self._HIGH_IMPACT:
            f['_impact'] = 'HIGH'
        else:
            f['_impact'] = 'MEDIUM'
        
        # Critical override
        if any(trigger in f.get('type', '').lower() for trigger in self._CRITICAL_TYPE_TRIGGERS):
            f['_impact'] = 'HIGH'
            f['_is_critical'] = True
        
        return f
```

#### 5. Report Generator (app/services/report_generator.py)
```python
from collections import Counter
from app.core.config import Config

class ReportGenerator:
    """Generates comprehensive dark pattern reports with legal compliance tags"""
    
    SEVERITY_WEIGHTS = {
        'CRITICAL': 12.0,
        'HIGH': 7.0,
        'MEDIUM': 4.0,
        'LOW': 0.0,
        'INFORMATIONAL': 0.0
    }
    
    CATEGORY_MULTIPLIERS = {
        'privacy': 1.50,
        'forced_continuity': 1.40,
        'obstruction': 1.30,
        'confirmshaming': 1.20,
        'hidden_costs': 1.20,
        'urgency': 1.00,
        'social_proof': 0.90
    }
    
    LEGAL_REFS = {
        'privacy': ['GDPR Art. 7', 'GDPR Art. 17', 'CCPA §1798.100'],
        'forced_continuity': ['FTC ROSCA Act', 'EU Consumer Rights Directive'],
        'hidden_costs': ['EU Price Indication Directive', 'FTC Act §5'],
        'preselection': ['GDPR Art. 7(4)', 'EU ePrivacy Directive Art. 5(3)'],
        'confirmshaming': ['FTC Act §5', 'EU Unfair Commercial Practices Directive'],
        'urgency': ['ASA CAP Code', 'FTC Act §5'],
        'social_proof': ['FTC Endorsement Guidelines', 'ASA CAP Code']
    }

    def generate_report(self, scan_data, dom_findings, text_findings, visual_findings):
        """Generate comprehensive dark pattern report"""
        all_findings = dom_findings + text_findings + visual_findings
        
        # Calculate trust score
        trust_score = self._calculate_trust_score(all_findings)
        
        # Risk level assessment
        risk_level = self._get_risk_level(trust_score)
        
        # Severity breakdown
        severity_counts = Counter(f.get('severity', 'LOW') for f in all_findings)
        
        return {
            'trust_score': trust_score,
            'risk_level': risk_level,
            'total_patterns': len(all_findings),
            'severity_breakdown': dict(severity_counts),
            'findings': all_findings,
            'grade': self._calculate_grade(trust_score)
        }

    def _calculate_trust_score(self, findings):
        """Calculate trust score based on findings"""
        base_score = 95
        total_penalty = 0
        
        for finding in findings:
            severity = finding.get('severity', 'LOW')
            category = finding.get('category', '')
            
            # Base penalty
            penalty = self.SEVERITY_WEIGHTS.get(severity, 0)
            
            # Category multiplier
            multiplier = self.CATEGORY_MULTIPLIERS.get(category, 1.0)
            
            total_penalty += penalty * multiplier
        
        return max(0, base_score - total_penalty)
```

#### 6. API Routes (app/routes/scan.py)
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.services.scanner import WebsiteScanner
from app.services.report_generator import ReportGenerator
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger('vigil.api')

class ScanRequest(BaseModel):
    url: str
    session_cookies: dict = None
    local_storage: dict = None

@router.post("/scan")
async def scan_website(request: ScanRequest):
    """Synchronous website scanning endpoint"""
    try:
        scanner = WebsiteScanner()
        scan_result = scanner.scan(request.url, request.session_cookies, request.local_storage)
        
        # Analyze findings
        from app.services import dom_analyzer, text_analyzer, visual_analyzer
        
        dom_findings = dom_analyzer.analyze(scan_result['html_content'])
        text_findings = text_analyzer.analyze(scan_result['text_content'])
        visual_findings = visual_analyzer.analyze(scan_result['screenshot_path'])
        
        # Generate report
        generator = ReportGenerator()
        report = generator.generate_report(scan_result, dom_findings, text_findings, visual_findings)
        
        return {
            "status": "success",
            "scan_id": scan_result['scan_id'],
            **report
        }
        
    except Exception as e:
        logger.error(f"Scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scan/extension")
async def scan_extension(request: ScanRequest):
    """Chrome extension scanning endpoint"""
    return await scan_website(request)

@router.get("/scan/status/{task_id}")
async def get_scan_status(task_id: str):
    """Get scan status for async operations"""
    # Implementation for async scan status checking
    return {"task_id": task_id, "status": "completed"}
```

---

## Frontend Chrome Extension

### Extension Structure
```
chrome_extension/
├── manifest.json          # Extension configuration
├── popup.html             # Extension popup UI
├── popup.js              # Extension logic
└── icons/                # Extension icons
```

### Core Extension Files

#### 1. Manifest (manifest.json)
```json
{
  "manifest_version": 3,
  "name": "Vigil AI Dark Pattern Radar",
  "version": "1.0",
  "description": "Enterprise-grade dark pattern detection directly in your browser.",
  "permissions": [
    "activeTab",
    "scripting"
  ],
  "host_permissions": [
    "http://localhost:5000/*"
  ],
  "action": {
    "default_popup": "popup.html",
    "default_title": "Vigil AI — Dark Pattern Radar"
  },
  "content_security_policy": {
    "extension_pages": "script-src 'self'; object-src 'self'"
  }
}
```

#### 2. Popup HTML (popup.html)
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    :root {
      --bg: #0a0a0b;
      --surface: #1a1a1c;
      --primary: #6366f1;
      --text-main: #f8f8f9;
      --text-muted: #9ca3af;
      --critical: #ef4444;
      --high: #f97316;
      --medium: #eab308;
      --safe: #10b981;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
      width: 380px;
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text-main);
    }

    .container {
      padding: 24px;
      background: var(--bg);
      min-height: 500px;
    }

    header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 24px;
    }

    h1 {
      font-size: 20px;
      font-weight: 700;
      margin: 0;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .url-pill {
      background: var(--surface);
      border: 1px solid #2a2a2d;
      border-radius: 12px;
      padding: 10px 14px;
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 24px;
    }

    button {
      width: 100%;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      border: none;
      padding: 14px 20px;
      border-radius: 12px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
    }

    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
    }

    .score-hero {
      display: flex;
      align-items: center;
      background: var(--surface);
      border: 1px solid #2a2a2d;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
    }

    .score-circle-wrapper {
      position: relative;
      width: 80px;
      height: 80px;
    }

    .score-value {
      position: absolute;
      top: 0; left: 0; width: 100%; height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      font-weight: 800;
    }

    .risk-badge {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 24px;
    }

    .stat-card {
      background: var(--surface);
      border: 1px solid #2a2a2d;
      border-radius: 12px;
      padding: 16px;
      text-align: center;
    }

    .stat-val {
      font-size: 20px;
      font-weight: 800;
      margin-bottom: 4px;
    }

    .stat-label {
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .loading-view, .results-view {
      display: none;
    }

    .loading-view.active, .results-view.active {
      display: block;
    }

    .error-msg {
      color: var(--critical);
      font-size: 13px;
      padding: 12px;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid var(--critical);
      border-radius: 12px;
      margin-top: 16px;
      display: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: flex; align-items: center; justify-content: center;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="white"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
      </div>
      <h1>Vigil AI</h1>
    </header>

    <div class="url-pill">
      <span id="url-text">Loading...</span>
    </div>

    <!-- MAIN VIEW -->
    <div id="main-view">
      <button id="scan-btn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        Scan Current Page
      </button>
      <div id="error-box" class="error-msg"></div>
    </div>

    <!-- LOADING VIEW -->
    <div id="loading-view">
      <div style="text-align: center; padding: 40px 0;">
        <div style="width: 80px; height: 80px; border-radius: 50%; border: 3px solid var(--primary); margin: 0 auto 20px; position: relative; animation: pulse 2s infinite;">
          <div style="position: absolute; top: 50%; left: 50%; width: 60%; height: 3px; background: var(--primary); transform-origin: 0 50%; animation: scan 1.5s linear infinite;"></div>
        </div>
        <div style="font-size: 14px; color: var(--text-muted);">Running comprehensive analysis...</div>
      </div>
    </div>

    <!-- RESULTS VIEW -->
    <div id="results-view">
      <div class="score-hero">
        <div class="score-circle-wrapper">
          <svg width="80" height="80" style="transform: rotate(-90deg);">
            <circle cx="40" cy="40" r="35" fill="none" stroke="#2a2a2d" stroke-width="8"></circle>
            <circle id="score-ring" cx="40" cy="40" r="35" fill="none" stroke="var(--primary)" stroke-width="8" stroke-dasharray="220" stroke-dashoffset="220"></circle>
          </svg>
          <div class="score-value" id="score-text">--</div>
        </div>
        <div style="margin-left: 20px;">
          <div class="risk-badge" id="risk-badge">Unscanned</div>
          <div style="font-size: 14px; color: var(--text-muted);">Grade <strong id="grade-text">--</strong></div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val" id="total-patterns">0</div>
          <div class="stat-label">Patterns Found</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="high-risk-patterns" style="color: var(--critical);">0</div>
          <div class="stat-label">High/Critical Risk</div>
        </div>
      </div>

      <div style="margin-bottom: 24px;">
        <h3 style="font-size: 14px; color: var(--text-muted); margin: 0 0 12px 0; text-transform: uppercase;">Top Findings</h3>
        <div id="findings-list"></div>
      </div>

      <a href="http://localhost:5000" target="_blank" style="display: block; text-align: center; color: var(--text-muted); font-size: 13px; text-decoration: none; padding: 12px; border: 1px solid #2a2a2d; border-radius: 12px;">
        Open Full Dashboard →
      </a>
    </div>
  </div>
  <script src="popup.js"></script>
</body>
</html>
```

#### 3. Extension Logic (popup.js)
```javascript
document.addEventListener('DOMContentLoaded', async () => {
  const urlEl = document.getElementById('url-text');
  const btn = document.getElementById('scan-btn');
  const mainView = document.getElementById('main-view');
  const loadingView = document.getElementById('loading-view');
  const resultsView = document.getElementById('results-view');
  const errorBox = document.getElementById('error-box');
  
  let currentUrl = '';

  // Progress Ring setup
  const circle = document.getElementById('score-ring');
  const circumference = 2 * Math.PI * 35;
  circle.style.strokeDasharray = `${circumference} ${circumference}`;
  circle.style.strokeDashoffset = circumference;

  function setProgress(percent, color) {
    const offset = circumference - (percent / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = color;
  }
  
  function showView(viewName) {
    const views = { mainView, loadingView, resultsView };
    Object.values(views).forEach(view => {
      view.style.display = 'none';
      view.classList.remove('active');
    });
    
    setTimeout(() => {
      views[viewName].style.display = 'block';
      views[viewName].classList.add('active');
    }, 300);
  }

  function processScanResults(data) {
    const score = data.trust_score || 0;
    let color = '#10b981'; // safe
    if (score < 80) color = '#eab308'; // medium
    if (score < 60) color = '#f97316'; // high
    if (score < 40) color = '#ef4444'; // critical

    setTimeout(() => setProgress(score, color), 300);
    document.getElementById('score-text').textContent = score;
    document.getElementById('score-text').style.color = color;
    
    // Risk badge
    const riskBadge = document.getElementById('risk-badge');
    let riskLevel = 'SAFE';
    if (score < 80) riskLevel = 'MEDIUM';
    if (score < 60) riskLevel = 'HIGH';
    if (score < 40) riskLevel = 'CRITICAL';
    
    riskBadge.textContent = riskLevel;
    riskBadge.style.color = color;
    riskBadge.style.background = color + '22';

    document.getElementById('grade-text').textContent = data.grade?.letter || '--';
    document.getElementById('total-patterns').textContent = data.total_patterns || 0;
    
    // High/Critical patterns
    const sevBreakdown = data.severity_breakdown || {};
    const highCriticalCount = (sevBreakdown.critical || 0) + (sevBreakdown.high || 0);
    document.getElementById('high-risk-patterns').textContent = highCriticalCount;
    document.getElementById('high-risk-patterns').style.color = highCriticalCount > 0 ? '#ef4444' : '#9ca3af';

    // Top findings
    const findingsList = document.getElementById('findings-list');
    const findings = data.findings || [];
    if (findings.length > 0) {
      const sevOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'INFORMATIONAL': 0 };
      findings.sort((a, b) => sevOrder[b.severity || 'LOW'] - sevOrder[a.severity || 'LOW']);
      
      const addedCats = new Set();
      let html = '';
      for (const f of findings) {
        if (!addedCats.has(f.category) && addedCats.size < 3) {
          addedCats.add(f.category);
          let tagColor = f.severity === 'CRITICAL' || f.severity === 'HIGH' ? '#ff8787' : '#9ca3af';
          let catName = f.category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          html += `<span style="display: inline-block; background: #1a1a1c; border: 1px solid #2a2a2d; padding: 8px 12px; border-radius: 8px; font-size: 12px; margin: 0 8px 8px 0; color: ${tagColor};">${catName}</span>`;
        }
      }
      findingsList.innerHTML = html;
    } else {
      findingsList.innerHTML = `<span style="color: #10b981;">No Dark Patterns Detected</span>`;
    }

    // Rescan button
    const existingRescan = resultsView.querySelector('.rescan-btn');
    if (existingRescan) existingRescan.remove();

    const rescanBtn = document.createElement('button');
    rescanBtn.innerHTML = '↻ Scan Again';
    rescanBtn.className = 'rescan-btn';
    rescanBtn.addEventListener('click', () => {
      showView('mainView');
    });
    resultsView.appendChild(rescanBtn);
  }

  // Get current tab URL
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    currentUrl = tabs[0].url;
  } catch (err) {
    urlEl.textContent = 'Error reading URL';
    btn.disabled = true;
    return;
  }
  
  if (!currentUrl || currentUrl.startsWith('chrome://') || currentUrl.startsWith('edge://') || currentUrl.startsWith('about:')) {
    urlEl.textContent = 'Invalid Protocol';
    btn.disabled = true;
    return;
  }
  
  const displayUrl = new URL(currentUrl);
  urlEl.textContent = displayUrl.hostname + (displayUrl.pathname.length > 20 ? displayUrl.pathname.substring(0, 20) + '...' : displayUrl.pathname === '/' ? '' : displayUrl.pathname);

  // Scan button handler
  btn.addEventListener('click', async () => {
    showView('loadingView');
    errorBox.style.display = 'none';
    
    try {
      const DEFAULT_BACKEND = 'http://localhost:5000';
      let backendUrl = DEFAULT_BACKEND;
      
      try {
        const stored = await chrome.storage.sync.get('vigilBackendUrl');
        if (stored.vigilBackendUrl) backendUrl = stored.vigilBackendUrl.replace(/\/+$/, '');
      } catch (_) { /* storage unavailable */ }

      // Submit scan request
      const submitRes = await fetch(`${backendUrl}/api/scan/extension`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: currentUrl })
      });
      
      if (!submitRes.ok) {
        throw new Error(`HTTP ${submitRes.status}: ${submitRes.statusText}`);
      }
      
      const data = await submitRes.json();
      
      if (data.status === 'error' || data.error) {
        throw new Error(data.error || "Scan failed");
      }
      
      showView('resultsView');
      processScanResults(data);
      
    } catch (err) {
      showView('mainView');
      
      let errorMsg = err.message;
      if (err.message.includes('Failed to fetch')) {
        errorMsg = `Cannot connect to backend at ${backendUrl || 'http://localhost:5000'}. Is Vigil AI running?`;
      }
      
      errorBox.textContent = errorMsg;
      errorBox.style.display = 'block';
    }
  });
});
```

---

## Installation & Setup

### Backend Setup
```bash
# Clone repository
git clone <repository-url>
cd VigilAI/backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application
python run.py
```

### Frontend Extension Setup
```bash
# Navigate to extension directory
cd chrome_extension

# Load extension in Chrome
1. Open Chrome
2. Go to chrome://extensions/
3. Enable "Developer mode"
4. Click "Load unpacked"
5. Select the chrome_extension directory
```

### Environment Variables (.env)
```bash
# Database
DATABASE_URL=sqlite:///vigil_scans.db

# API Configuration
VIGIL_API_KEY=your-secret-api-key
VIGIL_DEBUG=false
VIGIL_PORT=5000

# Scanning Settings
VIGIL_SCAN_TIMEOUT=90
VIGIL_MAX_SCANS=10

# ML Model
VIGIL_MODEL_PATH=models/vigil_model.pkl
```

---

## API Endpoints

### Scan Endpoints
- `POST /api/scan` - Synchronous website scanning
- `POST /api/scan/extension` - Chrome extension scanning
- `GET /api/scan/status/{task_id}` - Async scan status

### Request/Response Format
```json
// Request
{
  "url": "https://example.com",
  "session_cookies": {},
  "local_storage": {}
}

// Response
{
  "status": "success",
  "scan_id": "abc123",
  "trust_score": 85,
  "risk_level": "LOW",
  "total_patterns": 2,
  "severity_breakdown": {
    "critical": 0,
    "high": 1,
    "medium": 1,
    "low": 0
  },
  "findings": [...],
  "grade": {"letter": "A"}
}
```

---

## Dark Pattern Detection

### Detection Categories
1. **Privacy Violations** - Cookie walls, tracking without consent
2. **Forced Continuity** - Auto-billing, subscription traps
3. **Obstruction** - Roach motel, blocked cancellation
4. **Hidden Costs** - Drip pricing, late fee disclosure
5. **Preselection** - Pre-checked consent boxes
6. **Confirmshaming** - Guilt-tripping decline options
7. **Urgency** - Fake scarcity, countdown timers
8. **Social Proof** - Fake popularity claims
9. **Misdirection** - Visual/UI manipulation

### Legal Framework References
- **GDPR** - Articles 7, 12, 17 (consent, transparency, erasure)
- **CCPA** - California Consumer Privacy Act
- **FTC Act** - Section 5 (unfair/deceptive practices)
- **DSA** - Digital Services Act Article 25
- **ePrivacy Directive** - Cookie consent requirements

### Severity Levels
- **CRITICAL** (12.0 points) - Direct user harm, forced action
- **HIGH** (7.0 points) - Strong manipulation, reversible
- **MEDIUM** (4.0 points) - Behavioral nudging
- **LOW** (0.0 points) - Informational, no score impact

### Trust Score Calculation
```
Base Score: 95
- (Critical × 12.0 × Category Multiplier)
- (High × 7.0 × Category Multiplier)  
- (Medium × 4.0 × Category Multiplier)
= Final Trust Score (0-100)
```

---

This documentation provides a complete overview of the Vigil AI system architecture, implementation details, and usage instructions for both backend and frontend components.
