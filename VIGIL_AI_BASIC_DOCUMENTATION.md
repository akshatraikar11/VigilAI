# Vigil AI - Basic Implementation Guide

## Overview
Vigil AI is an enterprise-grade dark pattern detection system that analyzes websites for manipulative design techniques and legal compliance violations.

## Backend Architecture

### Project Structure
```
backend/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── config.py        # Configuration settings
│   │   └── logger.py        # Logging configuration
│   ├── routes/
│   │   └── scan.py          # API endpoints
│   ├── services/
│   │   ├── scanner.py       # Main scanning engine
│   │   ├── analyzer.py      # Pattern analysis engine
│   │   └── reporter.py      # Report generation
│   └── models/
│       └── database.py      # Database models
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

class Config:
    VERSION = "3.0.0"
    DEBUG = os.getenv("VIGIL_DEBUG", "false").lower() == "true"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///vigil_scans.db")
    
    # Scanning settings
    SCAN_TIMEOUT = int(os.getenv("VIGIL_SCAN_TIMEOUT", "90"))
    MAX_CONCURRENT_SCANS = int(os.getenv("VIGIL_MAX_SCANS", "10"))
    
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

        # Security validation
        if not self._is_safe_url(url):
            return self._error_result(url, 'URL blocked for security reasons.')

        scan_id = str(uuid.uuid4())[:8]
        result = self._init_result(scan_id, url, session_cookies)
        
        try:
            self._scan_with_playwright(url, result, session_cookies, local_storage)
            self._process_content(url, result)
        except Exception as e:
            logger.warning(f"Primary scan failed for {url}. Using fallback method...")
            self._scan_fallback(url, result)

        return result

    def _scan_with_playwright(self, url, result, session_cookies, local_storage):
        """Advanced scanning with browser automation"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1280, 'height': 800}
            )
            
            page = context.new_page()
            
            # Session injection if provided
            if session_cookies:
                context.add_cookies(session_cookies)
            
            # Navigate and extract content
            page.goto(url, wait_until='networkidle')
            result['html_content'] = page.content()
            result['text_content'] = page.evaluate('() => document.body.innerText')
            
            # Screenshot capture
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
            'text_content': '',
            'screenshot_path': None,
            'scan_state': 'authenticated' if session_cookies else 'unknown'
        }

    def _is_safe_url(self, url):
        """Security validation for URL scanning"""
        # Implementation includes IP validation, private network checks, etc.
        # Core security logic protected
        return True  # Simplified for documentation
```

#### 4. Analysis Engine (app/services/analyzer.py)
```python
from app.core.logger import get_logger

logger = get_logger('vigil.analyzer')

class PatternAnalyzer:
    """Core pattern detection engine"""
    
    def __init__(self):
        self.model = self._load_model()
        self.patterns = self._load_patterns()
    
    def analyze(self, html_content, text_content, screenshot_path):
        """Analyze content for dark patterns"""
        findings = []
        
        # Multiple analysis engines
        dom_findings = self._analyze_dom(html_content)
        text_findings = self._analyze_text(text_content)
        visual_findings = self._analyze_visual(screenshot_path)
        
        # Combine and validate findings
        all_findings = dom_findings + text_findings + visual_findings
        validated_findings = self._validate_findings(all_findings)
        
        return validated_findings
    
    def _analyze_dom(self, html_content):
        """DOM structure analysis"""
        # Core DOM analysis logic protected
        return []
    
    def _analyze_text(self, text_content):
        """Text content analysis"""
        # Core text analysis logic protected
        return []
    
    def _analyze_visual(self, screenshot_path):
        """Visual pattern analysis"""
        # Core visual analysis logic protected
        return []
    
    def _validate_findings(self, findings):
        """Validate and filter findings"""
        # Core validation logic protected
        return findings
```

#### 5. Report Generator (app/services/reporter.py)
```python
from collections import Counter
from app.core.config import Config

class ReportGenerator:
    """Generates comprehensive dark pattern reports"""
    
    def __init__(self):
        self.severity_weights = {
            'CRITICAL': 12.0,
            'HIGH': 7.0,
            'MEDIUM': 4.0,
            'LOW': 0.0,
            'INFORMATIONAL': 0.0
        }
        
        self.category_multipliers = {
            'privacy': 1.50,
            'forced_continuity': 1.40,
            'obstruction': 1.30,
            'confirmshaming': 1.20,
            'hidden_costs': 1.20,
            'urgency': 1.00,
            'social_proof': 0.90
        }

    def generate_report(self, scan_data, findings):
        """Generate comprehensive report"""
        # Calculate trust score
        trust_score = self._calculate_trust_score(findings)
        
        # Risk level assessment
        risk_level = self._get_risk_level(trust_score)
        
        # Severity breakdown
        severity_counts = Counter(f.get('severity', 'LOW') for f in findings)
        
        return {
            'trust_score': trust_score,
            'risk_level': risk_level,
            'total_patterns': len(findings),
            'severity_breakdown': dict(severity_counts),
            'findings': findings,
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
            penalty = self.severity_weights.get(severity, 0)
            
            # Category multiplier
            multiplier = self.category_multipliers.get(category, 1.0)
            
            total_penalty += penalty * multiplier
        
        return max(0, base_score - total_penalty)
```

#### 6. API Routes (app/routes/scan.py)
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.scanner import WebsiteScanner
from app.services.analyzer import PatternAnalyzer
from app.services.reporter import ReportGenerator

router = APIRouter()

class ScanRequest(BaseModel):
    url: str
    session_cookies: dict = None
    local_storage: dict = None

@router.post("/scan")
async def scan_website(request: ScanRequest):
    """Synchronous website scanning endpoint"""
    try:
        # Initialize components
        scanner = WebsiteScanner()
        analyzer = PatternAnalyzer()
        reporter = ReportGenerator()
        
        # Scan website
        scan_result = scanner.scan(request.url, request.session_cookies, request.local_storage)
        
        # Analyze findings
        findings = analyzer.analyze(
            scan_result['html_content'],
            scan_result['text_content'],
            scan_result['screenshot_path']
        )
        
        # Generate report
        report = reporter.generate_report(scan_result, findings)
        
        return {
            "status": "success",
            "scan_id": scan_result['scan_id'],
            **report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scan/extension")
async def scan_extension(request: ScanRequest):
    """Chrome extension scanning endpoint"""
    return await scan_website(request)

@router.get("/scan/status/{task_id}")
async def get_scan_status(task_id: str):
    """Get scan status for async operations"""
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
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
      width: 380px;
      margin: 0;
      padding: 24px;
      background: #0a0a0b;
      color: #f8f8f9;
    }

    .container {
      padding: 24px;
      background: #0a0a0b;
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
      background: #1a1a1c;
      border: 1px solid #2a2a2d;
      border-radius: 12px;
      padding: 10px 14px;
      font-size: 13px;
      color: #9ca3af;
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
      background: #1a1a1c;
      border: 1px solid #2a2a2d;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 24px;
    }

    .stat-card {
      background: #1a1a1c;
      border: 1px solid #2a2a2d;
      border-radius: 12px;
      padding: 16px;
      text-align: center;
    }

    .loading-view, .results-view {
      display: none;
    }

    .error-msg {
      color: #ef4444;
      font-size: 13px;
      padding: 12px;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid #ef4444;
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
        Scan Current Page
      </button>
      <div id="error-box" class="error-msg"></div>
    </div>

    <!-- LOADING VIEW -->
    <div id="loading-view">
      <div style="text-align: center; padding: 40px 0;">
        <div style="font-size: 14px; color: #9ca3af;">Running analysis...</div>
      </div>
    </div>

    <!-- RESULTS VIEW -->
    <div id="results-view">
      <div class="score-hero">
        <div style="margin-left: 20px;">
          <div id="risk-badge">Unscanned</div>
          <div style="font-size: 14px; color: #9ca3af;">Grade <strong id="grade-text">--</strong></div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-val" id="total-patterns">0</div>
          <div class="stat-label">Patterns Found</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="high-risk-patterns">0</div>
          <div class="stat-label">High/Critical Risk</div>
        </div>
      </div>

      <div style="margin-bottom: 24px;">
        <h3 style="font-size: 14px; color: #9ca3af; margin: 0 0 12px 0;">Top Findings</h3>
        <div id="findings-list"></div>
      </div>

      <a href="http://localhost:5000" target="_blank" style="display: block; text-align: center; color: #9ca3af; font-size: 13px; text-decoration: none; padding: 12px; border: 1px solid #2a2a2d; border-radius: 12px;">
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

    document.getElementById('grade-text').textContent = data.grade?.letter || '--';
    document.getElementById('total-patterns').textContent = data.total_patterns || 0;
    
    // High/Critical patterns
    const sevBreakdown = data.severity_breakdown || {};
    const highCriticalCount = (sevBreakdown.critical || 0) + (sevBreakdown.high || 0);
    document.getElementById('high-risk-patterns').textContent = highCriticalCount;

    // Top findings
    const findingsList = document.getElementById('findings-list');
    const findings = data.findings || [];
    if (findings.length > 0) {
      const addedCats = new Set();
      let html = '';
      for (const f of findings) {
        if (!addedCats.has(f.category) && addedCats.size < 3) {
          addedCats.add(f.category);
          let catName = f.category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          html += `<span style="display: inline-block; background: #1a1a1c; border: 1px solid #2a2a2d; padding: 8px 12px; border-radius: 8px; font-size: 12px; margin: 0 8px 8px 0;">${catName}</span>`;
        }
      }
      findingsList.innerHTML = html;
    } else {
      findingsList.innerHTML = `<span style="color: #10b981;">No Dark Patterns Detected</span>`;
    }
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
  
  const displayUrl = new URL(currentUrl);
  urlEl.textContent = displayUrl.hostname + (displayUrl.pathname.length > 20 ? '...' : displayUrl.pathname === '/' ? '' : displayUrl.pathname);

  // Scan button handler
  btn.addEventListener('click', async () => {
    showView('loadingView');
    errorBox.style.display = 'none';
    
    try {
      const backendUrl = 'http://localhost:5000';

      // Submit scan request
      const response = await fetch(`${backendUrl}/api/scan/extension`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: currentUrl })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'error' || data.error) {
        throw new Error(data.error || "Scan failed");
      }
      
      showView('resultsView');
      processScanResults(data);
      
    } catch (err) {
      showView('mainView');
      
      let errorMsg = err.message;
      if (err.message.includes('Failed to fetch')) {
        errorMsg = `Cannot connect to backend. Is Vigil AI running?`;
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
```

---

## API Endpoints

### Main Endpoints
- `POST /api/scan` - Synchronous website scanning
- `POST /api/scan/extension` - Chrome extension scanning
- `GET /api/scan/status/{task_id}` - Async scan status

### Request Format
```json
{
  "url": "https://example.com",
  "session_cookies": {},
  "local_storage": {}
}
```

### Response Format
```json
{
  "status": "success",
  "scan_id": "abc123",
  "trust_score": 85,
  "risk_level": "LOW",
  "total_patterns": 2,
  "findings": [...],
  "grade": {"letter": "A"}
}
```

---

## Dark Pattern Detection Categories

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

### Severity Levels
- **CRITICAL** - Direct user harm, forced action
- **HIGH** - Strong manipulation, reversible
- **MEDIUM** - Behavioral nudging
- **LOW** - Informational, minimal impact

### Trust Score Calculation
```
Base Score: 95
- (Critical × 12.0 × Category Multiplier)
- (High × 7.0 × Category Multiplier)  
- (Medium × 4.0 × Category Multiplier)
= Final Trust Score (0-100)
```

---

## Security Features

### SSRF Protection
- URL validation and sanitization
- Private IP range blocking
- DNS rebinding protection

### Authentication
- API key-based authentication
- Configurable auth modes
- Rate limiting support

### Data Privacy
- No personal data storage
- Session-based scanning only
- GDPR-compliant design

---

This basic implementation guide provides the structure and interface without exposing proprietary algorithms or core detection logic.
