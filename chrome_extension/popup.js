document.addEventListener('DOMContentLoaded', async () => {
  const urlEl = document.getElementById('url-text');
  const btn = document.getElementById('scan-btn');
  const mainView = document.getElementById('main-view');
  const loadingView = document.getElementById('loading-view');
  const resultsView = document.getElementById('results-view');
  const errorBox = document.getElementById('error-box');
  
  // Progress Ring logic (updated for new size)
  const circle = document.getElementById('score-ring');
  const radius = circle.r.baseVal.value;
  const circumference = radius * 2 * Math.PI;
  circle.style.strokeDasharray = `${circumference} ${circumference}`;
  circle.style.strokeDashoffset = circumference;

  function setProgress(percent, color) {
    const offset = circumference - (percent / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = color;
  }
  
  // Loading step management
  function showLoadingStep(stepNumber) {
    const steps = document.querySelectorAll('.loading-step');
    steps.forEach((step, index) => {
      setTimeout(() => {
        if (index < stepNumber) {
          step.classList.add('active');
        } else {
          step.classList.remove('active');
        }
      }, index * 200);
    });
  }
  
  // Enhanced view transitions
  function showView(viewName) {
    const views = { mainView, loadingView, resultsView };
    Object.values(views).forEach(view => {
      view.style.display = 'none';
      view.classList.remove('active');
    });
    
    setTimeout(() => {
      views[viewName].style.display = 'block';
      setTimeout(() => {
        views[viewName].classList.add('active');
      }, 50);
    }, 300);
  }
  
  // Add fade-in animations to results
  function animateResults() {
    const fadeElements = document.querySelectorAll('.fade-in-up');
    fadeElements.forEach((el, index) => {
      setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        setTimeout(() => {
          el.style.transition = 'all 0.5s ease-out';
          el.style.opacity = '1';
          el.style.transform = 'translateY(0)';
        }, 100);
      }, index * 150);
    });
  }

  let currentUrl = '';

  // Process scan results (common function for both sync and async)
  function processScanResults(data) {
    const score = data.trust_score || 0;
    let color = '#10b981'; // safe
    if (score < 80) color = '#eab308'; // medium
    if (score < 60) color = '#f97316'; // high
    if (score < 40) color = '#ef4444'; // critical

    // Enhanced animations & Data
    setTimeout(() => setProgress(score, color), 300);
    document.getElementById('score-text').textContent = score;
    document.getElementById('score-text').style.color = color;
    
    // Enhanced risk badge with status indicator
    const riskBadge = document.getElementById('risk-badge');
    let riskLevelClass = 'safe';
    if (score < 80) riskLevelClass = 'medium';
    if (score < 60) riskLevelClass = 'high';
    if (score < 40) riskLevelClass = 'critical';
    riskBadge.innerHTML = `<span class="status-indicator ${riskLevelClass}"></span>${data.risk_level?.label || 'UNKNOWN'}`;
    riskBadge.style.color = color;
    riskBadge.style.background = color + '22'; // 22 hex for transparency
    riskBadge.style.borderColor = color + '44';

    document.getElementById('grade-text').textContent = data.grade?.letter || '--';
    document.getElementById('total-patterns').textContent = data.total_patterns || 0;
    
    // Calculate High/Critical patterns
    const sevBreakdown = data.severity_breakdown || {};
    const highCriticalCount = (sevBreakdown.critical || 0) + (sevBreakdown.high || 0);
    document.getElementById('high-risk-patterns').textContent = highCriticalCount;
    document.getElementById('high-risk-patterns').style.color = highCriticalCount > 0 ? '#ef4444' : '#a1a1aa';

    // Top Findings list
    const findingsList = document.getElementById('findings-list');
    const findings = data.findings || [];
    if (findings.length > 0) {
      // Sort by severity 
      const sevOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'INFORMATIONAL': 0 };
      findings.sort((a, b) => sevOrder[b.severity || 'LOW'] - sevOrder[a.severity || 'LOW']);
      
      // Take top 3 unique categories
      const addedCats = new Set();
      let html = '';
      for (const f of findings) {
        if (!addedCats.has(f.category) && addedCats.size < 3) {
          addedCats.add(f.category);
          let tagColor = f.severity === 'CRITICAL' || f.severity === 'HIGH' ? '#ff8787' : '#a1a1aa';
          let catName = f.category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          html += `<span class="finding-tag" style="color: ${tagColor}">${catName}</span>`;
        }
      }
      findingsList.innerHTML = html;
    } else {
      findingsList.innerHTML = `<span class="finding-tag" style="color: #10b981;">No Dark Patterns Detected</span>`;
    }

    // UX-3 FIX: Remove existing rescan button before adding new one
    const existingRescan = resultsView.querySelector('.rescan-btn');
    if (existingRescan) existingRescan.remove();

    const rescanBtn = document.createElement('button');
    rescanBtn.innerHTML = '↻ Scan Again';
    rescanBtn.className = 'rescan-btn';
    rescanBtn.addEventListener('click', () => {
      showView('mainView');
      // Reset loading steps
      document.querySelectorAll('.loading-step').forEach(step => {
        step.classList.remove('active');
      });
    });
    resultsView.appendChild(rescanBtn);
  }

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

  btn.addEventListener('click', async () => {
    showView('loadingView');
    showLoadingStep(1);
    errorBox.style.display = 'none';
    
    try {
      // P0 FIX: Configurable backend URL (reads from chrome.storage, falls back to localhost)
      const DEFAULT_BACKEND = 'http://localhost:5000';
      let backendUrl = DEFAULT_BACKEND;
      try {
        const stored = await chrome.storage.sync.get('vigilBackendUrl');
        if (stored.vigilBackendUrl) backendUrl = stored.vigilBackendUrl.replace(/\/+$/, '');
      } catch (_) { /* storage unavailable — use default */ }

      // Submit to async scan queue
      showLoadingStep(2);
      const submitRes = await fetch(`${backendUrl}/api/scan/extension`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: currentUrl })
      });
      
      if (!submitRes.ok) {
        // Fallback to sync if async unavailable
        showLoadingStep(3);
        const syncRes = await fetch(`${backendUrl}/api/scan`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: currentUrl })
        });
        
        const syncData = await syncRes.json();
        if (syncData.status === 'error' || syncData.error) {
          throw new Error(syncData.error || "Scan failed");
        }
        
        showLoadingStep(4);
        await new Promise(resolve => setTimeout(resolve, 500));
        showView('resultsView');
        animateResults();
        
        // Process sync results
        processScanResults(syncData);
        return;
      }
      
      // Check if submission response is JSON
      const submitContentType = submitRes.headers.get('content-type');
      if (!submitContentType || !submitContentType.includes('application/json')) {
        throw new Error('Server returned invalid response format during submission');
      }
      
      const submitData = await submitRes.json();
      const task_id = submitData.task_id;
      if (!task_id) throw new Error('No task ID returned from server.');
      
      showLoadingStep(3);
      
      // Poll for results
      let attempts = 0;
      const maxAttempts = 60; // 2 minutes max
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Poll every 2s
        
        const pollRes = await fetch(`${backendUrl}/api/scan/status/${task_id}`);
        
        // Check if response is JSON
        const contentType = pollRes.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          throw new Error('Server returned invalid response format');
        }
        
        const poll = await pollRes.json();
        
        const state = poll.state || poll.status || 'PENDING';
        
        if (state === 'SUCCESS' || state === 'FAILURE') {
          showLoadingStep(4);
          await new Promise(resolve => setTimeout(resolve, 500));
          showView('resultsView');
          animateResults();
          
          if (state === 'FAILURE' || poll.status === 'error') {
            throw new Error(poll.error || 'Scan failed on worker.');
          }
          
          const data = poll.result || poll;
          processScanResults(data);
          return;
        }
        
        attempts++;
        
        // Update loading message based on stage
        const loaderText = document.querySelector('.loader-text');
        if (loaderText) {
          const stage = poll.stage || state;
          const messages = {
            'PENDING': 'Initializing scan...',
            'SCANNING': 'Extracting page content...',
            'ANALYZING': 'Running AI analysis...',
            'CALIBRATING': 'Generating report...'
          };
          loaderText.textContent = messages[stage] || 'Processing...';
        }
      }
      
      throw new Error('Scan timed out after 2 minutes.');
    } catch (err) {
      showView('mainView');
      
      let errorMsg = err.message;
      if (err.message.includes('Failed to fetch')) {
        errorMsg = `Cannot connect to backend at ${backendUrl || 'http://localhost:5000'}. Is Vigil AI running?`;
      }
      
      errorBox.textContent = errorMsg;
      errorBox.style.display = 'block';
      
      // Shake animation for error
      errorBox.style.animation = 'shake 0.5s';
      setTimeout(() => {
        errorBox.style.animation = '';
      }, 500);
    }
  });
});

// Add shake animation for errors
const style = document.createElement('style');
style.textContent = `
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
  }
`;
document.head.appendChild(style);
