/**
 * Kiron - Selbsttest Tab
 * Pattern: Background-Thread + Status-Polling.
 * Markup identisch zu kiara/tab_testsuite.js (config-form + config-section).
 */

let _selftestPollTimer = null;
let _selftestInitSeq = 0;

function _selftestStopPolling() {
    if (_selftestPollTimer !== null) {
        clearInterval(_selftestPollTimer);
        _selftestPollTimer = null;
    }
}

async function initSelftestTab() {
    _selftestInitSeq++;
    const mySeq = _selftestInitSeq;
    _selftestStopPolling();

    const container = document.getElementById('contentContainer');
    if (!container) return;
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await fetch('/api/tests/status');
        if (mySeq !== _selftestInitSeq) return;
        if (!resp.ok) throw new Error('Laden fehlgeschlagen');
        const data = await resp.json();
        renderSelftestTab(container, data, mySeq);
    } catch (e) {
        if (mySeq !== _selftestInitSeq) return;
        container.innerHTML = '<div class="no-data">Selbsttest-Status konnte nicht geladen werden.</div>';
    }
}

function renderSelftestTab(container, data, mySeq) {
    if (mySeq !== undefined && mySeq !== _selftestInitSeq) return;
    const isRunning = data.running || false;
    const hasResults = (data.results && data.results.length > 0) || data.error;

    let html = '<div class="config-form">';
    html += '<h3>Selbsttest</h3>';

    // Start-Button
    html += '<div class="config-section">';
    html += '<div style="display:flex;align-items:center;gap:12px;">';
    html += `<button class="action-btn primary" id="btnStartTests" ${isRunning ? 'disabled' : ''}>Selbsttest starten</button>`;
    html += '<span id="testProgressInfo" style="font-size:13px;color:var(--text-secondary);">';
    if (isRunning) html += 'Tests laufen...';
    html += '</span>';
    html += '</div>';
    html += '</div>';

    // Fortschrittsbereich
    html += '<div id="testProgressSection">';
    if (isRunning) html += renderSelftestProgress(data);
    html += '</div>';

    // Ergebnis-Bereich
    html += '<div id="testResultSection">';
    if (!isRunning && hasResults) html += renderSelftestResults(data);
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;

    const btn = document.getElementById('btnStartTests');
    if (btn) btn.addEventListener('click', startSelftest);

    _bindSelftestToggles(container);

    if (isRunning) pollSelftestStatus();
}

function renderSelftestProgress(data) {
    const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
    const barColor = 'var(--accent-primary)';

    let html = '<div class="config-section" style="margin-top:8px;">';
    html += '<h4>Fortschritt</h4>';

    // Balken
    html += '<div style="background:var(--bg-tertiary, var(--bg-secondary));border-radius:6px;height:22px;overflow:hidden;position:relative;">';
    html += `<div id="testBar" style="height:100%;width:${pct}%;background:${barColor};border-radius:6px;transition:width 0.3s ease;"></div>`;
    html += `<span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:12px;font-weight:600;color:var(--text-primary);" id="testBarLabel">${data.progress}/${data.total}  (${pct}%)</span>`;
    html += '</div>';

    // Aktueller Test
    html += `<div style="margin-top:6px;font-size:12px;color:var(--text-secondary);font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" id="testCurrentName">${escapeHtml(data.current_test || '')}</div>`;

    // Live-Zaehler
    html += '<div style="margin-top:8px;display:flex;gap:16px;font-size:13px;" id="testLiveCounters">';
    html += renderSelftestCounters(data);
    html += '</div>';

    html += '</div>';
    return html;
}

function renderSelftestCounters(data) {
    let html = '';
    html += `<span style="color:var(--color-bullish);">&#10003; ${data.passed}</span>`;
    html += `<span style="color:var(--color-bearish);">&#10007; ${data.failed}</span>`;
    if (data.errors > 0) html += `<span style="color:var(--color-bearish);">&#9888; ${data.errors}</span>`;
    html += `<span style="color:var(--text-secondary);">&#8856; ${data.skipped}</span>`;
    return html;
}

function renderSelftestResults(data) {
    let html = '';

    if (data.error) {
        html += '<div class="config-section" style="margin-top:8px;">';
        html += `<div class="widget-error" style="font-size:14px;padding:12px;white-space:pre-wrap;">${escapeHtml(data.error)}</div>`;
        html += '</div>';
    }

    if (!data.results || data.results.length === 0) return html;

    // KPI-Leiste
    html += '<div class="config-section" style="margin-top:8px;">';
    html += '<div style="display:flex;gap:20px;flex-wrap:wrap;font-size:14px;padding:12px 0;">';

    html += `<span style="color:var(--color-bullish);font-weight:600;">&#10003; ${data.passed} bestanden</span>`;
    if (data.failed > 0) {
        html += `<span style="color:var(--color-bearish);font-weight:600;">&#10007; ${data.failed} fehlgeschlagen</span>`;
    }
    if (data.errors > 0) {
        html += `<span style="color:var(--color-bearish);font-weight:600;">&#9888; ${data.errors} Fehler</span>`;
    }
    if (data.skipped > 0) {
        html += `<span style="color:var(--text-secondary);">&#8856; ${data.skipped} uebersprungen</span>`;
    }
    html += `<span style="color:var(--text-secondary);">&#9201; ${data.duration}s</span>`;

    html += '</div>';
    html += '</div>';

    // Fehlgeschlagene Tests (aufgeklappt)
    const failed = data.results.filter(r => r.status === 'failed' || r.status === 'error');
    if (failed.length > 0) {
        html += '<div class="config-section" style="margin-top:8px;">';
        html += `<h4 style="cursor:pointer;user-select:none;" data-toggle-section="failedSection">`;
        html += `<span id="failedSectionArrow" style="display:inline-block;transition:transform 0.2s;">&#9660;</span> Fehlgeschlagene Tests (${failed.length})</h4>`;
        html += '<div id="failedSection">';
        failed.forEach((t, i) => {
            const statusLabel = t.status === 'error' ? 'ERROR' : 'FAILED';
            const statusColor = 'var(--color-bearish)';
            html += `<div style="margin-bottom:8px;border:1px solid var(--border-color);border-radius:6px;overflow:hidden;">`;
            html += `<div style="padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:8px;background:var(--bg-secondary);" data-toggle-detail="failDetail${i}" data-toggle-arrow="failArrow${i}">`;
            html += `<span id="failArrow${i}" style="display:inline-block;transition:transform 0.2s;font-size:10px;">&#9654;</span>`;
            html += `<span style="color:${statusColor};font-weight:600;font-size:12px;">${statusLabel}</span>`;
            html += `<span style="font-family:monospace;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(t.name)}</span>`;
            html += '</div>';
            html += `<div id="failDetail${i}" style="display:none;padding:8px 12px;background:var(--bg-primary);">`;
            if (t.details) {
                html += `<pre style="margin:0;font-size:11px;line-height:1.5;overflow-x:auto;white-space:pre-wrap;word-break:break-all;color:var(--text-primary);">${escapeHtml(t.details)}</pre>`;
            } else {
                html += '<span style="font-size:12px;color:var(--text-secondary);">Keine Details verfuegbar.</span>';
            }
            html += '</div></div>';
        });
        html += '</div></div>';
    }

    // Bestandene Tests (eingeklappt)
    const passed = data.results.filter(r => r.status === 'passed');
    if (passed.length > 0) {
        html += '<div class="config-section" style="margin-top:8px;">';
        html += `<h4 style="cursor:pointer;user-select:none;" data-toggle-section="passedSection">`;
        html += `<span id="passedSectionArrow" style="display:inline-block;transition:transform 0.2s;transform:rotate(-90deg);">&#9660;</span> Bestanden (${passed.length})</h4>`;
        html += '<div id="passedSection" style="display:none;">';
        html += '<div style="columns:2;column-gap:16px;font-size:12px;font-family:monospace;">';
        passed.forEach(t => {
            html += `<div style="padding:2px 0;break-inside:avoid;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--color-bullish);">&#10003; ${escapeHtml(t.name)}</div>`;
        });
        html += '</div></div></div>';
    }

    // Uebersprungene Tests (eingeklappt)
    const skipped = data.results.filter(r => r.status === 'skipped');
    if (skipped.length > 0) {
        html += '<div class="config-section" style="margin-top:8px;">';
        html += `<h4 style="cursor:pointer;user-select:none;" data-toggle-section="skippedSection">`;
        html += `<span id="skippedSectionArrow" style="display:inline-block;transition:transform 0.2s;transform:rotate(-90deg);">&#9660;</span> Uebersprungen (${skipped.length})</h4>`;
        html += '<div id="skippedSection" style="display:none;">';
        html += '<div style="columns:2;column-gap:16px;font-size:12px;font-family:monospace;">';
        skipped.forEach(t => {
            html += `<div style="padding:2px 0;break-inside:avoid;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-secondary);">&#8856; ${escapeHtml(t.name)}</div>`;
        });
        html += '</div></div></div>';
    }

    return html;
}

function _bindSelftestToggles(root) {
    if (!root) return;
    root.querySelectorAll('[data-toggle-section]').forEach(el => {
        el.addEventListener('click', function () { toggleSelftestSection(this.dataset.toggleSection); });
    });
    root.querySelectorAll('[data-toggle-detail]').forEach(el => {
        el.addEventListener('click', function () { toggleSelftestDetail(this.dataset.toggleDetail, this.dataset.toggleArrow); });
    });
}

function toggleSelftestSection(sectionId) {
    const section = document.getElementById(sectionId);
    const arrow = document.getElementById(sectionId + 'Arrow');
    if (!section) return;
    const isHidden = section.style.display === 'none';
    section.style.display = isHidden ? '' : 'none';
    if (arrow) arrow.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(-90deg)';
}

function toggleSelftestDetail(detailId, arrowId) {
    const detail = document.getElementById(detailId);
    const arrow = document.getElementById(arrowId);
    if (!detail) return;
    const isHidden = detail.style.display === 'none';
    detail.style.display = isHidden ? '' : 'none';
    if (arrow) arrow.style.transform = isHidden ? 'rotate(90deg)' : 'rotate(0deg)';
}

async function startSelftest() {
    const btn = document.getElementById('btnStartTests');
    const info = document.getElementById('testProgressInfo');
    if (btn) btn.disabled = true;
    if (info) info.textContent = 'Tests werden gestartet...';

    try {
        const resp = await fetch('/api/tests/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || data.error || data.message || 'Fehler');
        }
        if (info) info.textContent = 'Tests laufen...';

        const progressSection = document.getElementById('testProgressSection');
        if (progressSection) {
            progressSection.innerHTML = renderSelftestProgress({
                progress: 0, total: 0, passed: 0, failed: 0, errors: 0, skipped: 0, current_test: '',
            });
        }
        const resultSection = document.getElementById('testResultSection');
        if (resultSection) resultSection.innerHTML = '';

        pollSelftestStatus();
    } catch (e) {
        if (btn) btn.disabled = false;
        if (info) info.textContent = '';
        showNotification(e.message || 'Fehler beim Start', 'error');
    }
}

function pollSelftestStatus() {
    _selftestStopPolling();
    const mySeq = _selftestInitSeq;

    const tick = async () => {
        if (mySeq !== _selftestInitSeq) { _selftestStopPolling(); return; }
        if (!document.getElementById('btnStartTests')) { _selftestStopPolling(); return; }

        try {
            const resp = await fetch('/api/tests/status');
            if (!resp.ok) return;
            const data = await resp.json();
            if (mySeq !== _selftestInitSeq) { _selftestStopPolling(); return; }

            if (data.running) {
                updateSelftestProgress(data);
            } else {
                _selftestStopPolling();
                onSelftestFinished(data);
            }
        } catch (e) {
            // next tick retries
        }
    };
    _selftestPollTimer = setInterval(tick, 1500);
}

function updateSelftestProgress(data) {
    const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;

    const bar = document.getElementById('testBar');
    if (bar) bar.style.width = pct + '%';

    const label = document.getElementById('testBarLabel');
    if (label) label.textContent = `${data.progress}/${data.total}  (${pct}%)`;

    const name = document.getElementById('testCurrentName');
    if (name) name.textContent = data.current_test || '';

    const counters = document.getElementById('testLiveCounters');
    if (counters) counters.innerHTML = renderSelftestCounters(data);
}

function onSelftestFinished(data) {
    const btn = document.getElementById('btnStartTests');
    const info = document.getElementById('testProgressInfo');
    if (btn) btn.disabled = false;
    if (info) info.textContent = '';

    const progressSection = document.getElementById('testProgressSection');
    if (progressSection) progressSection.innerHTML = '';

    const resultSection = document.getElementById('testResultSection');
    if (resultSection) {
        resultSection.innerHTML = renderSelftestResults(data);
        _bindSelftestToggles(resultSection);
    }

    if (data.error) {
        showNotification('Selbsttest fehlgeschlagen: ' + data.error, 'error');
    } else if (data.progress === 0 && data.total > 0) {
        showNotification('Selbsttest abgebrochen: Kein Test wurde ausgefuehrt (Collection-Fehler?)', 'error');
    } else if (data.failed > 0 || data.errors > 0) {
        showNotification(`Selbsttest: ${data.failed + data.errors} Test(s) fehlgeschlagen`, 'error');
    } else {
        showNotification(`Selbsttest: Alle ${data.passed} Tests bestanden (${data.duration}s)`, 'success');
    }
}
