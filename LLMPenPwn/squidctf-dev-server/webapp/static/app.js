const SquidUI = (() => {
  const els = {};
  let defaultDataset = null;
  let ws = null;
  let currentLogCid = null;
  let logSize = 0;
  let loadedStart = 0; // byte offset of first loaded char
  let loadedEnd = 0;   // byte offset after last loaded char
  let isPrepending = false;

  async function fetchJSON(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function fetchDefaultDataset() {
    try {
      const data = await fetchJSON('/api/datasets');
      defaultDataset = data.default || (data.datasets && data.datasets[0]) || null;
    } catch (e) {
      defaultDataset = null;
    }
  }

  async function loadChallenges() {
    const ds = defaultDataset;
    const all = await fetchJSON(`/api/challenges?dataset=${encodeURIComponent(ds)}`);
    const entries = all[ds] || {};
    const tbody = els.challengesTable.querySelector('tbody');
    tbody.innerHTML = '';
    for (const [id, info] of Object.entries(entries)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="px-3 py-2">${id}</td><td class="px-3 py-2">${info.category || ''}</td><td class="px-3 py-2">${info.path || ''}</td>`;
      const tdRun = document.createElement('td');
      tdRun.className = 'px-3 py-2';
      const btn = document.createElement('button');
      btn.className = 'px-2.5 py-1.5 rounded bg-brand-600 hover:brightness-110 text-white';
      btn.textContent = 'Run';
      btn.addEventListener('click', () => runChallenge(id));
      tdRun.appendChild(btn);
      tr.appendChild(tdRun);
      tbody.appendChild(tr);
    }
  }

  async function runChallenge(challengeId) {
    const ds = defaultDataset;
    try {
      await fetchJSON('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ challenge_id: challengeId, dataset: ds })
      });
      // refresh soon
      setTimeout(loadStatus, 500);
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  async function stopChallenge(challengeId) {
    try {
      await fetchJSON('/api/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ challenge_id: challengeId })
      });
      // Refresh status soon
      setTimeout(loadStatus, 500);
    } catch (e) {
      alert(e.message || String(e));
    }
  }

  async function loadStatus() {
    const data = await fetchJSON('/api/status');
    const tbody = els.statusTable.querySelector('tbody');
    tbody.innerHTML = '';
    for (const st of data.statuses) {
      const tr = document.createElement('tr');
      const cid = st.challenge_id || 'n/a';
      const state = st.state || 'pending';
      const step = st.step || 0;
      const result = st.result || 'pending';
      const restarts = st.restart_count || 0;
      
      // Create actions cell with View and optionally Stop button
      const tdActions = document.createElement('td');
      tdActions.className = 'px-3 py-2';
      
      const openBtn = document.createElement('button');
      openBtn.className = 'px-2.5 py-1.5 rounded bg-slate-700 hover:bg-slate-600 mr-2';
      openBtn.textContent = 'View';
      openBtn.addEventListener('click', () => openLogViewer(cid));
      tdActions.appendChild(openBtn);
      
      // Add Stop button for running or starting challenges
      if (state === 'running' || state === 'starting') {
        const stopBtn = document.createElement('button');
        stopBtn.className = 'px-2.5 py-1.5 rounded bg-red-600 hover:bg-red-700 text-white';
        stopBtn.textContent = 'Stop';
        stopBtn.addEventListener('click', () => {
          if (confirm(`Stop challenge "${cid}"?`)) {
            stopChallenge(cid);
          }
        });
        tdActions.appendChild(stopBtn);
      }
      
      tr.innerHTML = `<td class="px-3 py-2">${cid}</td><td class="px-3 py-2">${state}</td><td class="px-3 py-2">${step}</td><td class="px-3 py-2">${result}</td><td class="px-3 py-2">${restarts}</td>`;
      tr.appendChild(tdActions);
      tbody.appendChild(tr);
    }
  }

  async function loadFlags() {
    try {
      const data = await fetchJSON('/api/flags');
      const tbody = els.flagsTable.querySelector('tbody');
      tbody.innerHTML = '';
      for (const f of data.flags) {
        const tr = document.createElement('tr');
        const finished = f.time ? new Date(f.time * 1000).toLocaleString() : '';
        tr.innerHTML = `
          <td class="px-3 py-2">${f.challenge_id}</td>
          <td class="px-3 py-2 font-mono break-all">${f.flag}</td>
          <td class="px-3 py-2">${f.success ? 'yes' : 'no'}</td>
          <td class="px-3 py-2">${finished}</td>
        `;
        tbody.appendChild(tr);
      }
    } catch (e) {
      // ignore transient errors
    }
  }

  // --- ANSI rendering ---
  function ansiToHtml(text) {
    // Basic SGR support
    const ESC = '\u001b[';
    const parts = text.split(/(\u001b\[[0-9;]*m)/);
    let html = '';
    let open = [];
    function openSpan(cls) { open.push(cls); html += `<span class="${cls}">`; }
    function closeAll() { while (open.length) { html += '</span>'; open.pop(); } }
    for (const p of parts) {
      if (p.startsWith(ESC) && p.endsWith('m')) {
        const codes = p.slice(2, -1).split(';').map(x => parseInt(x || '0', 10));
        if (codes.length === 0) { closeAll(); continue; }
        for (const c of codes) {
          if (c === 0) { closeAll(); }
          else if (c === 1) { openSpan('ansi-bold'); }
          else if (c === 3) { openSpan('ansi-italic'); }
          else if (c === 4) { openSpan('ansi-underline'); }
          else if (30 <= c && c <= 37) {
            const map = ['ansi-black','ansi-red','ansi-green','ansi-yellow','ansi-blue','ansi-magenta','ansi-cyan','ansi-white'];
            openSpan(map[c-30]);
          } else if (40 <= c && c <= 47) {
            const map = ['ansi-bg-black','ansi-bg-red','ansi-bg-green','ansi-bg-yellow','ansi-bg-blue','ansi-bg-magenta','ansi-bg-cyan','ansi-bg-white'];
            openSpan(map[c-40]);
          }
        }
      } else {
        html += p.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      }
    }
    closeAll();
    return html;
  }

  function appendLog(text) {
    // Split to handle large chunks
    const lines = text.split(/(?<=\n)/);
    for (const line of lines) {
      if (!line) continue;
      els.logPre.insertAdjacentHTML('beforeend', ansiToHtml(line));
    }
    if (els.autoScroll.checked) {
      els.logPre.parentElement.scrollTop = els.logPre.parentElement.scrollHeight;
    }
  }

  function prependLog(text) {
    if (!text) return;
    const container = els.logPre.parentElement;
    const prevScrollHeight = container.scrollHeight;
    els.logPre.insertAdjacentHTML('afterbegin', ansiToHtml(text));
    // Maintain viewport position to prevent jump
    const newScrollHeight = container.scrollHeight;
    container.scrollTop += (newScrollHeight - prevScrollHeight);
  }

  async function fetchLogInfo(cid) {
    const res = await fetchJSON(`/api/logs/info/${encodeURIComponent(cid)}`);
    return res.size || 0;
  }

  async function fetchLogChunk(cid, offset, length) {
    const url = `/api/logs/chunk/${encodeURIComponent(cid)}?offset=${offset}&length=${length}`;
    return await fetchJSON(url);
  }

  async function loadInitialLog(cid) {
    logSize = await fetchLogInfo(cid);
    const tailBytes = 1 << 16; // 64KB
    const start = Math.max(0, logSize - tailBytes);
    const { data } = await fetchLogChunk(cid, start, logSize - start);
    loadedStart = start;
    loadedEnd = start + (data ? data.length : 0);
    els.logPre.innerHTML = '';
    prependLog(data || '');
    // Snap to bottom
    els.logPre.parentElement.scrollTop = els.logPre.parentElement.scrollHeight;
  }

  async function maybePrependMore() {
    if (isPrepending) return;
    if (loadedStart <= 0) return;
    const container = els.logPre.parentElement;
    if (container.scrollTop > 80) return; // only when near top
    isPrepending = true;
    try {
      const chunk = 1 << 16; // 64KB
      const wantStart = Math.max(0, loadedStart - chunk);
      const { data } = await fetchLogChunk(currentLogCid, wantStart, loadedStart - wantStart);
      loadedStart = wantStart;
      prependLog(data || '');
    } finally {
      isPrepending = false;
    }
  }

  function openLogViewer(challengeId) {
    currentLogCid = challengeId;
    els.logTitle.textContent = `Log: ${challengeId}`;
    els.logPre.innerHTML = '';
    els.logModal.classList.remove('hidden');
    loadInitialLog(challengeId).then(() => {
      // Gentle boost to stick to bottom initially
      const boostUntil = Date.now() + 1200;
      const booster = setInterval(() => {
        if (!els.autoScroll.checked || Date.now() > boostUntil || els.logModal.classList.contains('hidden')) {
          clearInterval(booster);
          return;
        }
        els.logPre.parentElement.scrollTop = els.logPre.parentElement.scrollHeight;
      }, 150);
      openLogWS(challengeId);
    });
  }

  function closeLogViewer() {
    els.logModal.classList.add('hidden');
    if (ws) { try { ws.close(); } catch(e){} ws = null; }
  }

  function openLogWS(challengeId) {
    if (ws) { try { ws.close(); } catch(e){} }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/logs/${encodeURIComponent(challengeId)}`;
    ws = new WebSocket(url);
    ws.onmessage = (ev) => {
      appendLog(ev.data);
    };
    ws.onclose = () => {
      // Try to reconnect while the modal is open
      if (!els.logModal.classList.contains('hidden') && currentLogCid === challengeId) {
        setTimeout(() => openLogWS(challengeId), 1500);
      }
    };
  }

  async function onAddChallenge(e) {
    e.preventDefault();
    const fd = new FormData();
    fd.append('challenge_id', els.challengeId.value);
    fd.append('category', els.category.value);
    if (defaultDataset) fd.append('dataset', defaultDataset);
    const desc = (els.description.value || '').trim();
    if (desc) fd.append('description', desc);
    // Append multiple loose files if provided
    if (els.files && els.files.files && els.files.files.length) {
      for (const f of els.files.files) {
        fd.append('files', f);
      }
    }
    const res = await fetch('/api/challenges', { method: 'POST', body: fd });
    if (!res.ok) { alert(await res.text()); return; }
    await loadChallenges();
  }

  function initDom() {
    els.refreshChallenges = document.getElementById('refreshChallenges');
    els.challengesTable = document.getElementById('challengesTable');
    els.statusTable = document.getElementById('statusTable');
    els.flagsTable = document.getElementById('flagsTable');
    els.refreshFlags = document.getElementById('refreshFlags');
    els.logModal = document.getElementById('logModal');
    els.logPre = document.getElementById('logPre');
    els.closeLog = document.getElementById('closeLog');
    els.logTitle = document.getElementById('logTitle');
    els.autoScroll = document.getElementById('autoScroll');
    els.addChallengeForm = document.getElementById('addChallengeForm');
    els.challengeId = document.getElementById('challengeId');
    els.category = document.getElementById('category');
    els.files = document.getElementById('files');
    els.description = document.getElementById('description');

    els.refreshChallenges.addEventListener('click', loadChallenges);
    els.addChallengeForm.addEventListener('submit', onAddChallenge);
    els.closeLog.addEventListener('click', closeLogViewer);
    els.logPre.parentElement.addEventListener('scroll', maybePrependMore);

    setInterval(loadStatus, 2000);
    setInterval(loadFlags, 4000);
    els.refreshFlags.addEventListener('click', loadFlags);
  }

  async function init() {
    initDom();
    await fetchDefaultDataset();
    await loadChallenges();
    await loadStatus();
    await loadFlags();
  }

  return { init };
})();

window.SquidUI = SquidUI;


