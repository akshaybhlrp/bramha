class BramhaAPI {
    constructor(baseUrl) { this.baseUrl = baseUrl; }

    async _fetch(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const res = await fetch(url, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ message: `HTTP ${res.status}` }));
            throw new Error(err.message);
        }
        return res.json();
    }

    async getStatus() { return this._fetch('/system/status'); }
    async getLogs(since = 0) { return this._fetch(`/system/logs?since=${since}`); }
    async getModels() { return this._fetch('/llm/models'); }
    async getCollections() { return this._fetch('/collections'); }

    async ingestModel(name, path) {
        return this._fetch('/llm/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: name, path }),
        });
    }

    async generate(model, prompt, device) {
        return this._fetch('/llm/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: model, prompt, device,
                max_new_tokens: 512, temperature: 0.7,
            }),
        });
    }

    async createCollection(name, dimension) {
        return this._fetch('/collections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, dimension, metric: 'Cosine' }),
        });
    }

    async getSpandaStatus() { return this._fetch('/system/spanda/status'); }
    async setSpandaDegraded(degraded) {
        return this._fetch('/system/spanda/degraded', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ degraded }),
        });
    }

    async getStats() { return this._fetch('/stats'); }
}

document.addEventListener('DOMContentLoaded', () => {
    const api = new BramhaAPI('http://localhost:8000/api');

    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-prompt');
    const sendBtn = document.getElementById('send-chat-btn');
    const modelSelect = document.getElementById('chat-model-select');
    const deviceSelect = document.getElementById('chat-device-select');
    const logMonitor = document.getElementById('log-monitor');
    const serverDot = document.getElementById('server-status-dot');
    const serverText = document.getElementById('server-status-text');
    const clearBtn = document.getElementById('clear-chat-btn');

    let lastLogTime = 0;

    // --- Navigation ---
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            item.classList.add('active');
            document.getElementById(item.dataset.view).classList.add('active');
        });
    });

    // --- Chat ---
    sendBtn.addEventListener('click', handleSend);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
    });
    clearBtn.addEventListener('click', () => {
        chatHistory.innerHTML = `<div class="welcome-screen"><div class="logo-symbol large">B</div><h2>Bramha Assistant</h2><p>Select a model to begin.</p></div>`;
    });

    async function handleSend() {
        const prompt = chatInput.value.trim();
        const model = modelSelect.value;
        if (!prompt) return;
        if (!model) { showToast('Select model first', 'error'); return; }

        appendMsg('user', prompt);
        chatInput.value = '';
        chatInput.disabled = sendBtn.disabled = true;

        try {
            const res = await api.generate(model, prompt, deviceSelect.value);
            appendMsg('assistant', res.completion);
        } catch (err) {
            appendMsg('assistant', `Error: ${err.message}`);
        } finally {
            chatInput.disabled = sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    function appendMsg(role, text) {
        if (chatHistory.querySelector('.welcome-screen')) chatHistory.innerHTML = '';
        const el = document.createElement('div');
        el.className = `chat-message ${role}`;
        el.textContent = text;
        chatHistory.appendChild(el);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function showToast(msg, type = 'success') {
        const t = document.getElementById('toast');
        t.textContent = msg;
        t.className = `toast show ${type}`;
        setTimeout(() => t.className = 'toast', 3000);
    }

    // --- Polling ---
    async function pollLogs() {
        try {
            const logs = await api.getLogs(lastLogTime);
            if (logs.length) {
                logs.forEach(l => {
                    const el = document.createElement('div');
                    el.textContent = `[${new Date(l.time).toLocaleTimeString()}] ${l.message}`;
                    logMonitor.appendChild(el);
                });
                lastLogTime = logs[logs.length - 1].time;
                logMonitor.scrollTop = logMonitor.scrollHeight;
            }
            serverDot.classList.add('online');
            serverText.textContent = 'Connected';
        } catch {
            serverDot.classList.remove('online');
            serverText.textContent = 'Offline';
        }
        setTimeout(pollLogs, 5000);
    }
    pollLogs();

    // --- Initial data ---
    async function init() {
        try {
            const [models, collections] = await Promise.all([api.getModels(), api.getCollections()]);
            modelSelect.innerHTML = '<option value="">Select model</option>';
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.name; opt.textContent = m.name;
                modelSelect.appendChild(opt);
            });

            const cl = document.getElementById('collections-list');
            if (cl) {
                cl.innerHTML = collections.map(c =>
                    `<div class="list-item">${c.name} (${c.vector_count || 0} vectors)</div>`
                ).join('');
            }

            // Engine status
            const eng = document.getElementById('engine-status-list');
            if (eng) {
                eng.innerHTML = `
                    <div class="list-item"><span>WGPU Dense</span><span style="color:var(--success-color)">● Healthy</span></div>
                    <div class="list-item"><span>SPANDA Sparse</span><span id="spanda-status" style="color:var(--text-muted)">● Checking...</span></div>
                `;
            }

            const spanda = await api.getSpandaStatus();
            const ss = document.getElementById('spanda-status');
            if (ss) {
                ss.textContent = spanda.healthy && !spanda.degraded ? '● Healthy' : '● Degraded';
                ss.style.color = spanda.degraded ? 'var(--error-color)' : 'var(--success-color)';
            }

            // Local models list
            const lm = document.getElementById('local-models-list');
            if (lm) {
                lm.innerHTML = models.map(m => `<div class="list-item">${m.name}</div>`).join('');
            }
        } catch (e) {
            console.error('init failed', e);
        }
    }
    init();

    // --- Ingest model ---
    document.getElementById('ingest-model-btn')?.addEventListener('click', async () => {
        const name = document.getElementById('model-name-input').value.trim();
        const path = document.getElementById('model-path-input').value.trim();
        if (!name || !path) { showToast('Name and path required', 'error'); return; }
        try {
            await api.ingestModel(name, path);
            showToast('Model ingested');
            document.getElementById('model-name-input').value = '';
            document.getElementById('model-path-input').value = '';
            init();
        } catch (e) { showToast(e.message, 'error'); }
    });

    // --- SPANDA controls ---
    document.getElementById('toggle-spanda-btn')?.addEventListener('click', async () => {
        try {
            const s = await api.getSpandaStatus();
            await api.setSpandaDegraded(!s.degraded);
            showToast(s.degraded ? 'SPANDA recovered' : 'SPANDA degraded');
            init();
        } catch (e) { showToast(e.message, 'error'); }
    });

    document.getElementById('run-spanda-test-btn')?.addEventListener('click', () => {
        deviceSelect.value = 'sparse';
        modelSelect.value && (
            chatInput.value = 'Compare block-sparse 2:4 weights vs dense matrix multiplications.',
            handleSend()
        );
    });

    // --- Create collection ---
    document.getElementById('create-collection-btn')?.addEventListener('click', async () => {
        const name = document.getElementById('collection-name-input').value.trim();
        const dim = parseInt(document.getElementById('collection-dim-input').value);
        if (!name || !dim) { showToast('Name and dimension required', 'error'); return; }
        try {
            await api.createCollection(name, dim);
            showToast('Collection created');
            document.getElementById('collection-name-input').value = '';
            init();
        } catch (e) { showToast(e.message, 'error'); }
    });
});
