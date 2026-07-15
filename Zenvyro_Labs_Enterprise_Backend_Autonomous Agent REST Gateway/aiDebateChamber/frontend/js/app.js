// --- PRODUCTION LOGIC (CONNECTS TO BACKEND) ---
let currentTopic = "";
let lastSpeaker = null;
let lastMessage = "";
let currentRound = 1;
let modelTrained = false;

// After this many full rounds (A + B = 1 round), the "Pass Turn" button
// turns into "Get ML Verdict" instead of generating another turn.
const MAX_ROUNDS = 3;

const topicInput = document.getElementById('topicInput');
const startBtn = document.getElementById('startBtn');
const nextBtn = document.getElementById('nextTurnBtn');
const feedA = document.getElementById('feedA');
const feedB = document.getElementById('feedB');

// New UI Selectors
const displayTopic = document.getElementById('displayTopic');
const networkStatus = document.getElementById('networkStatus');
const statusText = document.getElementById('statusText');
const dotA = document.getElementById('dotA');
const dotB = document.getElementById('dotB');
const globalDot = document.getElementById('globalDot');

// Judge overlay selectors
const judgeOverlay = document.getElementById('judgeOverlay');
const judgeCard = judgeOverlay ? judgeOverlay.querySelector('.judge-card') : null;

const API_BASE = "http://127.0.0.1:5000/api";
const DEBATE_URL = `${API_BASE}/debate`;
const ML_URL = `${API_BASE}/machine-learning`;

function toggleTyping(agent, show) {
    const feed = agent === 'A' ? feedA : feedB;
    const existing = document.getElementById('typingIndicator');

    if (show) {
        if (!existing) {
            const typingHTML = `<div class="typing-wrapper" id="typingIndicator"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
            feed.insertAdjacentHTML('beforeend', typingHTML);
            feed.scrollTop = feed.scrollHeight;
        }
    } else {
        if (existing) existing.remove();
    }
}

function setActive(agent) {
    if (agent === 'A') {
        dotA.classList.add('active');
        dotB.classList.remove('active');
        statusText.textContent = `Awaiting API Response for Agent A... (Round ${currentRound})`;
    } else {
        dotB.classList.add('active');
        dotA.classList.remove('active');
        statusText.textContent = `Awaiting API Response for Agent B... (Round ${currentRound})`;
    }
}

function appendMessage(agent, text, round) {
    toggleTyping(agent, false); // Clear typing before posting

    const div = document.createElement('div');
    div.classList.add('msg', agent === 'A' ? 'msg-adv' : 'msg-chal');
    div.innerHTML = `
        <div class="round-tag">Round ${round} · ${agent === 'A' ? 'Advocate' : 'Challenger'}</div>
        <div class="msg-text">${text}</div>
    `;

    const feed = agent === 'A' ? feedA : feedB;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
}

function setProcessingState(isLoading) {
    startBtn.disabled = isLoading;
    nextBtn.disabled = isLoading;
    if (isLoading) {
        topicInput.disabled = true;
        networkStatus.textContent = "COMPUTING";
        networkStatus.style.color = "#eab308";
        networkStatus.style.borderColor = "rgba(234,179,8,0.3)";
    } else {
        networkStatus.textContent = "IDLE";
        networkStatus.style.color = "#8b5cf6";
        networkStatus.style.borderColor = "rgba(139,92,246,0.3)";
    }
}

// Fire the training job once, in the background, right when the page
// loads. By the time the debate wraps up (a few turns later) the judge
// is almost always ready, so the final verdict doesn't have to wait on
// a model that hasn't been fit yet.
async function trainJudgeInBackground() {
    try {
        const response = await fetch(`${ML_URL}/train`, { method: 'POST' });
        const data = await response.json();
        if (data.metrics) {
            modelTrained = true;
            console.log("ML Judge trained:", data.metrics);
        }
    } catch (err) {
        console.warn("ML Judge training failed (will retry before verdict):", err);
    }
}

async function ensureJudgeTrained() {
    if (modelTrained) return;
    statusText.textContent = "Training ML Judge on historical data...";
    await trainJudgeInBackground();
}

function renderVerdict(data) {
    if (!judgeCard) {
        // Fallback if the overlay markup isn't present for some reason.
        alert(`Winner: ${data.winner}\nAdvocate: ${data.advocate_score}\nChallenger: ${data.challenger_score}`);
        return;
    }

    judgeCard.innerHTML = `
        <div class="judge-header">
            <div class="judge-icon">⚖️</div>
            <h2>ML Verdict: ${data.winner}</h2>
            <p>Scikit-Learn RandomForestRegressor evaluation complete</p>
        </div>
        <div class="judge-scores" style="display:flex; gap:24px; justify-content:center; margin-top:16px;">
            <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8;">ADVOCATE (A)</div>
                <div style="font-size:2rem; font-weight:800; color:#8b5cf6;">${data.advocate_score}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8;">CHALLENGER (B)</div>
                <div style="font-size:2rem; font-weight:800; color:#22c55e;">${data.challenger_score}</div>
            </div>
        </div>
        <button class="btn-reset" onclick="location.reload()" style="margin-top: 20px;">Reset API Connection</button>
    `;

    judgeOverlay.classList.add('active');
    judgeOverlay.style.display = 'flex';
}

async function getVerdict() {
    setProcessingState(true);
    statusText.textContent = "ML Judge evaluating the full transcript...";

    try {
        await ensureJudgeTrained();

        const response = await fetch(`${ML_URL}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Backend already keeps its own full transcript per agent, so
            // an empty body works too — but we pass topic along for logging.
            body: JSON.stringify({ topic: currentTopic })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Evaluation failed.');
        }

        const data = await response.json();
        renderVerdict(data);
        statusText.textContent = "Verdict delivered.";
    } catch (err) {
        console.error(err);
        alert(`ML Judge failed to reach a verdict: ${err.message}`);
    } finally {
        setProcessingState(false);
    }
}

startBtn.addEventListener('click', async () => {
    const topic = topicInput.value.trim();
    if (!topic) return alert("Please enter a custom topic first.");

    currentTopic = topic;
    displayTopic.textContent = `"${topic}"`;
    feedA.innerHTML = '';
    feedB.innerHTML = '';
    currentRound = 1;
    nextBtn.textContent = 'Pass Turn ➔';

    setProcessingState(true);
    globalDot.classList.add('live');

    // Kick off model training in parallel with the opening statement —
    // no need to make the user wait for it up front.
    trainJudgeInBackground();

    setActive('A');
    toggleTyping('A', true);

    try {
        const response = await fetch(`${DEBATE_URL}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic: currentTopic })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Failed to start debate.');
        }

        const data = await response.json();

        lastSpeaker = data.agent || "A";
        lastMessage = data.message;

        appendMessage(lastSpeaker, lastMessage, currentRound);

        setProcessingState(false);
        dotA.classList.remove('active');
        statusText.textContent = "API Idle. Waiting for User Execution...";

    } catch (err) {
        console.error("Backend connection failed.", err);
        alert(`Failed to connect to the AI Backend Python Server on port 5000!\n${err.message}`);
        setProcessingState(false);
        globalDot.classList.remove('live');
        toggleTyping('A', false);
    }
});

nextBtn.addEventListener('click', async () => {
    // Once we've hit the round cap, this button's job switches to
    // triggering the ML verdict instead of generating another turn.
    if (currentRound > MAX_ROUNDS) {
        return getVerdict();
    }

    setProcessingState(true);

    // Switch to whichever agent DID NOT speak last
    const nextAgent = lastSpeaker === 'A' ? 'B' : 'A';
    setActive(nextAgent);
    toggleTyping(nextAgent, true);

    try {
        const response = await fetch(`${DEBATE_URL}/next-turn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic: currentTopic,
                last_speaker: lastSpeaker,
                last_message: lastMessage
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Agent failed to respond.');
        }

        const data = await response.json();

        lastSpeaker = data.agent || nextAgent;
        lastMessage = data.message;

        appendMessage(lastSpeaker, lastMessage, currentRound);

        if (lastSpeaker === 'B') currentRound++; // Increment round after B goes

        setProcessingState(false);
        dotA.classList.remove('active');
        dotB.classList.remove('active');

        if (currentRound > MAX_ROUNDS) {
            nextBtn.textContent = '⚖️ Get ML Verdict';
            statusText.textContent = `Debate complete after ${MAX_ROUNDS} rounds. Ready for judgment.`;
        } else {
            statusText.textContent = "API Idle. Waiting for User Execution...";
        }

    } catch (err) {
        console.error(err);
        alert(`Agent failed to respond.\n${err.message}`);
        setProcessingState(false);
        toggleTyping(nextAgent, false);
    }
});