/**
 * app.js — Interactive AR Sign Language Platform Client
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const video = document.getElementById('webcam-video');
  const canvas = document.getElementById('hud-canvas');
  const ctx = canvas.getContext('2d');

  const btnCameraToggle = document.getElementById('btn-camera-toggle');
  const btnModeToggle = document.getElementById('btn-mode-toggle');
  const btnArToggle = document.getElementById('btn-ar-toggle');
  const btnAccept = document.getElementById('btn-accept');
  const btnFinalize = document.getElementById('btn-finalize');
  const btnReset = document.getElementById('btn-reset');

  const hudModeDisplay = document.getElementById('hud-mode-display');
  const hudPredLabel = document.getElementById('hud-pred-label');
  const hudPredConf = document.getElementById('hud-pred-conf');
  const hudSentenceDisplay = document.getElementById('hud-sentence-display');
  const hudPartnerBox = document.getElementById('hud-partner-box');
  const hudPartnerText = document.getElementById('hud-partner-text');
  const arVignetteLayer = document.getElementById('ar-vignette-layer');

  const dialogueTimeline = document.getElementById('dialogue-timeline');
  const partnerSpeechInput = document.getElementById('partner-speech-input');
  const btnPartnerMic = document.getElementById('btn-partner-mic');
  const btnPartnerSend = document.getElementById('btn-partner-send');
  const btnExportDialogue = document.getElementById('btn-export-dialogue');

  const geminiApiKeyInput = document.getElementById('gemini-api-key');
  const aiStatusBadge = document.getElementById('ai-status-badge');
  const toneButtons = document.querySelectorAll('.tone-btn');

  const vocabPillsContainer = document.getElementById('vocab-pills-container');
  const newWordInput = document.getElementById('new-word-input');
  const btnAddWord = document.getElementById('btn-add-word');
  const vocabCount = document.getElementById('vocab-count');
  const suggestionsContainer = document.getElementById('suggestions-container');
  const fpsText = document.getElementById('fps-text');

  // Application State
  let cameraActive = false;
  let stream = null;
  let mode = 'WORD'; // 'WORD' | 'LETTER'
  let arMode = true;
  let selectedTone = 'polite';
  let buildingWords = [];
  let currentPrediction = 'READY';
  let currentConfidence = 0.94;
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let recognition = null;
  let isListeningPartner = false;

  // Next Word Suggestions Table
  const suggestionsMap = {
    'I': ['WANT', 'NEED', 'AM', 'FEEL'],
    'NEED': ['HELP', 'WATER', 'FOOD', 'DOCTOR'],
    'WANT': ['WATER', 'MORE', 'FOOD', 'HELP'],
    'PLEASE': ['HELP', 'WAIT', 'COME', 'STOP'],
    'THANK': ['YOU'],
    'WHERE': ['IS', 'RESTROOM', 'DOCTOR'],
    'CALL': ['DOCTOR', 'POLICE', 'HELP'],
  };

  // Restore API key
  const savedKey = localStorage.getItem('gemini_api_key');
  if (savedKey) {
    geminiApiKeyInput.value = savedKey;
    aiStatusBadge.textContent = 'Gemini AI Active';
    aiStatusBadge.style.color = 'var(--accent-cyan)';
  }

  geminiApiKeyInput.addEventListener('input', (e) => {
    const val = e.target.value.trim();
    localStorage.setItem('gemini_api_key', val);
    if (val) {
      aiStatusBadge.textContent = 'Gemini AI Active';
      aiStatusBadge.style.color = 'var(--accent-cyan)';
    } else {
      aiStatusBadge.textContent = 'Smart Rules';
      aiStatusBadge.style.color = 'var(--accent-teal)';
    }
  });

  // Tone Selection
  toneButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      toneButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedTone = btn.dataset.tone;
    });
  });

  // ── Camera Management ──
  async function toggleCamera() {
    if (cameraActive) {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
      video.srcObject = null;
      cameraActive = false;
      btnCameraToggle.textContent = 'Start Camera';
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' }
      });
      video.srcObject = stream;
      cameraActive = true;
      btnCameraToggle.textContent = 'Stop Camera';
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        requestAnimationFrame(renderLoop);
      };
    } catch (err) {
      alert('Could not access webcam: ' + err.message);
    }
  }

  btnCameraToggle.addEventListener('click', toggleCamera);

  // ── Mode Toggle ──
  btnModeToggle.addEventListener('click', () => {
    mode = mode === 'WORD' ? 'LETTER' : 'WORD';
    btnModeToggle.textContent = `Mode: ${mode}`;
    hudModeDisplay.textContent = `◈ ${mode} MODE`;
    updateSimulation();
  });

  // ── AR Toggle ──
  btnArToggle.addEventListener('click', () => {
    arMode = !arMode;
    btnArToggle.textContent = `AR HUD: ${arMode ? 'ON' : 'OFF'}`;
    arVignetteLayer.style.display = arMode ? 'block' : 'none';
  });

  // ── Word Acceptance & Actions ──
  function acceptCurrent() {
    if (currentPrediction && currentPrediction !== '---') {
      buildingWords.push(currentPrediction);
      updateDisplay();
      updateSuggestions();
    }
  }

  function deleteLast() {
    if (buildingWords.length > 0) {
      buildingWords.pop();
      updateDisplay();
      updateSuggestions();
    }
  }

  function resetAll() {
    buildingWords = [];
    hudSentenceDisplay.textContent = 'Waiting for gestures...';
    updateSuggestions();
  }

  async function finalizeSentence() {
    if (buildingWords.length === 0) return;

    hudSentenceDisplay.textContent = 'Polishing sentence...';
    const apiKey = geminiApiKeyInput.value.trim();

    try {
      const resp = await fetch('/api/expand', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          words: buildingWords,
          tone: selectedTone,
          api_key: apiKey || null
        })
      });

      const data = await resp.json();
      const sentence = data.expanded_sentence || buildingWords.join(' ');
      hudSentenceDisplay.textContent = sentence;

      // Speak sentence
      speak(sentence);

      // Record in dialogue timeline
      await fetch('/api/dialogue/signer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: sentence,
          raw_signs: buildingWords,
          tone: selectedTone
        })
      });

      addDialogueTurn('Signer', sentence, selectedTone);
      buildingWords = [];
      updateSuggestions();
    } catch (err) {
      console.error('Finalize error:', err);
      hudSentenceDisplay.textContent = buildingWords.join(' ');
      speak(buildingWords.join(' '));
      buildingWords = [];
    }
  }

  btnAccept.addEventListener('click', acceptCurrent);
  btnFinalize.addEventListener('click', finalizeSentence);
  btnReset.addEventListener('click', resetAll);

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    if (e.code === 'Space') {
      e.preventDefault();
      acceptCurrent();
    } else if (e.key === 'd' || e.key === 'D') {
      deleteLast();
    } else if (e.key === 'm' || e.key === 'M') {
      btnModeToggle.click();
    } else if (e.key === 'g' || e.key === 'G') {
      btnArToggle.click();
    } else if (e.key === 'r' || e.key === 'R') {
      resetAll();
    } else if (e.key === 'Enter') {
      finalizeSentence();
    }
  });

  // ── Speech Synthesis ──
  function speak(text) {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }

  // ── Two-Way Speech Recognition (Hearing Partner) ──
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map(r => r[0].transcript)
        .join('');
      partnerSpeechInput.value = transcript;
    };

    recognition.onend = () => {
      isListeningPartner = false;
      btnPartnerMic.style.background = 'var(--accent-purple)';
      if (partnerSpeechInput.value.trim()) {
        sendPartnerSpeech(partnerSpeechInput.value.trim());
      }
    };

    recognition.onerror = () => {
      isListeningPartner = false;
      btnPartnerMic.style.background = 'var(--accent-purple)';
    };

    btnPartnerMic.addEventListener('click', () => {
      if (isListeningPartner) {
        recognition.stop();
      } else {
        partnerSpeechInput.value = '';
        recognition.start();
        isListeningPartner = true;
        btnPartnerMic.style.background = 'var(--accent-red)';
      }
    });
  } else {
    btnPartnerMic.title = 'Speech recognition not supported in this browser';
    btnPartnerMic.disabled = true;
  }

  async function sendPartnerSpeech(text) {
    if (!text.trim()) return;

    // Show floating partner subtitle on AR HUD
    hudPartnerText.textContent = text;
    hudPartnerBox.style.display = 'block';
    setTimeout(() => {
      hudPartnerBox.style.display = 'none';
    }, 8000);

    // Save turn
    await fetch('/api/dialogue/partner', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    addDialogueTurn('Partner', text, 'spoken');
    partnerSpeechInput.value = '';
  }

  btnPartnerSend.addEventListener('click', () => {
    sendPartnerSpeech(partnerSpeechInput.value.trim());
  });

  partnerSpeechInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      sendPartnerSpeech(partnerSpeechInput.value.trim());
    }
  });

  function addDialogueTurn(speaker, text, tone) {
    const turnDiv = document.createElement('div');
    turnDiv.className = `timeline-turn ${speaker === 'Signer' ? 'turn-signer' : 'turn-partner'}`;
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    turnDiv.innerHTML = `
      <div class="turn-author">
        <span>${speaker}</span>
        <span>${timeStr}</span>
      </div>
      <div>${text}</div>
    `;
    dialogueTimeline.appendChild(turnDiv);
    dialogueTimeline.scrollTop = dialogueTimeline.scrollHeight;
  }

  // ── Suggestions & Display ──
  function updateDisplay() {
    if (buildingWords.length === 0) {
      hudSentenceDisplay.textContent = 'Waiting for gestures...';
    } else {
      hudSentenceDisplay.textContent = buildingWords.join(' ');
    }
  }

  function updateSuggestions() {
    suggestionsContainer.innerHTML = '';
    const lastWord = buildingWords.length > 0 ? buildingWords[buildingWords.length - 1] : 'I';
    const suggestions = suggestionsMap[lastWord] || ['HELP', 'WATER', 'PLEASE', 'MORE'];

    suggestions.slice(0, 4).forEach((word, idx) => {
      const pill = document.createElement('span');
      pill.className = 'sug-pill';
      pill.textContent = `${idx + 1}. ${word}`;
      pill.addEventListener('click', () => {
        buildingWords.push(word);
        updateDisplay();
        updateSuggestions();
      });
      suggestionsContainer.appendChild(pill);
    });
  }

  // ── Vocabulary Management ──
  async function loadVocab() {
    try {
      const resp = await fetch('/api/vocabulary');
      const data = await resp.json();
      const words = data.words || [];
      vocabCount.textContent = `${words.length} Words`;
      vocabPillsContainer.innerHTML = '';

      words.forEach(word => {
        const pill = document.createElement('span');
        pill.className = 'sug-pill';
        pill.style.background = 'rgba(255, 255, 255, 0.05)';
        pill.style.borderColor = 'var(--border-glass)';
        pill.style.color = '#fff';
        pill.textContent = word;
        pill.title = 'Click to simulate prediction';
        pill.addEventListener('click', () => {
          currentPrediction = word;
          hudPredLabel.textContent = word;
          hudPredConf.textContent = '96% MATCH';
        });
        vocabPillsContainer.appendChild(pill);
      });
    } catch (err) {
      console.error('Failed to load vocabulary:', err);
    }
  }

  btnAddWord.addEventListener('click', async () => {
    const word = newWordInput.value.trim().toUpperCase();
    if (!word) return;

    try {
      const resp = await fetch('/api/vocabulary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word })
      });
      if (resp.ok) {
        newWordInput.value = '';
        loadVocab();
      }
    } catch (err) {
      alert('Error adding word: ' + err.message);
    }
  });

  // Export Dialogue
  btnExportDialogue.addEventListener('click', async () => {
    window.open('/api/dialogue/export?format=text', '_blank');
  });

  // ── Canvas AR HUD Drawing ──
  function drawCornerBrackets(w, h) {
    const margin = 24;
    const len = 32;
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2.5;

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(margin, margin + len);
    ctx.lineTo(margin, margin);
    ctx.lineTo(margin + len, margin);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(w - margin - len, margin);
    ctx.lineTo(w - margin, margin);
    ctx.lineTo(w - margin, margin + len);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(margin, h - margin - len);
    ctx.lineTo(margin, h - margin);
    ctx.lineTo(margin + len, h - margin);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(w - margin - len, h - margin);
    ctx.lineTo(w - margin, h - margin);
    ctx.lineTo(w - margin, h - margin - len);
    ctx.stroke();
  }

  function renderLoop(timestamp) {
    if (!cameraActive) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (arMode) {
      drawCornerBrackets(canvas.width, canvas.height);
    }

    // FPS calculation
    frameCount++;
    const elapsed = timestamp - lastFrameTime;
    if (elapsed >= 1000) {
      const fps = Math.round((frameCount * 1000) / elapsed);
      fpsText.textContent = `FPS: ${fps}`;
      frameCount = 0;
      lastFrameTime = timestamp;
    }

    requestAnimationFrame(renderLoop);
  }

  function updateSimulation() {
    const vocab = ['HELP', 'WATER', 'PLEASE', 'THANK YOU', 'NEED', 'YES', 'NO'];
    const letters = ['A', 'B', 'C', 'D', 'H', 'E', 'L', 'P'];
    const pool = mode === 'WORD' ? vocab : letters;
    currentPrediction = pool[Math.floor(Math.random() * pool.length)];
    hudPredLabel.textContent = currentPrediction;
  }

  // Initial load
  loadVocab();
  updateSuggestions();
  setInterval(updateSimulation, 4000);
});
