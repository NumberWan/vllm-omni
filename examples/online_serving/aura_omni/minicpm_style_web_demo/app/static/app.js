(() => {
  'use strict';

  const config = window.AURA_WEB_CONFIG || {};
  const callButton = document.getElementById('callButton');
  const muteButton = document.getElementById('muteButton');
  const cameraButton = document.getElementById('cameraButton');
  const cameraPreview = document.getElementById('cameraPreview');
  const cameraPlaceholder = document.getElementById('cameraPlaceholder');
  const promptPreset = document.getElementById('promptPreset');
  const systemPromptInput = document.getElementById('systemPrompt');
  const connectionState = document.getElementById('connectionState');
  const modelState = document.getElementById('modelState');
  const playbackState = document.getElementById('playbackState');
  const sessionTimer = document.getElementById('sessionTimer');
  const meterFill = document.getElementById('meterFill');
  const conversation = document.getElementById('conversation');
  const emptyConversation = document.getElementById('emptyConversation');
  const eventLog = document.getElementById('eventLog');
  const eventCount = document.getElementById('eventCount');
  const runtimeDetail = document.getElementById('runtimeDetail');
  const clearLogButton = document.getElementById('clearLogButton');

  const INPUT_RATE = 16000;
  const OUTPUT_RATE = 24000;
  // Client-only speed. Default 1.0 = bit-identical to dump (WSOLA 1.5x was unstable).
  const PLAYBACK_SPEED = Number(config.playbackSpeed) > 0 ? Number(config.playbackSpeed) : 1.0;
  const INITIAL_PLAYBACK_BUFFER_MS = 500;
  const ECHO_GUARD_MS = 300;
  const AUDIO_CHUNK_BYTES = 16000 * 2;
  const SILENT_TOKEN = '<|silent|>';

  const PROMPT_PRESETS = {
    chinese_live:
      "You are receiving a live video stream where the final frame is the present moment. "
      + "Respond only when a response is needed based on the user's message or the visual context. "
      + "Otherwise, output '<|silent|>' to signify silence. Respond in Chinese.",
    english_live:
      "You are receiving a live video stream where the final frame is the present moment. "
      + "Respond only when a response is needed based on the user's message or the visual context. "
      + "Otherwise, output '<|silent|>' to signify silence. Respond in English.",
  };

  let socket = null;
  let mediaStream = null;
  let captureContext = null;
  let captureNode = null;
  let playbackContext = null;
  let playbackNode = null;
  let clockTimer = null;
  let startedAt = 0;
  let running = false;
  let muted = false;
  let talking = false;
  let assistantActive = false;
  let captureRate = INPUT_RATE;
  let cameraStream = null;
  let cameraTimer = null;
  let playbackRate = OUTPUT_RATE;
  let pendingCapture = [];
  let responseHasAudio = false;
  // Pause camera while a spoken TTS turn is in flight so silent vision turns
  // do not hit Stage2 (max_num_seqs=1) and corrupt Talker output.
  let holdCameraForSpeech = false;
  let logCount = 0;
  let liveAssistantTurn = null;
  let assistantRawText = '';
  let pendingUserTurns = [];
  let framesSent = 0;
  let turnsSeen = 0;
  let activePlaybackRequestId = null;
  let lastCameraFrame = null;
  // Serialize decode→enqueue so parallel decodeAudioData cannot reorder chunks.
  let playbackChain = Promise.resolve();

  function staticAssetUrl(path) {
    const version = String(config.appVersion || '').trim();
    return version ? `${path}?v=${encodeURIComponent(version)}` : path;
  }

  if (promptPreset && systemPromptInput) {
    promptPreset.addEventListener('change', () => {
      const preset = PROMPT_PRESETS[promptPreset.value];
      if (preset !== undefined) systemPromptInput.value = preset;
    });
    systemPromptInput.addEventListener('input', () => {
      promptPreset.value = 'custom';
    });
  }

  function streamUrl() {
    const url = new URL(config.streamPath || 'v1/video/chat/stream', window.location.href);
    url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return url.toString();
  }

  function cameraIntervalMs() {
    const fps = Number(config.videoFps);
    if (!Number.isFinite(fps) || fps <= 0) return 500;
    return Math.max(100, Math.round(1000 / fps));
  }

  function setConnection(label, kind) {
    connectionState.textContent = label;
    connectionState.className = `status status-${kind}`;
  }

  function setModel(label) {
    // Push-to-talk hold owns the model chip; ignore Thinking/Speaking flicker.
    if (talking && label !== 'Recording') {
      modelState.textContent = 'Recording';
      return;
    }
    modelState.textContent = label;
  }

  function setPlayback(label) {
    playbackState.textContent = label;
  }

  function appendLog(message, error = false) {
    const time = new Date().toLocaleTimeString([], { hour12: false });
    const line = document.createElement('span');
    if (error) line.className = 'log-error';
    line.textContent = `${time}  ${message}\n`;
    eventLog.appendChild(line);
    eventLog.scrollTop = eventLog.scrollHeight;
    logCount += 1;
    eventCount.textContent = `${logCount} ${logCount === 1 ? 'event' : 'events'}`;
  }

  function appendEventLog(event) {
    const extras = [];
    if (event.text) extras.push(`text_len=${String(event.text).length}`);
    if (event.delta) extras.push(`delta_len=${String(event.delta).length}`);
    if (event.data) extras.push(`data_len=${String(event.data).length}`);
    if (event.message) extras.push(String(event.message));
    appendLog(`${event.type || 'unknown'}${extras.length ? `  ${extras.join(' ')}` : ''}`, event.type === 'error');
  }

  function ensureTurn(role) {
    if (emptyConversation) emptyConversation.remove();
    if (role === 'assistant') {
      if (liveAssistantTurn) return liveAssistantTurn;
      const row = document.createElement('div');
      row.className = 'turn turn-assistant turn-live';
      const label = document.createElement('div');
      label.className = 'turn-role';
      label.textContent = 'Assistant';
      const text = document.createElement('div');
      text.className = 'turn-text';
      row.append(label, text);
      conversation.appendChild(row);
      conversation.scrollTop = conversation.scrollHeight;
      liveAssistantTurn = { row, text, value: '' };
      return liveAssistantTurn;
    }
    if (role === 'user') {
      const row = document.createElement('div');
      row.className = 'turn turn-user';
      const label = document.createElement('div');
      label.className = 'turn-role';
      label.textContent = 'You';
      const text = document.createElement('div');
      text.className = 'turn-text';
      row.append(label, text);
      conversation.appendChild(row);
      conversation.scrollTop = conversation.scrollHeight;
      return { row, text, value: '' };
    }
    return null;
  }

  function addUserTurn(message) {
    const turn = ensureTurn('user');
    turn.value = message;
    turn.text.textContent = message;
    return turn;
  }

  function finishUserTranscript(text) {
    const transcript = String(text || '').trim();
    if (!transcript) return;
    const turn = pendingUserTurns.shift() || addUserTurn(transcript);
    turn.value = transcript;
    turn.text.textContent = transcript;
    conversation.scrollTop = conversation.scrollHeight;
  }

  function grabPreviewFrame() {
    if (!cameraStream || cameraPreview.videoWidth === 0) return lastCameraFrame;
    const canvas = document.createElement('canvas');
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;
    canvas.getContext('2d').drawImage(cameraPreview, 0, 0);
    lastCameraFrame = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    return lastCameraFrame;
  }

  function visibleAssistantText(text) {
    let visible = String(text || '').replaceAll(SILENT_TOKEN, '');
    for (let length = SILENT_TOKEN.length - 1; length > 0; length -= 1) {
      if (visible.endsWith(SILENT_TOKEN.slice(0, length))) {
        visible = visible.slice(0, -length);
        break;
      }
    }
    return visible.trim();
  }

  function addTranscript(delta) {
    if (!delta) return;
    assistantRawText += delta;
    const visible = visibleAssistantText(assistantRawText);
    if (!visible && !liveAssistantTurn) return;
    const turn = ensureTurn('assistant');
    turn.value = visible;
    turn.text.textContent = visible;
    conversation.scrollTop = conversation.scrollHeight;
  }

  function finishTranscript(finalText = '') {
    const visible = visibleAssistantText(finalText || assistantRawText);
    assistantRawText = '';
    if (!visible) {
      if (liveAssistantTurn) liveAssistantTurn.row.remove();
      liveAssistantTurn = null;
      return;
    }
    const current = liveAssistantTurn || ensureTurn('assistant');
    current.value = visible;
    current.text.textContent = visible;
    current.row.classList.remove('turn-live');
    liveAssistantTurn = null;
  }

  function bytesToBase64(bytes) {
    let binary = '';
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
  }

  function int16ToBase64(pcm) {
    return bytesToBase64(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength));
  }

  function base64ToBytes(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return bytes;
  }

  function resampleInt16(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return input;
    const ratio = sourceRate / targetRate;
    const output = new Int16Array(Math.floor(input.length / ratio));
    for (let index = 0; index < output.length; index += 1) {
      const start = Math.floor(index * ratio);
      const end = Math.max(start + 1, Math.min(input.length, Math.floor((index + 1) * ratio)));
      let sum = 0;
      for (let source = start; source < end; source += 1) sum += input[source];
      output[index] = sum / (end - start);
    }
    return output;
  }

  async function decodeAudioDelta(event) {
    const encoded = event.data || event.delta;
    if (!encoded) return null;
    const bytes = base64ToBytes(encoded);
    const format = String(event.format || 'wav').toLowerCase();
    if (format.includes('wav') || (bytes.length >= 4 && String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF')) {
      const decoded = await playbackContext.decodeAudioData(bytes.buffer.slice(0));
      const channel = decoded.getChannelData(0);
      const pcm = new Int16Array(channel.length);
      for (let index = 0; index < channel.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, channel[index]));
        pcm[index] = sample < 0 ? sample * 32768 : sample * 32767;
      }
      return { pcm, sourceRate: decoded.sampleRate };
    }
    return {
      pcm: new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2)),
      sourceRate: Number(event.sample_rate_hz || event.sample_rate || OUTPUT_RATE),
    };
  }

  function updateMeter(pcm) {
    let peak = 0;
    for (let index = 0; index < pcm.length; index += 8) peak = Math.max(peak, Math.abs(pcm[index]));
    meterFill.style.width = `${Math.min(100, (peak / 32768) * 150).toFixed(0)}%`;
  }

  function microphoneCaptureEnabled() {
    return running && !muted;
  }

  function sendJson(payload) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(payload));
    return true;
  }

  function mergePendingCapture() {
    if (pendingCapture.length === 0) return null;
    const length = pendingCapture.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Int16Array(length);
    let offset = 0;
    for (const chunk of pendingCapture) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    pendingCapture = [];
    return resampleInt16(merged, captureRate, INPUT_RATE);
  }

  function commitVoiceTurn() {
    const pcm = mergePendingCapture();
    if (!pcm || pcm.length === 0) {
      appendLog('push-to-talk: no audio buffered; voice turn not committed');
      return;
    }
    // Continuous video.frame streaming already feeds the session; still send one
    // fresh frame with the utterance so the voice turn has present-moment vision.
    const frameB64 = grabPreviewFrame();
    if (frameB64) sendCameraFrame(frameB64);
    else if (!lastCameraFrame) {
      appendLog('push-to-talk: enable Camera before speaking', true);
      return;
    }
    const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (let offset = 0; offset < bytes.length; offset += AUDIO_CHUNK_BYTES) {
      const slice = bytes.subarray(offset, offset + AUDIO_CHUNK_BYTES);
      sendJson({ type: 'audio.chunk', data: bytesToBase64(slice) });
    }
    sendJson({ type: 'audio.done' });
    const seconds = (pcm.length / INPUT_RATE);
    pendingUserTurns.push(addUserTurn(`🎤 Transcribing… (${seconds.toFixed(1)}s)`));
    if (assistantActive) {
      appendLog('push-to-talk: current turn busy; server may defer this turn');
    }
    setModel('Thinking');
    appendLog(`committed voice turn  ${bytes.length} pcm bytes`);
  }

  function sendCameraFrame(frameB64) {
    if (!frameB64) return;
    lastCameraFrame = frameB64;
    if (holdCameraForSpeech || responseHasAudio || activePlaybackRequestId) return;
    if (sendJson({ type: 'video.frame', data: frameB64 })) {
      framesSent += 1;
      if (framesSent === 1 || framesSent % 10 === 0) {
        appendLog(`video.frame sent  #${framesSent}`);
      }
    }
  }

  function feedPlayback(decoded) {
    if (!decoded || !decoded.pcm || decoded.pcm.length === 0 || !playbackNode) return;
    const pcm = resampleInt16(decoded.pcm, decoded.sourceRate, playbackRate);
    responseHasAudio = true;
    assistantActive = true;
    setPlayback('Buffering');
    playbackNode.port.postMessage({
      type: 'audio',
      pcm,
      responseId: activePlaybackRequestId || `turn-${turnsSeen}`,
      initialBufferMs: INITIAL_PLAYBACK_BUFFER_MS,
    }, [pcm.buffer]);
  }

  function requestPlaybackDrain() {
    if (!playbackNode) return;
    playbackNode.port.postMessage({
      type: 'drain',
      responseId: activePlaybackRequestId || `turn-${turnsSeen}`,
    });
  }

  function clearPlayback() {
    // Drop queued spoken audio. Never call this for <|silent|> turns.
    playbackChain = Promise.resolve();
    activePlaybackRequestId = null;
    holdCameraForSpeech = false;
    if (!playbackNode) return;
    playbackNode.port.postMessage({ type: 'clear' });
    responseHasAudio = false;
    setPlayback('Idle');
  }

  function bindPlaybackRequest(rid) {
    // Switch spoken request without transiently nulling activePlaybackRequestId
    // (that race dropped in-flight decode chunks for the new rid).
    if (activePlaybackRequestId === rid) return;
    playbackChain = Promise.resolve();
    if (playbackNode) playbackNode.port.postMessage({ type: 'clear' });
    responseHasAudio = false;
    activePlaybackRequestId = rid;
    holdCameraForSpeech = true;
    setPlayback('Buffering');
  }

  function enqueuePlaybackDelta(event) {
    const rid = eventRequestId(event) || `legacy-${turnsSeen}`;
    playbackChain = playbackChain
      .then(async () => {
        // Skip stale chunks after clearPlayback / request switch.
        if (activePlaybackRequestId !== rid) return;
        const decoded = await decodeAudioDelta(event);
        if (activePlaybackRequestId !== rid) return;
        feedPlayback(decoded);
      })
      .catch((error) => appendLog(`audio decode failed: ${error.message || error}`, true));
  }

  function eventRequestId(event) {
    const rid = event && event.request_id;
    return rid ? String(rid) : null;
  }

  function playbackDrained(message) {
    setPlayback('Idle');
    holdCameraForSpeech = false;
    activePlaybackRequestId = null;
    if (message.underrunMs > 0) appendLog(`playback underrun ${message.underrunMs} ms`);
    window.setTimeout(() => {
      assistantActive = false;
      responseHasAudio = false;
      if (running) setModel(muted ? 'Idle' : 'Listening');
    }, ECHO_GUARD_MS);
  }

  function handleEvent(event) {
    appendEventLog(event);
    switch (event.type) {
      case 'response.start':
        // Do not clearPlayback here: silent vision turns also emit response.start
        // and must not cut an in-progress spoken reply.
        turnsSeen += 1;
        assistantActive = true;
        responseHasAudio = false;
        assistantRawText = '';
        setModel('Thinking');
        break;
      case 'user.transcript.done':
        finishUserTranscript(event.text || '');
        break;
      case 'response.text.delta':
        setModel('Speaking');
        addTranscript(event.delta || '');
        break;
      case 'response.text.done': {
        const text = event.text || '';
        finishTranscript(text);
        if (text.includes(SILENT_TOKEN)) {
          holdCameraForSpeech = false;
          setModel(muted ? 'Idle' : 'Listening');
          assistantActive = false;
        } else if (!responseHasAudio) {
          holdCameraForSpeech = true;
          setModel('Speaking');
        }
        break;
      }
      case 'response.audio.delta': {
        // Bind playback to request_id. Silent vision turns also bump turnsSeen via
        // response.start; clearing on turnsSeen would wipe in-flight spoken audio
        // while dump (grouped by request_id) stays complete.
        const rid = eventRequestId(event) || `legacy-${turnsSeen}`;
        bindPlaybackRequest(rid);
        assistantActive = true;
        setModel('Speaking');
        enqueuePlaybackDelta(event);
        break;
      }
      case 'response.audio.done': {
        const rid = eventRequestId(event);
        // Ignore done from a non-active request (overlapping silent / stale turns).
        if (rid && activePlaybackRequestId && rid !== activePlaybackRequestId) break;
        // Drain only after in-flight ordered decodes finish.
        playbackChain = playbackChain
          .then(() => { requestPlaybackDrain(); })
          .catch(() => { requestPlaybackDrain(); });
        break;
      }
      case 'session.done':
        appendLog('session.done');
        if (running) stopSession({ terminal: false });
        break;
      case 'error':
        setConnection('Error', 'error');
        runtimeDetail.textContent = String(event.message || event.error || 'Server error');
        break;
      case 'pong':
        break;
      default:
        break;
    }
  }

  async function openPlayback() {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_RATE });
    playbackRate = playbackContext.sampleRate;
    await playbackContext.audioWorklet.addModule(staticAssetUrl('static/playback_worklet.js'));
    playbackNode = new AudioWorkletNode(playbackContext, 'fullduplex-pcm-playback');
    playbackNode.port.postMessage({ type: 'config', playbackSpeed: PLAYBACK_SPEED });
    playbackNode.port.onmessage = (message) => {
      if (message.data.type === 'playback-started') setPlayback('Playing');
      else if (message.data.type === 'playback-drained') playbackDrained(message.data);
      else if (message.data.type === 'playback-underrun') {
        runtimeDetail.textContent = `Playback underrun ${message.data.underrunMs || 0} ms`;
      }
    };
    playbackNode.connect(playbackContext.destination);
    await playbackContext.resume();
    appendLog(`playback speed ${PLAYBACK_SPEED}x (1.0 = dump-identical; dump unchanged)`);
  }

  function audioConstraints() {
    return {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      sampleRate: { ideal: INPUT_RATE },
    };
  }

  async function openCapture() {
    if (!mediaStream || mediaStream.getAudioTracks().every((track) => track.readyState === 'ended')) {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints() });
    }
    try {
      captureContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: INPUT_RATE });
    } catch (_error) {
      captureContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    captureRate = captureContext.sampleRate;
    await captureContext.audioWorklet.addModule(staticAssetUrl('static/pcm_worklet.js'));
    const source = captureContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(captureContext, 'fullduplex-pcm-capture');
    captureNode.port.onmessage = (message) => {
      const pcm = new Int16Array(message.data);
      updateMeter(pcm);
      if (microphoneCaptureEnabled()) pendingCapture.push(pcm);
    };
    const silentSink = captureContext.createGain();
    silentSink.gain.value = 0;
    source.connect(captureNode);
    captureNode.connect(silentSink).connect(captureContext.destination);
    await captureContext.resume();
  }

  function openSocket() {
    return new Promise((resolve, reject) => {
      const url = streamUrl();
      socket = new WebSocket(url);
      let settled = false;
      socket.onopen = () => {
        settled = true;
        const session = {
          type: 'session.config',
          model: config.model || 'aurateam/AURA',
          modalities: ['text', 'audio'],
          auto_trigger: true,
          auto_trigger_min_frames: 2,
          max_frames: 256,
          max_frames_per_round: 16,
          video_fps: Number(config.videoFps) > 0 ? Number(config.videoFps) : 2.0,
          stream_text_deltas: true,
          enable_frame_filter: false,
          aura_system_prompt: systemPromptInput ? systemPromptInput.value.trim() : '',
          tts_task_type: config.ttsTaskType || 'CustomVoice',
          tts_language: config.ttsLanguage || 'Chinese',
          tts_speaker: config.ttsSpeaker || 'Vivian',
          // Empty string is intentional (Native-aligned, no style instruct).
          // Do not use `||` — falsy "" would re-inject a long English instruct
          // and inflate Talker prompt_len by ~100 zero pads.
          tts_instruct: (typeof config.ttsInstruct === 'string') ? config.ttsInstruct : '',
        };
        socket.send(JSON.stringify(session));
        runtimeDetail.textContent = `${captureRate} Hz capture / ${playbackRate} Hz playback`;
        appendLog(`websocket open  ${url}`);
        appendLog('session.config sent');
        resolve();
      };
      socket.onmessage = (message) => {
        if (typeof message.data !== 'string') return;
        try {
          handleEvent(JSON.parse(message.data));
        } catch (error) {
          appendLog(`invalid server event: ${error.message || error}`, true);
        }
      };
      socket.onerror = () => {
        if (!settled) {
          settled = true;
          reject(new Error(`WebSocket connection failed: ${url}`));
        }
      };
      socket.onclose = (event) => {
        appendLog(`websocket closed  code=${event.code}`);
        if (running) {
          setConnection('Disconnected', 'error');
          stopSession({ terminal: false });
        }
      };
    });
  }

  function formatElapsed(seconds) {
    const minutes = Math.floor(seconds / 60);
    return `${String(minutes).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  }

  function startClock() {
    startedAt = Date.now();
    clockTimer = window.setInterval(() => {
      sessionTimer.textContent = formatElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
  }

  async function startSession() {
    if (running) return;
    callButton.disabled = true;
    setConnection('Connecting', 'connecting');
    runtimeDetail.textContent = 'Requesting microphone access';
    try {
      await openPlayback();
      await openCapture();
      await startCamera({ force: !isCameraLive() });
      await openSocket();
      running = true;
      muted = true;
      assistantActive = false;
      framesSent = 0;
      turnsSeen = 0;
      activePlaybackRequestId = null;
      startClock();
      callButton.textContent = 'End session';
      callButton.classList.add('is-active');
      muteButton.disabled = false;
      cameraButton.disabled = false;
      setConnection('Connected', 'online');
      setModel('Ready');
      appendLog('session started (hold to talk; release commits the voice turn)');
    } catch (error) {
      appendLog(`start failed: ${error.message || error}`, true);
      setConnection('Error', 'error');
      runtimeDetail.textContent = String(error.message || error);
      await stopSession({ terminal: false });
    } finally {
      callButton.disabled = false;
    }
  }

  function setCameraPlaceholder(message) {
    if (!cameraPlaceholder) return;
    cameraPlaceholder.hidden = false;
    cameraPlaceholder.textContent = message;
  }

  function liveVideoTracks(stream) {
    if (!stream) return [];
    return stream.getVideoTracks().filter((track) => track.readyState === 'live');
  }

  function isCameraLive() {
    return liveVideoTracks(cameraStream).length > 0
      && cameraPreview.srcObject === cameraStream
      && !cameraPreview.paused
      && cameraPreview.videoWidth > 0;
  }

  function releaseCameraStream() {
    if (cameraTimer !== null) clearInterval(cameraTimer);
    cameraTimer = null;
    if (cameraStream) {
      for (const track of cameraStream.getTracks()) track.stop();
    }
    cameraStream = null;
    cameraPreview.srcObject = null;
    cameraPreview.style.display = 'none';
  }

  async function startCamera({ force = false } = {}) {
    if (!force && isCameraLive() && cameraTimer !== null) return;
    if (force || !liveVideoTracks(cameraStream).length) {
      releaseCameraStream();
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: false,
        });
      } catch (error) {
        const name = error && error.name ? error.name : 'Error';
        const detail = error && error.message ? error.message : String(error);
        let hint = `${name}: ${detail}`;
        if (name === 'NotReadableError' || /Could not start video source/i.test(detail)) {
          hint += ' — close other tabs/apps using the camera, then click Camera again.';
        } else if (name === 'NotAllowedError') {
          hint += ' — allow camera for this site, then click Camera again.';
        }
        setCameraPlaceholder(hint);
        cameraButton.textContent = 'Retry camera';
        cameraButton.classList.remove('is-active');
        cameraButton.disabled = false;
        throw error;
      }
    }
    cameraPreview.srcObject = cameraStream;
    cameraPreview.style.display = 'block';
    try {
      await cameraPreview.play();
    } catch (error) {
      setCameraPlaceholder(`Camera preview blocked: ${error.message || error}`);
      throw error;
    }
    // Wait briefly for the first decoded frame so placeholder is not left up.
    for (let attempt = 0; attempt < 20 && cameraPreview.videoWidth === 0; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 50));
    }
    if (cameraPreview.videoWidth === 0) {
      setCameraPlaceholder('Camera opened but no frames yet. Click Retry camera.');
      cameraButton.textContent = 'Retry camera';
      cameraButton.classList.remove('is-active');
      cameraButton.disabled = false;
      throw new Error('camera preview has no frames');
    }
    if (cameraPlaceholder) cameraPlaceholder.hidden = true;
    if (cameraTimer !== null) clearInterval(cameraTimer);
    const intervalMs = cameraIntervalMs();
    cameraTimer = window.setInterval(() => {
      // Product behavior: stream live frames while the session is running so
      // AURA can auto-trigger silent/proactive vision turns.
      const frameB64 = grabPreviewFrame();
      if (!frameB64) return;
      lastCameraFrame = frameB64;
      if (running) sendCameraFrame(frameB64);
    }, intervalMs);
    cameraButton.textContent = 'Camera off';
    cameraButton.classList.add('is-active');
    cameraButton.disabled = false;
    appendLog(`camera streaming on (${(1000 / intervalMs).toFixed(1)} fps while session active)`);
  }

  function stopCamera() {
    releaseCameraStream();
    setCameraPlaceholder('Camera off. Click Camera to reopen.');
    cameraButton.textContent = 'Camera';
    cameraButton.classList.remove('is-active');
  }

  cameraButton.addEventListener('click', () => {
    if (isCameraLive() || cameraTimer !== null) {
      stopCamera();
      appendLog('camera off');
      return;
    }
    startCamera({ force: true }).catch((error) => {
      appendLog(`camera failed: ${error.message || error}`, true);
    });
  });

  async function stopSession({ terminal = true } = {}) {
    const wasRunning = running;
    running = false;
    talking = false;
    assistantActive = false;
    pendingCapture = [];
    if (clockTimer !== null) clearInterval(clockTimer);
    clockTimer = null;
    if (socket) {
      const closingSocket = socket;
      socket = null;
      closingSocket.onclose = null;
      if (terminal && wasRunning && closingSocket.readyState === WebSocket.OPEN) {
        closingSocket.send(JSON.stringify({ type: 'video.done' }));
      }
      closingSocket.close(1000, 'client stop');
    }
    playbackChain = Promise.resolve();
    if (playbackNode) playbackNode.port.postMessage({ type: 'clear' });
    if (mediaStream) {
      for (const track of mediaStream.getTracks()) track.stop();
    }
    mediaStream = null;
    stopCamera();
    cameraButton.disabled = false;
    cameraButton.textContent = 'Retry camera';
    if (captureContext) await captureContext.close().catch(() => {});
    if (playbackContext) await playbackContext.close().catch(() => {});
    captureContext = null;
    captureNode = null;
    playbackContext = null;
    playbackNode = null;
    responseHasAudio = false;
    pendingUserTurns = [];
    framesSent = 0;
    turnsSeen = 0;
    activePlaybackRequestId = null;
    lastCameraFrame = null;
    meterFill.style.width = '0%';
    sessionTimer.textContent = '00:00';
    callButton.textContent = 'Start session';
    callButton.classList.remove('is-active');
    muteButton.textContent = 'Hold to talk';
    muteButton.classList.remove('is-active');
    muteButton.setAttribute('aria-pressed', 'false');
    muteButton.disabled = true;
    setConnection('Offline', 'offline');
    setModel('Idle');
    setPlayback('Idle');
    if (!runtimeDetail.textContent.startsWith('start failed')
        && !runtimeDetail.textContent.startsWith('Playback underrun')) {
      runtimeDetail.textContent = 'No active connection';
    }
  }

  function startPushToTalk() {
    if (!running || !muted) return;
    muted = false;
    talking = true;
    pendingCapture = [];
    clearPlayback();
    muteButton.textContent = 'Release to send';
    muteButton.classList.add('is-active');
    muteButton.setAttribute('aria-pressed', 'true');
    setModel('Recording');
  }

  function finishPushToTalk() {
    if (!running || muted) return;
    muted = true;
    talking = false;
    muteButton.textContent = 'Hold to talk';
    muteButton.classList.remove('is-active');
    muteButton.setAttribute('aria-pressed', 'false');
    commitVoiceTurn();
    setModel('Thinking');
  }

  callButton.addEventListener('click', () => {
    if (running) stopSession();
    else startSession();
  });
  muteButton.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    muteButton.setPointerCapture(event.pointerId);
    startPushToTalk();
  });
  muteButton.addEventListener('pointerup', finishPushToTalk);
  muteButton.addEventListener('pointercancel', finishPushToTalk);
  muteButton.addEventListener('keydown', (event) => {
    if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
      event.preventDefault();
      startPushToTalk();
    }
  });
  muteButton.addEventListener('keyup', (event) => {
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault();
      finishPushToTalk();
    }
  });
  clearLogButton.addEventListener('click', () => {
    eventLog.textContent = '';
    logCount = 0;
    eventCount.textContent = '0 events';
  });
  async function requestMediaAccess() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraPlaceholder('Camera/microphone APIs unavailable in this browser.');
      appendLog('camera/microphone access is unavailable in this browser', true);
      return;
    }
    runtimeDetail.textContent = 'Requesting camera and microphone access';
    setCameraPlaceholder('Requesting camera and microphone access…');
    try {
      // Prefer separate requests so a busy camera still allows mic, and vice versa.
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints() });
      } catch (error) {
        appendLog(`microphone access failed: ${error.message || error}`, true);
      }
      await startCamera({ force: true });
      cameraButton.disabled = false;
      runtimeDetail.textContent = mediaStream
        ? 'Camera and microphone ready'
        : 'Camera ready (microphone still needed for talk)';
      appendLog('camera access granted');
    } catch (error) {
      runtimeDetail.textContent = 'Camera/microphone permission needed';
      appendLog(`media access failed: ${error.message || error}`, true);
      cameraButton.disabled = false;
      cameraButton.textContent = 'Retry camera';
    }
  }

  requestMediaAccess();
  window.addEventListener('beforeunload', () => { stopSession({ terminal: false }); });
})();
