class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.started = false;
    this.activeResponseId = null;
    this.initialBufferFrames = Math.round(sampleRate * 0.2);
    this.bufferWaitFrames = this.initialBufferFrames;
    // Soften codec first-chunk pop (initial_codec_chunk_frames=1); 5ms was too short.
    this.fadeFrames = Math.max(1, Math.round(sampleRate * 0.07));
    this.fadeInFrames = 0;
    this.lastUnderrunReportFrames = 0;
    // Client-only speedup (dump/TTS unchanged). Default 1.0 = dump-identical.
    this.playbackSpeed = 1.0;
    this.frameSize = Math.max(256, Math.round(sampleRate * 0.04));
    this.hopOut = Math.max(64, Math.floor(this.frameSize / 2));
    this.search = Math.max(32, Math.round(sampleRate * 0.008));
    this.window = new Float32Array(this.frameSize);
    for (let i = 0; i < this.frameSize; i += 1) {
      this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.frameSize - 1));
    }
    this.input = new Float32Array(0);
    this.inputStart = 0;
    this.analysisPos = 0;
    this.ola = new Float32Array(this.frameSize * 3);
    this.olaLen = 0;
    this.prevTail = new Float32Array(this.hopOut);
    this.hasPrev = false;
    this.flushedTail = false;
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    if (message.type === 'config' && Number.isFinite(message.playbackSpeed)) {
      this.playbackSpeed = Math.max(0.5, Math.min(2.5, Number(message.playbackSpeed)));
      return;
    }
    if (message.type === 'audio' && message.pcm) {
      const wasEmpty = this.bufferedFrames() === 0;
      if (!this.started && !this.activeResponseId) {
        this.activeResponseId = message.responseId || null;
      }
      if (!this.started && Number.isFinite(message.initialBufferMs)) {
        this.initialBufferFrames = Math.max(0, Math.round((sampleRate * message.initialBufferMs) / 1000));
      }
      this.pushPcm(message.pcm);
      if (!this.started && wasEmpty) {
        this.bufferWaitFrames = this.initialBufferFrames;
      }
    } else if (message.type === 'drain') {
      this.drain = { responseId: message.responseId || null };
      if (!this.started && this.bufferedFrames() > 0) {
        this.bufferWaitFrames = 0;
        this.startPlayback();
      }
      this.notifyIfDrained();
    } else if (message.type === 'clear') {
      this.resetBuffers();
      this.playedFrames = 0;
      this.underrunFrames = 0;
      this.drain = null;
      this.started = false;
      this.activeResponseId = null;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.fadeInFrames = 0;
      this.lastUnderrunReportFrames = 0;
    }
  }

  resetBuffers() {
    this.input = new Float32Array(0);
    this.inputStart = 0;
    this.analysisPos = 0;
    this.ola.fill(0);
    this.olaLen = 0;
    this.prevTail.fill(0);
    this.hasPrev = false;
    this.flushedTail = false;
  }

  pushPcm(pcm) {
    const n = pcm.length;
    const next = new Float32Array(this.input.length - this.inputStart + n);
    next.set(this.input.subarray(this.inputStart));
    const base = this.input.length - this.inputStart;
    for (let i = 0; i < n; i += 1) {
      next[base + i] = pcm[i] / 32768;
    }
    this.input = next;
    this.inputStart = 0;
    this.flushedTail = false;
  }

  bufferedFrames() {
    return this.input.length - this.inputStart;
  }

  inputEnd() {
    return this.inputStart + this.input.length;
  }

  sampleAt(absIndex) {
    const i = absIndex - this.inputStart;
    if (i < 0 || i >= this.input.length) return 0;
    return this.input[i];
  }

  compactInput() {
    const keepFrom = Math.max(0, Math.floor(this.analysisPos) - this.search - this.frameSize);
    if (keepFrom <= this.inputStart) return;
    const drop = keepFrom - this.inputStart;
    if (drop <= 0 || drop >= this.input.length) return;
    this.input = this.input.subarray(drop);
    this.inputStart = keepFrom;
  }

  startPlayback() {
    if (this.started) return;
    this.started = true;
    this.fadeInFrames = this.fadeFrames;
    if (this.playedFrames === 0) {
      this.port.postMessage({ type: 'playback-started', responseId: this.activeResponseId });
    }
  }

  notifyIfDrained() {
    const remainingInput = this.inputEnd() - this.analysisPos;
    if (!this.drain || this.olaLen > 0 || remainingInput > 8) return;
    this.port.postMessage({
      type: 'playback-drained',
      responseId: this.drain.responseId,
      playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
      underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
    });
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.started = false;
    this.activeResponseId = null;
    this.bufferWaitFrames = this.initialBufferFrames;
    this.fadeInFrames = 0;
    this.lastUnderrunReportFrames = 0;
    this.resetBuffers();
  }

  reportUnderrun() {
    if (this.underrunFrames - this.lastUnderrunReportFrames < sampleRate * 0.1) return;
    this.lastUnderrunReportFrames = this.underrunFrames;
    this.port.postMessage({
      type: 'playback-underrun',
      responseId: this.activeResponseId,
      underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
    });
  }

  bestDelta(ideal) {
    if (!this.hasPrev) return 0;
    const overlap = this.hopOut;
    let best = 0;
    let bestCorr = -Infinity;
    for (let d = -this.search; d <= this.search; d += 1) {
      let corr = 0;
      let e0 = 0;
      let e1 = 0;
      const base = ideal + d;
      for (let i = 0; i < overlap; i += 1) {
        const a = this.prevTail[i];
        const b = this.sampleAt(base + i);
        corr += a * b;
        e0 += a * a;
        e1 += b * b;
      }
      const ncorr = corr / (Math.sqrt(e0 * e1) + 1e-8);
      if (ncorr > bestCorr) {
        bestCorr = ncorr;
        best = d;
      }
    }
    return best;
  }

  canStretchFrame() {
    const need = Math.ceil(this.analysisPos + this.search + this.frameSize);
    if (need <= this.inputEnd()) return true;
    if (this.drain && this.analysisPos + 16 < this.inputEnd()) return true;
    return false;
  }

  ensureOlaCapacity(needLen) {
    if (needLen <= this.ola.length) return;
    const grown = new Float32Array(Math.max(needLen, this.ola.length * 2));
    grown.set(this.ola.subarray(0, this.olaLen + this.frameSize));
    this.ola = grown;
  }

  stretchOneFrame() {
    const speed = Math.max(0.5, this.playbackSpeed);
    const have = this.inputEnd();
    let ideal = this.analysisPos;
    if (ideal + this.frameSize > have) {
      ideal = Math.max(this.inputStart, have - this.frameSize);
    }
    const delta = this.bestDelta(ideal);
    const pos = Math.max(this.inputStart, Math.min(ideal + delta, Math.max(this.inputStart, have - this.frameSize)));

    this.ensureOlaCapacity(this.olaLen + this.frameSize);
    for (let i = 0; i < this.frameSize; i += 1) {
      this.ola[this.olaLen + i] += this.sampleAt(pos + i) * this.window[i];
    }
    // Next grain's start should match this grain's mid (overlap region).
    for (let i = 0; i < this.hopOut; i += 1) {
      this.prevTail[i] = this.sampleAt(pos + this.hopOut + i);
    }
    this.hasPrev = true;
    this.olaLen += this.hopOut;
    this.analysisPos = ideal + this.hopOut * speed;
    this.compactInput();
  }

  flushOlaTail() {
    if (this.flushedTail) return;
    this.flushedTail = true;
    // Release remaining overlap region after last grain.
    const tail = this.frameSize - this.hopOut;
    this.ensureOlaCapacity(this.olaLen + tail);
    this.olaLen += tail;
    this.analysisPos = this.inputEnd();
  }

  popSample() {
    // Bypass stretch at ~1x: direct read, no pitch/time change.
    if (Math.abs(this.playbackSpeed - 1) < 0.02) {
      if (this.analysisPos >= this.inputEnd()) return null;
      const s = this.sampleAt(this.analysisPos);
      this.analysisPos += 1;
      return s;
    }
    while (this.olaLen < 1 && this.canStretchFrame()) {
      this.stretchOneFrame();
    }
    if (this.olaLen < 1 && this.drain && this.hasPrev) {
      this.flushOlaTail();
    }
    if (this.olaLen < 1) return null;
    const s = this.ola[0];
    this.ola.copyWithin(0, 1, this.olaLen + this.frameSize);
    this.olaLen -= 1;
    return s;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (!this.started) {
      if (this.bufferedFrames() > 0 && this.bufferWaitFrames > 0) {
        this.bufferWaitFrames = Math.max(0, this.bufferWaitFrames - output.length);
        return true;
      }
      if (this.bufferedFrames() > 0) {
        this.startPlayback();
      }
    }
    if (!this.started) {
      this.notifyIfDrained();
      return true;
    }
    let target = 0;
    while (target < output.length) {
      const sample = this.popSample();
      if (sample === null) break;
      let out = sample;
      if (this.fadeInFrames > 0) {
        const elapsed = this.fadeFrames - this.fadeInFrames;
        out *= elapsed / this.fadeFrames;
        this.fadeInFrames -= 1;
      }
      output[target] = out;
      target += 1;
      this.playedFrames += 1;
    }
    if (target < output.length && !this.drain) {
      this.underrunFrames += output.length - target;
      this.reportUnderrun();
    }
    this.notifyIfDrained();
    return true;
  }
}

registerProcessor('fullduplex-pcm-playback', FullDuplexPcmPlayback);
