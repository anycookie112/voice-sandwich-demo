const STAGE_ICONS = {
  STT: "🎤",
  Agent: "🤖",
  Chunker: "💬",
  TTS: "🔊",
  Sentence: "📝",
};

class PipelineVisualizer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.stages = new Map();
    this.stageOrder = [];
    this.recentEvents = [];
    this.ws = null;
    this.startTime = Date.now();
    this.animationFrame = null;
    // Decay timers for different states
    this.INPUT_DECAY_MS = 300;
    this.PROCESSING_TIMEOUT_MS = 10000; // Long timeout for processing
    this.OUTPUT_DECAY_MS = 400;
  }

  connectedCallback() {
    this.render();
    this.connect();
    this.startTimer();
  }

  disconnectedCallback() {
    if (this.ws) {
      this.ws.close();
    }
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/pipeline`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.setConnectionStatus("connected");
    };

    this.ws.onclose = () => {
      this.setConnectionStatus("disconnected");
      // Try to reconnect after 2 seconds
      setTimeout(() => this.connect(), 2000);
    };

    this.ws.onerror = () => {
      this.setConnectionStatus("error");
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleEvent(data);
      } catch (e) {
        console.error("Failed to parse pipeline event:", e);
      }
    };
  }

  handleEvent(event) {
    switch (event.type) {
      case "stage_registered":
        this.registerStage(event);
        break;
      case "turn_start":
        this.handleTurnStart(event);
        break;
      case "stage_input":
        this.handleStageInput(event);
        break;
      case "stage_processing":
        this.handleStageProcessing(event);
        break;
      case "first_chunk":
        this.handleFirstChunk(event);
        break;
      case "chunk":
        this.handleChunk(event);
        break;
      case "stage_complete":
        this.handleStageComplete(event);
        break;
      case "pipeline_summary":
        this.handleSummary(event);
        break;
    }
  }

  registerStage(event) {
    if (this.stages.has(event.stageName)) return;

    const stage = {
      name: event.stageName,
      shortName: event.shortName,
      color: event.color,
      // Activity states
      isReceivingInput: false,
      isProcessing: false,
      isOutputting: false,
      // Timestamps for decay
      lastInputAt: 0,
      lastProcessingAt: 0,
      lastOutputAt: 0,
      // Metrics
      currentTurn: 0,
      chunksProcessed: 0,
      inputChunks: 0,
      totalBytes: 0,
      ttfc: null,
      lastChunkPreview: "",
      lastInputPreview: "",
    };

    this.stages.set(event.stageName, stage);
    this.stageOrder.push(event.stageName);
    this.renderStages();
  }

  handleTurnStart(event) {
    const stage = this.stages.get(event.stageName);
    if (!stage) return;

    stage.currentTurn = event.turnNumber;
    this.addEvent(stage.shortName, `Turn #${event.turnNumber} started`, stage.color);
    this.renderStages();
    this.renderEvents();
  }

  handleStageInput(event) {
    const stage = this.stages.get(event.stageName);
    if (!stage) return;

    stage.isReceivingInput = true;
    stage.lastInputAt = Date.now();
    stage.inputChunks++;
    if (event.chunkPreview) {
      stage.lastInputPreview = event.chunkPreview;
    }

    this.addEvent(
      stage.shortName,
      `← Input${event.chunkPreview ? `: "${this.truncate(event.chunkPreview, 25)}"` : ""}`,
      stage.color
    );
    this.renderStages();
    this.renderEvents();
    this.pulseStageInput(event.stageName);
  }

  handleStageProcessing(event) {
    const stage = this.stages.get(event.stageName);
    if (!stage) return;

    stage.isProcessing = true;
    stage.lastProcessingAt = Date.now();
    stage.currentTurn = event.turnNumber;

    this.addEvent(stage.shortName, "⏳ Processing...", stage.color);
    this.renderStages();
    this.renderEvents();
  }

  handleFirstChunk(event) {
    const stage = this.stages.get(event.stageName);
    if (!stage) return;

    // Switch from processing to outputting
    stage.isProcessing = false;
    stage.isOutputting = true;
    stage.lastOutputAt = Date.now();
    stage.ttfc = event.ttfc;
    if (event.chunkPreview) {
      stage.lastChunkPreview = event.chunkPreview;
    }

    this.addEvent(
      stage.shortName,
      `→ First output (${event.ttfc}ms)${event.chunkPreview ? `: "${this.truncate(event.chunkPreview, 20)}"` : ""}`,
      stage.color
    );
    this.renderStages();
    this.renderEvents();
    this.pulseStageOutput(event.stageName);
  }

  handleChunk(event) {
    const stage = this.stages.get(event.stageName);
    if (!stage) return;

    stage.isProcessing = false;
    stage.isOutputting = true;
    stage.lastOutputAt = Date.now();
    stage.chunksProcessed = event.metrics.chunksProcessed;
    stage.totalBytes = event.metrics.totalBytes;
    stage.currentTurn = event.metrics.currentTurn;

    if (event.chunkPreview) {
      stage.lastChunkPreview = event.chunkPreview;
    }

    this.renderStages();
    this.pulseStageOutput(event.stageName);
  }

  handleStageComplete(event) {
    const stage = this.stages.get(event.stageName);
    if (!stage) return;

    stage.isReceivingInput = false;
    stage.isProcessing = false;
    stage.isOutputting = false;
    this.addEvent(stage.shortName, "✓ Completed", stage.color);
    this.renderStages();
    this.renderEvents();
  }

  handleSummary(event) {
    this.addEvent("Pipeline", "Session ended", "#888");
    this.renderEvents();
  }

  addEvent(stageName, message, color) {
    this.recentEvents.unshift({
      time: Date.now(),
      stageName,
      message,
      color,
    });
    if (this.recentEvents.length > 8) {
      this.recentEvents.pop();
    }
  }

  pulseStageInput(stageName) {
    const stageEl = this.shadowRoot.querySelector(`[data-stage="${stageName}"]`);
    if (stageEl) {
      const inputIndicator = stageEl.querySelector(".input-indicator");
      if (inputIndicator) {
        inputIndicator.classList.remove("pulse");
        void inputIndicator.offsetWidth;
        inputIndicator.classList.add("pulse");
      }
    }
  }

  pulseStageOutput(stageName) {
    const stageEl = this.shadowRoot.querySelector(`[data-stage="${stageName}"]`);
    if (stageEl) {
      stageEl.classList.remove("pulse-output");
      void stageEl.offsetWidth;
      stageEl.classList.add("pulse-output");
      
      const outputIndicator = stageEl.querySelector(".output-indicator");
      if (outputIndicator) {
        outputIndicator.classList.remove("pulse");
        void outputIndicator.offsetWidth;
        outputIndicator.classList.add("pulse");
      }
    }
  }

  startTimer() {
    const updateTimer = () => {
      const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
      const timerEl = this.shadowRoot.querySelector(".timer");
      if (timerEl) {
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        timerEl.textContent = `${mins}:${secs.toString().padStart(2, "0")}`;
      }

      // Update activity states with decay
      const now = Date.now();
      let needsRender = false;
      
      for (const stage of this.stages.values()) {
        // Decay input state
        if (stage.isReceivingInput && now - stage.lastInputAt > this.INPUT_DECAY_MS) {
          stage.isReceivingInput = false;
          needsRender = true;
        }
        
        // Decay output state
        if (stage.isOutputting && now - stage.lastOutputAt > this.OUTPUT_DECAY_MS) {
          stage.isOutputting = false;
          needsRender = true;
        }
        
        // Processing state has longer timeout (agent thinking can take time)
        if (stage.isProcessing && now - stage.lastProcessingAt > this.PROCESSING_TIMEOUT_MS) {
          stage.isProcessing = false;
          needsRender = true;
        }
      }
      
      if (needsRender) {
        this.renderStages();
      }

      this.animationFrame = requestAnimationFrame(updateTimer);
    };
    updateTimer();
  }

  setConnectionStatus(status) {
    const statusEl = this.shadowRoot.querySelector(".connection-status");
    if (statusEl) {
      statusEl.className = `connection-status ${status}`;
      statusEl.textContent = status === "connected" ? "●" : status === "error" ? "✕" : "○";
    }
  }

  truncate(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.slice(0, maxLength) + "…";
  }

  formatBytes(bytes) {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  }

  formatAge(timestamp) {
    const age = Math.floor((Date.now() - timestamp) / 1000);
    if (age === 0) return "now";
    if (age < 60) return `${age}s`;
    return `${Math.floor(age / 60)}m`;
  }

  render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
          font-size: 13px;
          color: #e4e4e7;
          background: linear-gradient(145deg, #18181b 0%, #09090b 100%);
          border-radius: 12px;
          padding: 20px;
          box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 12px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .title {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 14px;
          font-weight: 600;
          color: #fafafa;
        }

        .title-icon {
          font-size: 18px;
        }

        .header-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .timer {
          color: #a1a1aa;
          font-variant-numeric: tabular-nums;
        }

        .connection-status {
          font-size: 10px;
          transition: color 0.3s ease;
        }

        .connection-status.connected { color: #22c55e; }
        .connection-status.disconnected { color: #a1a1aa; }
        .connection-status.error { color: #ef4444; }

        .pipeline-flow {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          margin-bottom: 24px;
          padding: 16px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
        }

        .stage-node {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 8px;
          transition: all 0.2s ease;
          min-width: 80px;
          position: relative;
        }

        .stage-node.receiving-input {
          border-left: 2px solid var(--stage-color);
          box-shadow: -4px 0 12px rgba(var(--stage-color-rgb), 0.4);
        }

        .stage-node.processing {
          background: rgba(var(--stage-color-rgb), 0.1);
          border-color: var(--stage-color);
          animation: processing-pulse 1.5s ease-in-out infinite;
        }

        .stage-node.outputting {
          border-right: 2px solid var(--stage-color);
          box-shadow: 4px 0 12px rgba(var(--stage-color-rgb), 0.4);
          background: rgba(var(--stage-color-rgb), 0.08);
        }

        .stage-node.pulse-output {
          animation: pulse-output 0.3s ease-out;
        }

        @keyframes processing-pulse {
          0%, 100% { 
            box-shadow: 0 0 8px rgba(var(--stage-color-rgb), 0.3);
          }
          50% { 
            box-shadow: 0 0 20px rgba(var(--stage-color-rgb), 0.6);
          }
        }

        @keyframes pulse-output {
          0% { transform: scale(1); }
          50% { transform: scale(1.03); }
          100% { transform: scale(1); }
        }

        .stage-icon {
          font-size: 20px;
          line-height: 1;
          transition: transform 0.2s ease;
        }

        .stage-node.processing .stage-icon {
          animation: icon-spin 2s linear infinite;
        }

        @keyframes icon-spin {
          0% { transform: rotate(0deg); }
          25% { transform: rotate(5deg); }
          75% { transform: rotate(-5deg); }
          100% { transform: rotate(0deg); }
        }

        .stage-name {
          font-size: 11px;
          color: #a1a1aa;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        /* Flow indicators container */
        .flow-indicators {
          display: flex;
          justify-content: space-between;
          width: 100%;
          margin-top: 4px;
        }

        .input-indicator, .output-indicator {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.15);
          transition: all 0.2s ease;
        }

        .input-indicator.active {
          background: var(--stage-color);
          box-shadow: 0 0 6px var(--stage-color);
          animation: indicator-pulse 0.3s ease-out;
        }

        .output-indicator.active {
          background: var(--stage-color);
          box-shadow: 0 0 6px var(--stage-color);
        }

        .input-indicator.pulse, .output-indicator.pulse {
          animation: indicator-pulse 0.3s ease-out;
        }

        @keyframes indicator-pulse {
          0% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.8); opacity: 0.8; }
          100% { transform: scale(1); opacity: 1; }
        }

        /* Processing indicator */
        .processing-indicator {
          display: none;
          align-items: center;
          gap: 3px;
          margin-top: 4px;
        }

        .stage-node.processing .processing-indicator {
          display: flex;
        }

        .processing-dot {
          width: 4px;
          height: 4px;
          border-radius: 50%;
          background: var(--stage-color);
          animation: processing-dots 1.2s ease-in-out infinite;
        }

        .processing-dot:nth-child(2) { animation-delay: 0.2s; }
        .processing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes processing-dots {
          0%, 80%, 100% { 
            transform: scale(0.6);
            opacity: 0.4;
          }
          40% { 
            transform: scale(1);
            opacity: 1;
          }
        }

        /* Connecting arrows */
        .arrow {
          color: #3f3f46;
          font-size: 18px;
          transition: color 0.2s ease;
        }

        .arrow.active {
          color: var(--arrow-color, #52525b);
        }

        .arrow.flowing {
          animation: arrow-flow 0.5s ease-out;
        }

        @keyframes arrow-flow {
          0% { transform: translateX(0); opacity: 0.5; }
          50% { transform: translateX(3px); opacity: 1; }
          100% { transform: translateX(0); opacity: 0.7; }
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 12px;
          margin-bottom: 20px;
        }

        .metric-card {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.06);
          border-radius: 8px;
          padding: 12px;
          transition: all 0.2s ease;
        }

        .metric-card.processing {
          border-color: var(--stage-color);
          box-shadow: 0 0 12px rgba(var(--stage-color-rgb), 0.2);
          animation: card-processing 1.5s ease-in-out infinite;
        }

        .metric-card.receiving {
          border-left: 3px solid var(--stage-color);
        }

        .metric-card.streaming {
          border-right: 3px solid var(--stage-color);
          background: rgba(var(--stage-color-rgb), 0.05);
        }

        @keyframes card-processing {
          0%, 100% { box-shadow: 0 0 8px rgba(var(--stage-color-rgb), 0.15); }
          50% { box-shadow: 0 0 16px rgba(var(--stage-color-rgb), 0.3); }
        }

        .metric-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 10px;
        }

        .metric-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.3);
          transition: all 0.2s ease;
        }

        .metric-dot.processing {
          background: var(--stage-color);
          animation: dot-pulse 1s ease-in-out infinite;
        }

        .metric-dot.receiving,
        .metric-dot.streaming {
          background: var(--stage-color);
          box-shadow: 0 0 6px var(--stage-color);
        }

        @keyframes dot-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(0.8); }
        }

        .metric-title {
          font-size: 12px;
          font-weight: 500;
          color: #fafafa;
          flex: 1;
        }

        .state-badge {
          font-size: 10px;
          padding: 2px 6px;
          border-radius: 4px;
          font-weight: 500;
          animation: badge-appear 0.2s ease-out;
        }

        .state-badge.processing {
          background: rgba(var(--stage-color-rgb), 0.2);
          color: var(--stage-color);
        }

        .state-badge.receiving {
          background: rgba(var(--stage-color-rgb), 0.15);
          color: var(--stage-color);
        }

        .state-badge.streaming {
          background: rgba(var(--stage-color-rgb), 0.25);
          color: var(--stage-color);
        }

        @keyframes badge-appear {
          from { opacity: 0; transform: scale(0.8); }
          to { opacity: 1; transform: scale(1); }
        }

        .metric-values {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }

        .metric-item {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .metric-label {
          font-size: 10px;
          color: #71717a;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .metric-value {
          font-size: 14px;
          font-weight: 500;
          color: #e4e4e7;
          font-variant-numeric: tabular-nums;
        }

        .metric-previews {
          margin-top: 8px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .metric-preview {
          padding: 6px 8px;
          background: rgba(0, 0, 0, 0.3);
          border-radius: 4px;
          font-size: 11px;
          color: #a1a1aa;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .metric-preview.input {
          border-left: 2px solid rgba(var(--stage-color-rgb), 0.5);
        }

        .metric-preview.output {
          border-right: 2px solid rgba(var(--stage-color-rgb), 0.5);
          text-align: right;
        }

        .events-section {
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          padding: 12px;
        }

        .events-title {
          font-size: 11px;
          color: #71717a;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 10px;
        }

        .event-list {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .event-item {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 12px;
          opacity: 0.9;
          animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateX(-10px); }
          to { opacity: 0.9; transform: translateX(0); }
        }

        .event-time {
          color: #52525b;
          min-width: 32px;
          text-align: right;
          font-variant-numeric: tabular-nums;
        }

        .event-stage {
          font-weight: 500;
          min-width: 60px;
        }

        .event-message {
          color: #a1a1aa;
          flex: 1;
        }

        /* Event type indicators */
        .event-item[data-type="input"] .event-stage::before {
          content: "← ";
          opacity: 0.6;
        }

        .event-item[data-type="output"] .event-stage::after {
          content: " →";
          opacity: 0.6;
        }

        .event-item[data-type="processing"] {
          opacity: 0.85;
        }

        .empty-state {
          text-align: center;
          padding: 40px 20px;
          color: #52525b;
        }

        .empty-state-icon {
          font-size: 32px;
          margin-bottom: 12px;
          opacity: 0.5;
        }
      </style>

      <div class="header">
        <div class="title">
          <span>Pipeline Monitor</span>
        </div>
        <div class="header-right">
          <span class="timer">0:00</span>
          <span class="connection-status disconnected">○</span>
        </div>
      </div>

      <div class="pipeline-flow" id="pipeline-flow">
        <div class="empty-state">
          <div class="empty-state-icon">⏳</div>
          <div>Waiting for pipeline stages...</div>
        </div>
      </div>

      <div class="metrics-grid" id="metrics-grid"></div>

      <div class="events-section">
        <div class="events-title">Recent Activity</div>
        <div class="event-list" id="event-list">
          <div class="event-item" style="color: #52525b;">No activity yet...</div>
        </div>
      </div>
    `;
  }

  renderStages() {
    const flowEl = this.shadowRoot.getElementById("pipeline-flow");
    const metricsEl = this.shadowRoot.getElementById("metrics-grid");

    if (this.stageOrder.length === 0) return;

    // Render pipeline flow
    flowEl.innerHTML = this.stageOrder
      .map((stageName, i) => {
        const stage = this.stages.get(stageName);
        const icon = STAGE_ICONS[stage.shortName] || "⚙️";
        
        // Determine stage state classes
        const stateClasses = [];
        if (stage.isReceivingInput) stateClasses.push("receiving-input");
        if (stage.isProcessing) stateClasses.push("processing");
        if (stage.isOutputting) stateClasses.push("outputting");

        const colorRgb = this.hexToRgb(stage.color);
        
        // Arrow between stages - highlight if outputting to next
        const prevStage = i > 0 ? this.stages.get(this.stageOrder[i - 1]) : null;
        const arrowActive = prevStage?.isOutputting || stage.isReceivingInput;

        return `
          ${i > 0 ? `<span class="arrow ${arrowActive ? 'active flowing' : ''}" style="--arrow-color: ${prevStage?.color || stage.color};">→</span>` : ""}
          <div 
            class="stage-node ${stateClasses.join(" ")}" 
            data-stage="${stageName}"
            style="--stage-color: ${stage.color}; --stage-color-rgb: ${colorRgb};"
          >
            <span class="stage-icon">${icon}</span>
            <span class="stage-name">${stage.shortName}</span>
            <div class="flow-indicators">
              <div class="input-indicator ${stage.isReceivingInput ? 'active' : ''}"></div>
              <div class="output-indicator ${stage.isOutputting ? 'active' : ''}"></div>
            </div>
            <div class="processing-indicator">
              <span class="processing-dot"></span>
              <span class="processing-dot"></span>
              <span class="processing-dot"></span>
            </div>
          </div>
        `;
      })
      .join("");

    // Render metrics cards
    metricsEl.innerHTML = this.stageOrder
      .map((stageName) => {
        const stage = this.stages.get(stageName);
        const icon = STAGE_ICONS[stage.shortName] || "⚙️";
        const colorRgb = this.hexToRgb(stage.color);
        
        // Determine state indicator
        let stateIndicator = "";
        let stateClass = "";
        if (stage.isProcessing) {
          stateIndicator = "⏳ Processing";
          stateClass = "processing";
        } else if (stage.isReceivingInput) {
          stateIndicator = "← Receiving";
          stateClass = "receiving";
        } else if (stage.isOutputting) {
          stateIndicator = "→ Streaming";
          stateClass = "streaming";
        }

        return `
          <div class="metric-card ${stateClass}" style="--stage-color: ${stage.color}; --stage-color-rgb: ${colorRgb};">
            <div class="metric-header">
              <div class="metric-dot ${stateClass}"></div>
              <span class="metric-title">${icon} ${stage.shortName}</span>
              ${stateIndicator ? `<span class="state-badge ${stateClass}">${stateIndicator}</span>` : ""}
            </div>
            <div class="metric-values">
              <div class="metric-item">
                <span class="metric-label">Turn</span>
                <span class="metric-value">${stage.currentTurn}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">In / Out</span>
                <span class="metric-value">${stage.inputChunks} / ${stage.chunksProcessed}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Bytes Out</span>
                <span class="metric-value">${this.formatBytes(stage.totalBytes)}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">TTFC</span>
                <span class="metric-value">${stage.ttfc !== null ? stage.ttfc + "ms" : "-"}</span>
              </div>
            </div>
            ${stage.lastInputPreview || stage.lastChunkPreview ? `
              <div class="metric-previews">
                ${stage.lastInputPreview ? `<div class="metric-preview input">← "${this.truncate(stage.lastInputPreview, 30)}"</div>` : ""}
                ${stage.lastChunkPreview ? `<div class="metric-preview output">→ "${this.truncate(stage.lastChunkPreview, 30)}"</div>` : ""}
              </div>
            ` : ""}
          </div>
        `;
      })
      .join("");
  }

  renderEvents() {
    const eventListEl = this.shadowRoot.getElementById("event-list");

    if (this.recentEvents.length === 0) {
      eventListEl.innerHTML = '<div class="event-item" style="color: #52525b;">No activity yet...</div>';
      return;
    }

    eventListEl.innerHTML = this.recentEvents
      .map(
        (event) => `
        <div class="event-item">
          <span class="event-time">${this.formatAge(event.time)}</span>
          <span class="event-stage" style="color: ${event.color};">${event.stageName}</span>
          <span class="event-message">${event.message}</span>
        </div>
      `
      )
      .join("");
  }

  hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return "255, 255, 255";
    return `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`;
  }
}

customElements.define("pipeline-visualizer", PipelineVisualizer);

