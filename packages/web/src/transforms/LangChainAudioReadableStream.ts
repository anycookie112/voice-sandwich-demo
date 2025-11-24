/**
 * A ReadableStream wrapper that provides observability into transform pipeline stages.
 * Tracks timing, throughput, and logs information about each stage.
 */

import type { PipelineVisualizer } from "./PipelineVisualizer";

interface TurnMetrics {
  turnNumber: number;
  startedAt: number;
  firstChunkAt: number | null;
  chunksProcessed: number;
  totalBytes: number;
}

interface StageMetrics {
  name: string;
  chunksProcessed: number;
  totalBytes: number;
  firstChunkAt: number | null;
  lastChunkAt: number | null;
  startedAt: number;
  processingTimeMs: number;
  /** Per-turn metrics for this stage */
  turns: TurnMetrics[];
  /** Current turn being processed */
  currentTurn: TurnMetrics | null;
}

/** Common interface for visualizers */
interface VisualizerInterface {
  registerStage(stageName: string): void;
  onTurnStart(stageName: string, turnNumber: number): void;
  onInput(stageName: string, turnNumber: number, chunkPreview?: string): void;
  onProcessing(stageName: string, turnNumber: number): void;
  onFirstChunk(stageName: string, turnNumber: number, ttfc: number, chunkPreview?: string): void;
  onChunk(stageName: string, metrics: StageMetrics, chunkPreview?: string): void;
}

interface ObservabilityOptions {
  /** Enable verbose logging for each chunk */
  verbose?: boolean;
  /** Custom logger function (disabled when visualizer is used) */
  logger?: (message: string, data?: Record<string, unknown>) => void;
  /** Callback fired when a stage processes a chunk */
  onChunkProcessed?: (stageName: string, metrics: StageMetrics) => void;
  /** Callback fired when the pipeline completes */
  onPipelineComplete?: (allMetrics: Map<string, StageMetrics>) => void;
  /** 
   * Time in ms after which a new chunk is considered a new "turn" 
   * Default: 1000ms
   */
  turnIdleThresholdMs?: number;
  /**
   * Visualizer for streaming metrics to the frontend.
   * Pass a PipelineVisualizer instance.
   */
  visualizer?: PipelineVisualizer;
}

/** Internal options with resolved visualizer */
interface InternalOptions extends ObservabilityOptions {
  _visualizer?: VisualizerInterface;
}

/** No-op logger for when visualizer is active */
const noopLogger = () => {};

/**
 * Creates an observable TransformStream wrapper that tracks metrics
 */
function createObservableTransform<I, O>(
  transform: TransformStream<I, O>,
  stageName: string,
  metrics: Map<string, StageMetrics>,
  options: InternalOptions
): TransformStream<I, O> {
  const turnIdleThreshold = options.turnIdleThresholdMs ?? 1000;
  const visualizer = options._visualizer;
  
  const stageMetrics: StageMetrics = {
    name: stageName,
    chunksProcessed: 0,
    totalBytes: 0,
    firstChunkAt: null,
    lastChunkAt: null,
    startedAt: Date.now(),
    processingTimeMs: 0,
    turns: [],
    currentTurn: null,
  };
  metrics.set(stageName, stageMetrics);

  // Use no-op logger when visualizer is active to avoid log interference
  const log = visualizer ? noopLogger : (options.logger ?? console.log);

  // Register stage with visualizer
  if (visualizer) {
    visualizer.registerStage(stageName);
  } else {
    log(`[Pipeline] Stage "${stageName}" initialized`, { timestamp: new Date().toISOString() });
  }

  const reader = transform.readable.getReader();
  const writer = transform.writable.getWriter();

  /**
   * Start a new turn for this stage
   */
  function startNewTurn(now: number): TurnMetrics {
    // Finalize previous turn if exists
    if (stageMetrics.currentTurn) {
      stageMetrics.turns.push(stageMetrics.currentTurn);
    }
    
    const newTurn: TurnMetrics = {
      turnNumber: stageMetrics.turns.length + 1,
      startedAt: now,
      firstChunkAt: null,
      chunksProcessed: 0,
      totalBytes: 0,
    };
    stageMetrics.currentTurn = newTurn;
    return newTurn;
  }

  const observableReadable = new ReadableStream<O>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          // Finalize last turn
          if (stageMetrics.currentTurn) {
            stageMetrics.turns.push(stageMetrics.currentTurn);
            stageMetrics.currentTurn = null;
          }
          
          const totalDuration = Date.now() - stageMetrics.startedAt;
          log(`[Pipeline] Stage "${stageName}" completed`, {
            chunksProcessed: stageMetrics.chunksProcessed,
            totalBytes: stageMetrics.totalBytes,
            totalDurationMs: totalDuration,
            totalTurns: stageMetrics.turns.length,
            avgChunkSize: stageMetrics.chunksProcessed > 0 
              ? Math.round(stageMetrics.totalBytes / stageMetrics.chunksProcessed) 
              : 0,
          });
          controller.close();
          return;
        }

        const now = Date.now();
        const chunkSize = getChunkSize(value);
        const chunkPreview = getChunkPreview(value);
        const timeSinceLastChunk = stageMetrics.lastChunkAt 
          ? now - stageMetrics.lastChunkAt 
          : Infinity;

        // Detect new turn based on idle threshold
        const isNewTurn = !stageMetrics.currentTurn || timeSinceLastChunk > turnIdleThreshold;
        
        if (isNewTurn) {
          const turn = startNewTurn(now);
          if (visualizer) {
            visualizer.onTurnStart(stageName, turn.turnNumber);
          } else {
            log(`[Pipeline] Stage "${stageName}" turn #${turn.turnNumber} started`, {
              timeSinceLastChunkMs: timeSinceLastChunk === Infinity ? "first" : timeSinceLastChunk,
            });
          }
        }

        // Update overall metrics
        stageMetrics.chunksProcessed++;
        stageMetrics.totalBytes += chunkSize;
        stageMetrics.lastChunkAt = now;
        
        // Update turn metrics
        const currentTurn = stageMetrics.currentTurn!;
        currentTurn.chunksProcessed++;
        currentTurn.totalBytes += chunkSize;

        // Track first chunk (overall and per-turn)
        if (stageMetrics.firstChunkAt === null) {
          stageMetrics.firstChunkAt = now;
        }
        
        const isFirstChunkOfTurn = currentTurn.firstChunkAt === null;
        if (isFirstChunkOfTurn) {
          currentTurn.firstChunkAt = now;
          const timeToFirstChunk = now - currentTurn.startedAt;
          // Clear processing state - we're now outputting
          isProcessing = false;
          if (visualizer) {
            visualizer.onFirstChunk(stageName, currentTurn.turnNumber, timeToFirstChunk, chunkPreview);
          } else {
            log(`[Pipeline] Stage "${stageName}" turn #${currentTurn.turnNumber} first chunk`, {
              timeToFirstChunkMs: timeToFirstChunk,
              chunkSize,
              chunkPreview,
            });
          }
        }

        // Update visualizer with chunk data
        if (visualizer) {
          visualizer.onChunk(stageName, stageMetrics, chunkPreview);
        }

        if (options.verbose && !visualizer) {
          log(`[Pipeline] Stage "${stageName}" chunk #${stageMetrics.chunksProcessed} (turn #${currentTurn.turnNumber})`, {
            chunkSize,
            totalBytes: stageMetrics.totalBytes,
            turnBytes: currentTurn.totalBytes,
            elapsedMs: now - stageMetrics.startedAt,
          });
        }

        options.onChunkProcessed?.(stageName, { ...stageMetrics });
        controller.enqueue(value);
      } catch (error) {
        log(`[Pipeline] Stage "${stageName}" error`, { error: String(error) });
        controller.error(error);
      }
    },
    cancel(reason) {
      log(`[Pipeline] Stage "${stageName}" cancelled`, { reason: String(reason) });
      reader.cancel(reason);
    },
  });

  // Track if we're waiting for output (processing state)
  let isProcessing = false;
  let inputTurnNumber = 1;

  const observableWritable = new WritableStream<I>({
    async write(chunk) {
      const startTime = Date.now();
      
      // Check if this is meaningful input worth visualizing
      // Skip binary audio data (continuous stream) and empty strings
      const isMeaningfulInput = isSignificantChunk(chunk);
      
      if (isMeaningfulInput) {
        const chunkPreview = getChunkPreview(chunk);
        
        // Detect turn based on current state
        inputTurnNumber = stageMetrics.turns.length + 1;
        
        // Notify visualizer of input
        if (visualizer) {
          visualizer.onInput(stageName, inputTurnNumber, chunkPreview);
          
          // If not already processing, mark as processing (waiting for output)
          if (!isProcessing) {
            isProcessing = true;
            visualizer.onProcessing(stageName, inputTurnNumber);
          }
        }
      }
      
      await writer.write(chunk);
      stageMetrics.processingTimeMs += Date.now() - startTime;
    },
    async close() {
      await writer.close();
    },
    async abort(reason) {
      await writer.abort(reason);
    },
  });

  return {
    readable: observableReadable,
    writable: observableWritable,
  };
}

/**
 * Check if a chunk is significant enough to visualize as input
 * Filters out binary audio data and empty strings
 */
function isSignificantChunk(chunk: unknown): boolean {
  // Binary data (audio) - skip visualization (it's continuous)
  if (chunk instanceof Buffer || chunk instanceof ArrayBuffer || chunk instanceof Uint8Array) {
    return false;
  }
  
  // Empty or whitespace-only strings
  if (typeof chunk === "string") {
    return chunk.trim().length > 0;
  }
  
  // Objects (like AIMessageChunk) - check if they have meaningful content
  if (typeof chunk === "object" && chunk !== null) {
    // Check for content property (common in LangChain messages)
    if ("content" in chunk) {
      const content = (chunk as { content: unknown }).content;
      if (typeof content === "string") {
        return content.trim().length > 0;
      }
    }
    // Other objects are considered significant
    return true;
  }
  
  return true;
}

/**
 * Get a preview of chunk content for logging (truncated for readability)
 */
function getChunkPreview(chunk: unknown, maxLength: number = 50): string {
  if (typeof chunk === "string") {
    return chunk.length > maxLength ? chunk.slice(0, maxLength) + "..." : chunk;
  }
  if (chunk instanceof Buffer || chunk instanceof Uint8Array) {
    return `<binary ${chunk.length} bytes>`;
  }
  if (typeof chunk === "object" && chunk !== null) {
    const str = JSON.stringify(chunk);
    return str.length > maxLength ? str.slice(0, maxLength) + "..." : str;
  }
  return String(chunk);
}

/**
 * Get the size of a chunk in bytes
 */
function getChunkSize(chunk: unknown): number {
  if (chunk instanceof Buffer) {
    return chunk.length;
  }
  if (chunk instanceof ArrayBuffer) {
    return chunk.byteLength;
  }
  if (chunk instanceof Uint8Array) {
    return chunk.byteLength;
  }
  if (typeof chunk === "string") {
    return Buffer.byteLength(chunk, "utf-8");
  }
  if (typeof chunk === "object" && chunk !== null) {
    return JSON.stringify(chunk).length;
  }
  return 0;
}

/**
 * Extract a meaningful name from a transform object
 */
function getTransformName(transform: TransformStream<unknown, unknown>): string {
  const constructor = transform.constructor;
  if (constructor && constructor.name && constructor.name !== "TransformStream") {
    return constructor.name;
  }
  return "AnonymousTransform";
}

/**
 * LangChainAudioReadableStream - A ReadableStream with built-in observability
 * 
 * @example
 * ```ts
 * const pipeline = new LangChainAudioReadableStream(inputStream, {
 *   verbose: true,
 *   onPipelineComplete: (metrics) => {
 *     console.log("Pipeline complete!", metrics);
 *   }
 * })
 *   .pipeThrough(new AssemblyAISTTTransform({ ... }))
 *   .pipeThrough(new AgentTransform(agent))
 *   .pipeThrough(new AIMessageChunkTransform())
 *   .pipeThrough(new HumeTTSTransform({ ... }));
 * ```
 */
export class LangChainAudioReadableStream<T> extends ReadableStream<T> {
  private _metrics: Map<string, StageMetrics> = new Map();
  private _options: InternalOptions;
  private _stageIndex: number = 0;
  private _pipelineStartedAt: number;
  private _visualizer?: PipelineVisualizer;

  constructor(
    underlyingSource?: UnderlyingSource<T> | ReadableStream<T>,
    options: ObservabilityOptions = {}
  ) {
    if (underlyingSource instanceof ReadableStream) {
      // Wrap an existing ReadableStream
      const reader = underlyingSource.getReader();
      super({
        async pull(controller) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            return;
          }
          controller.enqueue(value);
        },
        cancel(reason) {
          reader.cancel(reason);
        },
      });
    } else {
      super(underlyingSource);
    }

    // Use visualizer if provided
    if (options.visualizer) {
      this._visualizer = options.visualizer;
    }
    
    this._options = {
      ...options,
      _visualizer: this._visualizer,
    };
    this._pipelineStartedAt = Date.now();
    
    // Only log if not using visualizer
    if (!this._visualizer) {
      const log = options.logger ?? console.log;
      log("[Pipeline] Stream initialized", { timestamp: new Date().toISOString() });
    }
  }

  /**
   * Pipe through a transform with observability
   */
  pipeThrough<O>(
    transform: TransformStream<T, O>,
    options?: StreamPipeOptions & { stageName?: string }
  ): LangChainAudioReadableStream<O> {
    this._stageIndex++;
    const stageName = options?.stageName ?? `${this._stageIndex}. ${getTransformName(transform)}`;
    
    const observableTransform = createObservableTransform(
      transform,
      stageName,
      this._metrics,
      this._options
    );

    // Use native pipeThrough
    const resultStream = super.pipeThrough(observableTransform, options);

    // Wrap result in a new LangChainAudioReadableStream to maintain chainability
    // Pass the existing visualizer to avoid creating a new one
    const wrappedStream = new LangChainAudioReadableStream<O>(resultStream, {
      ...this._options,
      visualizer: this._visualizer,
    });
    wrappedStream._metrics = this._metrics;
    wrappedStream._stageIndex = this._stageIndex;
    wrappedStream._pipelineStartedAt = this._pipelineStartedAt;
    wrappedStream._visualizer = this._visualizer;

    return wrappedStream;
  }
}

export type { StageMetrics, TurnMetrics, ObservabilityOptions };

