import "dotenv/config";

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { serve } from "@hono/node-server";
import { createNodeWebSocket } from "@hono/node-ws";
import wrtc from "@roamhq/wrtc";
import { agent } from "@voice-sandwich-demo/graphs";
import {
  AssemblyAISTTTransform,
  AgentTransform,
  AIMessageChunkTransform,
  HumeTTSTransform,
  LangChainAudioReadableStream,
  PipelineVisualizer,
} from "@voice-sandwich-demo/web";
import { Hono } from "hono";
import { cors } from "hono/cors";

const { RTCPeerConnection, RTCSessionDescription } = wrtc;

const app = new Hono();

const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });

app.use("/*", cors());

// Shared pipeline visualizer for WebSocket streaming
const pipelineVisualizer = new PipelineVisualizer();

// Serve static HTML
const htmlPath = join(process.cwd(), "src", "static", "index.html");
const html = readFileSync(htmlPath, "utf-8");

// Serve pipeline visualizer JS
const visualizerJsPath = fileURLToPath(import.meta.resolve("@voice-sandwich-demo/web/visualizer"));
const visualizerJs = readFileSync(visualizerJsPath, "utf-8");

app.get("/", (c) => c.html(html));

app.get("/pipeline-visualizer.js", (c) => {
  c.header("Content-Type", "application/javascript");
  return c.body(visualizerJs);
});

// Pipeline visualizer WebSocket endpoint (still uses WebSocket for visualization)
app.get(
  "/ws/pipeline",
  upgradeWebSocket(() => ({
    onOpen(_evt, ws) {
      console.log("Pipeline visualizer connected");
      pipelineVisualizer.setWebSocket(ws);
    },
    onClose() {
      console.log("Pipeline visualizer disconnected");
      pipelineVisualizer.clearWebSocket();
    },
  }))
);

// WebRTC signaling endpoint - uses WebSocket for signaling only
app.get(
  "/ws/signaling",
  upgradeWebSocket(() => {
    let peerConnection: InstanceType<typeof RTCPeerConnection> | null = null;
    let audioDataChannel: wrtc.RTCDataChannel | null = null;
    let controller: ReadableStreamDefaultController<Buffer>;
    let pipelineClosed = false;
    // Store WebSocket reference for use in callbacks
    let signalingWs: { send: (data: string) => void; readyState: number } | null = null;

    const inputStream = new ReadableStream<Buffer>({
      start(c) {
        controller = c;
      },
    });

    const observableStream = new LangChainAudioReadableStream(inputStream, {
      visualizer: pipelineVisualizer,
      turnIdleThresholdMs: 1000,
    });

    const pipeline = observableStream
      .pipeThrough(
        new AssemblyAISTTTransform({
          apiKey: process.env.ASSEMBLYAI_API_KEY!,
          sampleRate: 16000,
        })
      )
      .pipeThrough(new AgentTransform(agent))
      .pipeThrough(new AIMessageChunkTransform())
      .pipeThrough(new HumeTTSTransform({
        apiKey: process.env.HUME_API_KEY!,
        voiceName: process.env.HUME_VOICE_NAME!,
      }));

    const reader = pipeline.getReader();

    // Start reading from pipeline and send through data channel
    async function startPipelineReader() {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done || pipelineClosed) break;
          
          if (audioDataChannel && audioDataChannel.readyState === "open") {
            // Send audio data through WebRTC data channel
            audioDataChannel.send(value);
          }
        }
      } catch (e) {
        console.error("Pipeline error:", e);
      }
    }

    // Helper to send signaling messages
    function sendSignalingMessage(message: object) {
      if (signalingWs && signalingWs.readyState === 1) {
        signalingWs.send(JSON.stringify(message));
      }
    }

    return {
      onOpen(_evt, ws) {
        console.log("Signaling WebSocket connected");
        // Store the WebSocket reference
        signalingWs = ws;

        // Create peer connection with ICE servers
        peerConnection = new RTCPeerConnection({
          iceServers: [
            { urls: "stun:stun.l.google.com:19302" },
            { urls: "stun:stun1.l.google.com:19302" },
          ],
        });

        // Handle ICE candidates - send to client via signaling WebSocket
        peerConnection.onicecandidate = (event) => {
          if (event.candidate) {
            console.log("Sending ICE candidate to client");
            sendSignalingMessage({
              type: "ice-candidate",
              candidate: event.candidate,
            });
          }
        };

        peerConnection.onicegatheringstatechange = () => {
          console.log("ICE gathering state:", peerConnection?.iceGatheringState);
        };

        peerConnection.oniceconnectionstatechange = () => {
          console.log("ICE connection state:", peerConnection?.iceConnectionState);
        };

        peerConnection.onconnectionstatechange = () => {
          console.log("Connection state:", peerConnection?.connectionState);
          if (peerConnection?.connectionState === "connected") {
            console.log("WebRTC connection established");
          }
        };

        // Handle incoming data channel (created by client)
        peerConnection.ondatachannel = (event) => {
          const channel = event.channel;
          console.log("Data channel received:", channel.label);

          if (channel.label === "audio") {
            audioDataChannel = channel;

            channel.onopen = () => {
              console.log("Audio data channel open");
              // Start the pipeline reader when data channel opens
              startPipelineReader();
            };

            channel.onmessage = (msgEvent) => {
              // Receive audio data from client
              const data = msgEvent.data;
              if (data instanceof ArrayBuffer) {
                controller.enqueue(Buffer.from(data));
              } else if (typeof data === "string") {
                // Could be a control message
                console.log("Received string message:", data);
              }
            };

            channel.onclose = () => {
              console.log("Audio data channel closed");
            };

            channel.onerror = (err) => {
              console.error("Audio data channel error:", err);
            };
          }
        };
      },

      async onMessage(evt) {
        try {
          const message = JSON.parse(evt.data as string);

          if (message.type === "offer") {
            // Received SDP offer from client
            console.log("Received offer");
            await peerConnection!.setRemoteDescription(
              new RTCSessionDescription(message.sdp)
            );

            // Create and send answer
            const answer = await peerConnection!.createAnswer();
            await peerConnection!.setLocalDescription(answer);

            sendSignalingMessage({
              type: "answer",
              sdp: peerConnection!.localDescription,
            });
            console.log("Sent answer");
          } else if (message.type === "ice-candidate") {
            // Received ICE candidate from client
            console.log("Received ICE candidate from client");
            if (message.candidate) {
              await peerConnection!.addIceCandidate(message.candidate);
            }
          }
        } catch (e) {
          console.error("Error processing signaling message:", e);
        }
      },

      onClose() {
        console.log("Signaling WebSocket disconnected");
        pipelineClosed = true;
        signalingWs = null;
        
        // Clean up
        if (audioDataChannel) {
          audioDataChannel.close();
          audioDataChannel = null;
        }
        
        if (peerConnection) {
          peerConnection.close();
          peerConnection = null;
        }
        
        try {
          controller.close();
        } catch {
          // Ignore if already closed
        }
      },
    };
  })
);

const port = 3001;
const server = serve({
  fetch: app.fetch,
  port,
});

injectWebSocket(server);

console.log(`WebRTC Server running on http://localhost:${port}`);
