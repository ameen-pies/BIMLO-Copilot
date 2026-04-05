/**
 * CallPage.tsx
 *
 * A phone-call–style conversation page that chains:
 * mic → Whisper (/transcribe) → RAG agents (/query-stream) → Groq TTS (/tts) → speaker
 *
 * Key behaviours
 * ──────────────
 * • Voice Activity Detection (VAD) — energy + zero-crossing + sustained-duration gate.
 * Rejects coughs, clicks, background noise; only fires on real speech.
 * • Barge-in — if you speak while the AI is talking, audio stops within ~150 ms and
 * the system re-enters LISTENING immediately.
 * • Hands-free loop — after TTS finishes (or is interrupted) it auto-returns to LISTENING.
 * • Session memory — shares the same session_id as the main chat so history is continuous.
 * • All agents available — the /query-stream router picks RAG / report / graph etc.
 */

import React, {
  useState, useEffect, useRef, useCallback,
} from "react";

// ── Colour helpers ────────────────────────────────────────────────────────────
function hueToRgb(h: number, isDark: boolean): [number, number, number] {
  // light mode: very dark + saturated so lines contrast against pale bg
  const s = isDark ? 0.65 : 0.90;
  const l = isDark ? 0.55 : 0.30;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

function rgbToHex(r: number, g: number, b: number): number {
  return (Math.round(r * 255) << 16) | (Math.round(g * 255) << 8) | Math.round(b * 255);
}

function lerpRgb(
  a: [number,number,number],
  b: [number,number,number],
  t: number
): [number,number,number] {
  return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
}

function targetRgb(orbHue: number, isDark: boolean): [number,number,number] {
  // orbHue 0 = idle → indigo/blue; 360 = speaking → red
  if (orbHue === 0)   return isDark ? [0.39,0.40,0.95] : [0.08,0.35,0.90]; // indigo / deep blue
  if (orbHue === 360) return isDark ? [0.20,0.50,0.95] : [0.08,0.35,0.90]; // blue
  return hueToRgb(orbHue, isDark);
}

// ── Vanta NET background with side-only mask ──────────────────────────────────
const VantaBackground: React.FC<{ isDark: boolean; orbHue: number }> = ({ isDark, orbHue }) => {
  const vantaRef   = useRef<HTMLDivElement>(null);
  const effectRef  = useRef<any>(null);
  const readyRef   = useRef(false);
  const [visible, setVisible] = useState(false);

  // current animated rgb (stored in ref so RAF doesn't stale-close)
  const currentRgb = useRef<[number,number,number]>(targetRgb(orbHue, isDark));
  const targetRef  = useRef<[number,number,number]>(targetRgb(orbHue, isDark));
  const rafRef     = useRef<number>(0);

  // smooth lerp loop
  const startLerp = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    const tick = () => {
      const cur = currentRgb.current;
      const tgt = targetRef.current;
      const next = lerpRgb(cur, tgt, 0.035) as [number,number,number];
      const diff = Math.abs(next[0]-tgt[0]) + Math.abs(next[1]-tgt[1]) + Math.abs(next[2]-tgt[2]);
      currentRgb.current = next;
      if (effectRef.current) {
        effectRef.current.setOptions({ color: rgbToHex(...next), lineColor: rgbToHex(...next) });
      }
      if (diff > 0.001) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  // init Vanta once
  useEffect(() => {
    const loadScript = (src: string) =>
      new Promise<void>((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
        const s = document.createElement("script");
        s.src = src; s.onload = () => resolve(); s.onerror = reject;
        document.head.appendChild(s);
      });

    let cancelled = false;

    const init = async () => {
      await loadScript("https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js");
      await loadScript("https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js");
      if (cancelled || !vantaRef.current || !(window as any).VANTA) return;

      const initRgb = targetRgb(orbHue, isDark);
      currentRgb.current = initRgb;
      targetRef.current  = initRgb;

      effectRef.current = (window as any).VANTA.NET({
        el: vantaRef.current,
        mouseControls: true, touchControls: true, gyroControls: false,
        minHeight: 200, minWidth: 200, scale: 1.0, scaleMobile: 1.0,
        color: rgbToHex(...initRgb),
        lineColor: rgbToHex(...initRgb),
        backgroundColor: isDark ? 0x07080f : 0xf5f4fb,
        points: 5.0, maxDistance: 24.0, spacing: 20.0,
      });
      readyRef.current = true;
      setVisible(true);
    };

    init().catch(console.error);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      if (effectRef.current) { effectRef.current.destroy(); effectRef.current = null; }
      readyRef.current = false;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // animate toward new target when orbHue or theme changes
  useEffect(() => {
    if (!readyRef.current) return;
    targetRef.current = targetRgb(orbHue, isDark);
    startLerp();
  }, [orbHue, isDark, startLerp]);

  // instantly swap bg color on theme change
  useEffect(() => {
    if (!readyRef.current || !effectRef.current) return;
    effectRef.current.setOptions({ backgroundColor: isDark ? 0x07080f : 0xf5f4fb });
  }, [isDark]);

  return (
    <div
      ref={vantaRef}
      style={{
        position: "fixed", inset: 0, zIndex: 0,
        opacity: visible ? 1 : 0,
        transition: "opacity 0.8s ease",
        WebkitMaskImage:
          "linear-gradient(to right, black 0%, black 18%, transparent 38%, transparent 62%, black 82%, black 100%)",
        maskImage:
          "linear-gradient(to right, black 0%, black 18%, transparent 38%, transparent 62%, black 82%, black 100%)",
        pointerEvents: "none",
      }}
    />
  );
};
import { useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { PhoneOff, Mic, MicOff, Volume2, VolumeX, ChevronDown, X } from "lucide-react";
import Orb from "../components/Orb";
import ThemeToggle from "../components/ThemeToggle";

// ── API base (same helper as Chat.tsx) ───────────────────────────────────────
const API =
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
  "http://localhost:8000";

// ── VAD tuning ────────────────────────────────────────────────────────────────
const VAD_ENERGY_THRESHOLD    = 0.030;  // RMS energy gate — raised to reject bg noise/TV/AC
const VAD_ZCR_MAX             = 0.35;   // tighter ZCR ceiling; speech is low-ZCR, noise is high
const VAD_HOLD_MS             = 700;    // silence after speech before we cut — short leeway for second thoughts
const VAD_MIN_SPEECH_MS       = 500;    // minimum voiced duration — rejects short noise bursts
const VAD_BARGE_IN_ENERGY     = 0.040;  // higher bar for barge-in to avoid TTS feedback
const VAD_FRAME_MS            = 30;     // analysis frame duration

// ── Call state machine ────────────────────────────────────────────────────────
type CallState =
  | "idle"          // before call starts
  | "connecting"    // acquiring mic
  | "listening"     // actively capturing, waiting for speech
  | "detecting"     // speech energy detected, filling buffer
  | "transcribing"  // sending audio to Whisper
  | "thinking"      // waiting for RAG response
  | "speaking"      // playing TTS audio
  | "ended";        // call finished

interface Turn {
  role:    "user" | "assistant";
  content: string;
  id:      string;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function rms(buf: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  return Math.sqrt(sum / buf.length);
}

function zcr(buf: Float32Array): number {
  let crossings = 0;
  for (let i = 1; i < buf.length; i++) {
    if ((buf[i] >= 0) !== (buf[i - 1] >= 0)) crossings++;
  }
  return crossings / buf.length;
}

// Strip markdown for cleaner TTS (no "hashtag" or asterisks read aloud)
function stripMarkdown(text: string): string {
  return text
    .replace(/#{1,6}\s/g, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/`{1,3}[^`]*`{1,3}/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/>\s*/g, "")
    .replace(/[-*+]\s/g, "")
    .replace(/\n{2,}/g, ". ")
    .replace(/\n/g, " ")
    .trim();
}

// ── Status labels ─────────────────────────────────────────────────────────────
const STATE_LABEL: Record<CallState, string> = {
  idle:         "Ready",
  connecting:   "Connecting…",
  listening:    "Listening…",
  detecting:    "Hearing you…",
  transcribing: "Understanding…",
  thinking:     "Thinking…",
  speaking:     "Speaking…",
  ended:        "Call ended",
};

const STATE_COLOR: Record<CallState, string> = {
  idle:         "text-zinc-400",
  connecting:   "text-amber-400",
  listening:    "text-emerald-400",
  detecting:    "text-sky-400",
  transcribing: "text-violet-400",
  thinking:     "text-amber-400",
  speaking:     "text-blue-400",
  ended:        "text-zinc-500",
};

// ── Component ─────────────────────────────────────────────────────────────────
const CallPage: React.FC = () => {
  const navigate   = useNavigate();
  const location   = useLocation();
  const locState   = (location.state ?? {}) as { sessionId?: string; convId?: string };

  // Inherit the chat session if launched from Chat, otherwise start fresh
  const sessionRef  = useRef<string>(locState.sessionId ?? `call-${uuid4()}`);
  // Remember which conv to post the call-card back to
  const convIdRef   = useRef<string>(locState.convId ?? "default");
  // Track when the call actually started (for the card timestamp)
  const callStartRef = useRef<Date>(new Date());

  // ── State ──────────────────────────────────────────────────────────────────
  const [callState, setCallState]   = useState<CallState>("idle");
  const [turns, setTurns]           = useState<Turn[]>([]);
  const [muted, setMuted]           = useState(false);
  const [speakerOff, setSpeakerOff] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState("hannah");
  const [showVoicePicker, setShowVoicePicker] = useState(false);
  const [liveTranscript, setLiveTranscript]   = useState("");
  const [thinkingMessage, setThinkingMessage] = useState("");
  // Subtitle state: current sentence key (for AnimatePresence) + words revealed so far
  const [subtitleKey,   setSubtitleKey]   = useState(0);
  const [subtitleWords, setSubtitleWords] = useState<string[]>([]);
  const spokenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const waveBarRefsRef = useRef<(HTMLSpanElement | null)[]>([]);
  const [callDuration, setCallDuration] = useState(0);
  const [error, setError]           = useState<string | null>(null);
  const [showSilenceWarning, setShowSilenceWarning] = useState(false);
  const [isDark, setIsDark]         = useState(() =>
    document.documentElement.classList.contains("dark")
  );

  // Keep isDark in sync whenever ThemeToggle mutates the html class
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  // ── Refs ───────────────────────────────────────────────────────────────────
  const stateRef          = useRef<CallState>("idle");
  const mutedRef          = useRef(false);
  const speakerOffRef     = useRef(false);

  const streamRef         = useRef<MediaStream | null>(null);
  const audioCtxRef       = useRef<AudioContext | null>(null);
  const analyserRef       = useRef<AnalyserNode | null>(null);
  const recorderRef       = useRef<MediaRecorder | null>(null);
  const chunksRef         = useRef<Blob[]>([]);
  const mimeRef           = useRef("audio/webm;codecs=opus");

  const speechStartRef    = useRef<number>(0);
  const lastVoicedRef     = useRef<number>(0);
  const vadTimerRef       = useRef<ReturnType<typeof setInterval> | null>(null);
  const animFrameRef      = useRef<number>(0);

  const ttsAudioRef       = useRef<HTMLAudioElement | null>(null);
  const ttsAbortRef       = useRef<AbortController | null>(null);
  const ragAbortRef       = useRef<AbortController | null>(null);
  const speakWavRef       = useRef<((b64: string, text: string, isFiller: boolean) => Promise<void>) | null>(null);
  const startListeningRef = useRef<(() => void) | null>(null);

  const durationTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const silenceWarnShownRef = useRef(false);

  // Keep refs in sync with state
  useEffect(() => { stateRef.current   = callState; }, [callState]);
  useEffect(() => { mutedRef.current   = muted;     }, [muted]);
  useEffect(() => { speakerOffRef.current = speakerOff; }, [speakerOff]);

  // Duration timer
  useEffect(() => {
    if (callState === "listening" || callState === "detecting" ||
        callState === "transcribing" || callState === "thinking" ||
        callState === "speaking") {
      if (!durationTimerRef.current) {
        durationTimerRef.current = setInterval(
          () => setCallDuration(d => d + 1), 1000
        );
      }
    } else {
      if (durationTimerRef.current) {
        clearInterval(durationTimerRef.current);
        durationTimerRef.current = null;
      }
    }
  }, [callState]);

  // ── Helpers ────────────────────────────────────────────────────────────────
  function setState(s: CallState) {
    stateRef.current = s;
    setCallState(s);
  }

  function addTurn(role: "user" | "assistant", content: string) {
    setTurns(prev => [...prev, { role, content, id: uuid4() }]);
  }

  // ── VAD loop (runs on animationFrame while LISTENING or DETECTING) ─────────
  const startVAD = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const bufSize = analyser.fftSize;
    const buf     = new Float32Array(bufSize);

    let inSpeech     = false;
    let speechStart  = 0;
    let lastVoiced   = 0;
    let lastHeardTime    = Date.now();
    silenceWarnShownRef.current = false;

    const tick = () => {
      animFrameRef.current = requestAnimationFrame(tick);

      const st = stateRef.current;
      if (st !== "listening" && st !== "detecting") return;

      analyser.getFloatTimeDomainData(buf);
      const energy = rms(buf);
      const rate   = zcr(buf);

      // ── Waveform visualisation (direct DOM — avoids setState re-render loop) ─
      const bars = waveBarRefsRef.current;
      if (bars.length) {
        const BAR_COUNT = bars.length;
        for (let b = 0; b < BAR_COUNT; b++) {
          const el = bars[b];
          if (!el) continue;
          const start = Math.floor((b / BAR_COUNT) * bufSize);
          const end   = Math.floor(((b + 1) / BAR_COUNT) * bufSize);
          let sq = 0;
          for (let j = start; j < end; j++) sq += buf[j] * buf[j];
          const amp = Math.min(1, Math.sqrt(sq / (end - start)) * 8);
          const minH = 3, maxH = 26;
          el.style.height = `${Math.max(minH, Math.round(minH + amp * (maxH - minH)))}px`;
        }
      }

      if (mutedRef.current) return;

      const isSpeech = energy > VAD_ENERGY_THRESHOLD && rate < VAD_ZCR_MAX;

      // ── Silence warning ───────────────────────────────────────────────────
      if (energy > 0.04) {
        lastHeardTime = Date.now();
        if (silenceWarnShownRef.current) {
          silenceWarnShownRef.current = false;
          setShowSilenceWarning(false);
        }
      } else if (!silenceWarnShownRef.current && Date.now() - lastHeardTime > 3000) {
        silenceWarnShownRef.current = true;
        setShowSilenceWarning(true);
      }

      if (isSpeech) {
        lastVoiced = Date.now();
        if (!inSpeech) {
          inSpeech    = true;
          speechStart = Date.now();
          setState("detecting");
        }
      } else if (inSpeech) {
        const silenceMs = Date.now() - lastVoiced;
        if (silenceMs > VAD_HOLD_MS) {
          // Silence held long enough — check minimum speech duration
          const speechMs = Date.now() - speechStart;
          inSpeech = false;
          if (speechMs >= VAD_MIN_SPEECH_MS) {
            // Real speech — commit
            cancelAnimationFrame(animFrameRef.current);
            commitSpeech();
          } else {
            // Too short — noise, reset
            setState("listening");
          }
        }
      }
    };

    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  // ── Barge-in detection (runs while SPEAKING) ───────────────────────────────
  const startBargeInDetector = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const bufSize = analyser.fftSize;
    const buf     = new Float32Array(bufSize);
    let voicedCount = 0;

    const tick = () => {
      if (stateRef.current !== "speaking") return;
      animFrameRef.current = requestAnimationFrame(tick);

      analyser.getFloatTimeDomainData(buf);
      const energy = rms(buf);
      const rate   = zcr(buf);

      const isSpeech = energy > VAD_BARGE_IN_ENERGY && rate < VAD_ZCR_MAX;
      if (isSpeech) {
        voicedCount++;
        // 3 consecutive voiced frames (~90 ms) → barge-in confirmed
        if (voicedCount >= 3) {
          cancelAnimationFrame(animFrameRef.current);
          handleBargeIn();
        }
      } else {
        voicedCount = 0;
      }
    };

    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  // ── Stop TTS immediately with a natural stutter/cutoff ───────────────────
  const stopTTS = useCallback((withStutter = false) => {
    ttsAbortRef.current?.abort();
    ttsAbortRef.current = null;
    if (spokenTimerRef.current) {
      const ref = spokenTimerRef.current as any;
      if (ref._cancel)   ref._cancel();
      if (ref._timeouts) ref._timeouts.forEach(clearTimeout);
      else if (typeof ref === "number") clearInterval(ref);
      spokenTimerRef.current = null;
    }
    setSubtitleWords([]);

    const audio = ttsAudioRef.current;
    if (!audio) return;

    if (withStutter && !audio.paused && audio.duration > 0.3) {
      // Natural stutter: replay the last ~220ms of audio at reduced volume,
      // then hard-cut — mimics a human being interrupted mid-word.
      const stutterStart = Math.max(0, audio.currentTime - 0.22);
      audio.currentTime  = stutterStart;
      audio.volume       = 0.45;
      // Hard cut after ~180ms
      const cutTimer = setTimeout(() => {
        audio.pause();
        audio.src = "";
        ttsAudioRef.current = null;
      }, 180);
      // Safety: also cut if audio ends naturally before the timer
      audio.addEventListener("ended", () => {
        clearTimeout(cutTimer);
        ttsAudioRef.current = null;
      }, { once: true });
    } else {
      audio.pause();
      audio.src = "";
      ttsAudioRef.current = null;
    }
  }, []);

  // ── Barge-in handler ───────────────────────────────────────────────────────
  const handleBargeIn = useCallback(() => {
    stopTTS(true);
    // startListening is defined later — call via a small timeout so React's
    // hook order is respected and the circular dep is avoided.
    setState("detecting");
    setTimeout(() => {
      if (stateRef.current !== "ended") startListeningRef.current?.();
    }, 0);
  }, [stopTTS]);

  // ── Commit speech: stop recorder → transcribe → query → TTS ───────────────
  const commitSpeech = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      // Recorder already stopped — start a fresh one then commit
      startListening();
      return;
    }

    // Stop recorder; onstop will fire with the chunks
    recorder.stop();
  }, []);

  // ── Process committed audio blob ───────────────────────────────────────────
  const processBlob = useCallback(async (blob: Blob) => {
    if (stateRef.current === "ended") return;

    // WebM init segments (header-only, no audio frames) are typically ~965 bytes.
    // Real speech is always well above 5 KB — reject anything smaller outright.
    if (blob.size < 5000) {
      startListening();
      return;
    }

    setState("transcribing");
    setLiveTranscript("");

    try {
      // ── 1. Transcribe ──────────────────────────────────────────────────
      const form = new FormData();
      form.append("audio", blob, "recording.webm");
      form.append("mime_type", mimeRef.current);

      const txRes = await fetch(`${API}/transcribe`, { method: "POST", body: form });
      if (!txRes.ok) throw new Error("Transcription failed");

      const { transcript } = (await txRes.json()) as { transcript: string };
      const rawText = transcript?.trim();

      if (!rawText || rawText.length < 3) { startListening(); return; }
      // Reject transcripts that are just punctuation/noise
      if (/^[.،,;:!?…\s]+$/.test(rawText)) { startListening(); return; }
      if (stateRef.current === "ended") return;

      const text = rawText;
      setLiveTranscript(text);
      setTurns(prev => [...prev, { role: "user", content: text, id: uuid4() }]);

      if (stateRef.current === "ended") return;

      // ── 2. RAG query + spoken thought-process narration ───────────────
      setState("thinking");
      setThinkingMessage("");
      ragAbortRef.current = new AbortController();

      // ── Serial spoken queue ────────────────────────────────────────────
      // Status phrases are enqueued as the RAG graph emits nodes.
      // A single async drain loop plays them one-by-one so they never overlap.
      // After RAG finishes we wait for the queue to empty, then play the answer.
      const spokenStatuses  = new Set<string>();
      const speechQueue: string[] = [];
      let   queueDraining   = false;

      const enqueueSpeech = (phrase: string) => {
        if (speakerOffRef.current || spokenStatuses.has(phrase)) return;
        spokenStatuses.add(phrase);
        speechQueue.push(phrase);
        if (!queueDraining) drainQueue();
      };

      const drainQueue = async () => {
        queueDraining = true;
        while (speechQueue.length > 0) {
          const phrase = speechQueue.shift()!;
          if (stateRef.current === "ended") break;
          try {
            const r = await fetch(`${API}/tts`, {
              method:  "POST",
              headers: { "Content-Type": "application/json" },
              body:    JSON.stringify({ text: phrase, voice: selectedVoice }),
            });
            if (!r.ok) continue;
            const bytes = await r.arrayBuffer();
            if (!bytes.byteLength || stateRef.current === "ended") continue;
            const rec = recorderRef.current;
            if (rec && rec.state !== "inactive") { rec.onstop = () => {}; rec.stop(); }
            recorderRef.current = null; chunksRef.current = [];
            setState("speaking");
            const b64 = btoa(String.fromCharCode(...new Uint8Array(bytes)));
            await speakWavRef.current!(b64, phrase, true);
            if (stateRef.current !== "ended") setState("thinking");
          } catch { /* non-fatal */ }
        }
        queueDraining = false;
      };

      // Friendly spoken phrases per RAG node
      const NODE_PHRASES: Record<string, string> = {
        retrieve:         "Let me look that up…",
        rewrite_query:    "Rephrasing your question…",
        check_retrieval:  "Checking what I found…",
        judge_plan:       "Planning my answer…",
        synthesise:       "Putting it together…",
        judge_evaluate:   "Double-checking…",
        graph_node:       "Querying the knowledge graph…",
        report_node:      "Building a report…",
        analytics_node:   "Running the analysis…",
      };

      const SLOW_NODES = new Set(Object.keys(NODE_PHRASES));

      const qRes = await fetch(`${API}/query-stream`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          query:      `[Voice call — answer conversationally and concisely. Never say you cannot hear.]\n\nUser said: ${text}`,
          session_id: sessionRef.current,
          top_k:      3,
          voice_mode: true,
        }),
        signal: ragAbortRef.current.signal,
      });
      if (!qRes.ok) throw new Error("Query failed");

      let ragAnswer = "";
      const reader  = qRes.body!.getReader();
      const decoder = new TextDecoder();
      let   ssBuf   = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        ssBuf += decoder.decode(value, { stream: true });
        const lines = ssBuf.split("\n");
        ssBuf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "status") {
              if (ev.message) setThinkingMessage(ev.message);
              if (ev.node && SLOW_NODES.has(ev.node)) {
                enqueueSpeech(NODE_PHRASES[ev.node]);
              }
            }
            if (ev.type === "result") ragAnswer = ev.answer ?? "";
          } catch { /* skip */ }
        }
      }

      ragAnswer = ragAnswer.trim();
      if (!ragAnswer) { startListening(); return; }
      if (stateRef.current === "ended") return;
      if (speakerOffRef.current) { startListening(); return; }

      // ── 3. Drain remaining queue, then play the answer immediately ────
      while (queueDraining || speechQueue.length > 0) {
        await new Promise(r => setTimeout(r, 30));
        if (stateRef.current === "ended") return;
      }

      setState("speaking");
      ttsAbortRef.current = new AbortController();

      // Stop mic while AI speaks
      const activeRec = recorderRef.current;
      if (activeRec && activeRec.state !== "inactive") {
        activeRec.onstop = () => {};
        activeRec.stop();
      }
      recorderRef.current = null;
      chunksRef.current   = [];

      // Hit /tts directly — saves one full round-trip vs /call/respond
      const ttsRes = await fetch(`${API}/tts`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text: ragAnswer, voice: selectedVoice }),
        signal:  ttsAbortRef.current.signal,
      });
      if (!ttsRes.ok) throw new Error("TTS failed");
      const ttsBytes = await ttsRes.arrayBuffer();
      if (!ttsBytes.byteLength || stateRef.current === "ended") return;

      addTurn("assistant", ragAnswer);
      const answerB64 = btoa(String.fromCharCode(...new Uint8Array(ttsBytes)));
      await speakWavRef.current!(answerB64, ragAnswer, false);

    } catch (err: any) {
      if (err?.name === "AbortError") return;
      console.error("[CallPage] processBlob error:", err);
      setError(String(err));
    } finally {
      const st = stateRef.current;
      // Don't restart if already listening/detecting (continuation window did it)
      // or if the call ended
      if (st !== "ended" && st !== "listening" && st !== "detecting") startListening();
    }
  }, [selectedVoice]);

  // ── Play a base64 MP3 chunk with tightly-synced word subtitles ──────────────
  const speakWav = useCallback(async (
    b64: string,
    spokenText: string,
    isFiller: boolean,
  ): Promise<void> => {
    if (stateRef.current === "ended") return;

    const raw   = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const blob  = new Blob([bytes], { type: "audio/mpeg" });
    const url   = URL.createObjectURL(blob);
    const audio = new Audio(url);
    ttsAudioRef.current = audio;

    const words = spokenText.trim().split(/\s+/).filter(Boolean);
    setSubtitleWords([]);

    await new Promise<void>((resolve) => {
      audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
      audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };

      // ── Subtitle sync ──────────────────────────────────────────────────────
      // Cues are stored as 0-1 PROGRESS FRACTIONS (not seconds), so they stay
      // in sync even when the browser's currentTime resolution drifts on long audio.
      // Each rAF tick we recompute progress = currentTime / duration and fire any
      // cues whose fraction has been passed — self-correcting by design.

      let rafId  = 0;
      let cueIdx = 0;

      type Cue =
        | { kind: "word";  frac: number; word: string }
        | { kind: "flush"; frac: number };

      let cues: Cue[] = [];

      const buildCues = (durSec: number) => {
        if (!words.length || isFiller || durSec <= 0) return;

        const sylCount = (w: string) => {
          const chars = w.replace(/[^a-zA-Z0-9\u00C0-\u024F\u0600-\u06FF]/g, "").length || 1;
          return Math.max(1, Math.ceil(chars / 3));
        };
        const pauseWeight = (w: string): number => {
          if (/[.!?؟。！？]$/.test(w)) return 1.8;
          if (/[…]$/.test(w))          return 1.2;
          if (/[,،;:]$/.test(w))       return 0.7;
          if (/[-–—]/.test(w))         return 0.4;
          return 0;
        };

        const totalWeight = words.reduce((s, w) => s + sylCount(w) + pauseWeight(w), 0);
        // fraction-per-unit; push start slightly into audio to account for ElevenLabs lead-in
        const LEAD_IN_FRAC = Math.min(0.04, 0.08 / durSec); // ~80ms lead-in, capped at 4%
        const usable       = (1 - LEAD_IN_FRAC) * 0.97;    // leave last 3% for trailing audio
        const fracPerUnit  = usable / totalWeight;

        let cursor = LEAD_IN_FRAC;
        cues = [];

        for (const w of words) {
          const wFrac = sylCount(w) * fracPerUnit;
          cues.push({ kind: "word", frac: cursor, word: w });
          cursor += wFrac;

          const pWeight = pauseWeight(w);
          if (pWeight > 0) {
            const pFrac = pWeight * fracPerUnit;
            if (/[.!?؟。！？]$/.test(w)) {
              cues.push({ kind: "flush", frac: cursor + pFrac * 0.55 });
            }
            cursor += pFrac;
          }
        }
      };

      const startRaf = (durSec: number) => {
        if (!cues.length) return;
        const tick = () => {
          if (stateRef.current === "ended") return;
          // Use fraction so long-audio drift cancels out automatically
          const progress = durSec > 0 ? audio.currentTime / durSec : 0;
          while (cueIdx < cues.length && progress >= cues[cueIdx].frac) {
            const cue = cues[cueIdx++];
            if (cue.kind === "word") {
              setSubtitleWords(prev => [...prev, cue.word]);
            } else {
              setSubtitleKey(k => k + 1);
              setSubtitleWords([]);
            }
          }
          if (cueIdx < cues.length) rafId = requestAnimationFrame(tick);
        };
        rafId = requestAnimationFrame(tick);
        spokenTimerRef.current = {
          _timeouts: [],
          _cancel: () => cancelAnimationFrame(rafId),
        } as any;
      };

      // canplaythrough = duration is confirmed; play + start rAF immediately
      const onReady = () => {
        const dur = audio.duration;
        buildCues(dur);
        audio.play().catch(() => resolve());
        startRaf(dur);
        if (!isFiller) startBargeInDetector();
      };
      audio.addEventListener("canplaythrough", onReady, { once: true });

      // Fallback: canplaythrough may not fire on some browsers for blob URLs
      const fallbackTimer = setTimeout(() => {
        if (cues.length === 0) {
          // Estimate duration at 155 WPM if still unknown
          const dur = isFinite(audio.duration) && audio.duration > 0
            ? audio.duration
            : words.length * (60 / 155);
          buildCues(dur);
          audio.play().catch(() => resolve());
          startRaf(dur);
          if (!isFiller) startBargeInDetector();
        }
      }, 280);

      const cleanup = () => { cancelAnimationFrame(rafId); clearTimeout(fallbackTimer); };
      audio.addEventListener("ended", cleanup, { once: true });
      audio.addEventListener("error", cleanup, { once: true });

      audio.preload = "auto";
      audio.load();
    });

    ttsAudioRef.current = null;
  }, [startBargeInDetector]);

  // ── Start (or restart) the listening state ─────────────────────────────────
  const startListening = useCallback(() => {
    if (stateRef.current === "ended") return;

    const stream = streamRef.current;
    if (!stream) return;

    const mime     = mimeRef.current;
    const recorder = new MediaRecorder(stream, { mimeType: mime });
    recorderRef.current = recorder;
    // Cleared AFTER assigning recorderRef so any stale ondataavailable firing
    // between here and recorder.start() is either ref-guarded or immediately flushed.
    chunksRef.current = [];

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      // Guard: only process if this recorder is still the active one.
      // Prevents stale onstop callbacks from sending header-only blobs.
      if (recorderRef.current !== recorder) return;
      const blob = new Blob(chunksRef.current, { type: mime });
      chunksRef.current = [];
      processBlob(blob);
    };

    recorder.start(100);
    setState("listening");
    startVAD();
  }, [processBlob, startVAD]);

  // Keep refs in sync so callbacks defined earlier can call functions defined later
  useEffect(() => { speakWavRef.current = speakWav; }, [speakWav]);
  useEffect(() => { startListeningRef.current = startListening; }, [startListening]);

  // ── Start the call ─────────────────────────────────────────────────────────
  const startCall = useCallback(async () => {
    setState("connecting");
    setError(null);
    setTurns([]);
    setCallDuration(0);
    setLiveTranscript("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation:   true,
          noiseSuppression:   true,
          autoGainControl:    true,
          sampleRate:         16000,
        },
      });
      streamRef.current = stream;

      // Web Audio analyser for VAD + waveform
      const ctx      = new AudioContext({ sampleRate: 16000 });
      const source   = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize            = 512;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);
      audioCtxRef.current  = ctx;
      analyserRef.current  = analyser;

      // Pick best MIME
      mimeRef.current = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/ogg";

      // Record when the call started (used for the call card)
      callStartRef.current = new Date();

      startListening();

    } catch (err: any) {
      setState("idle");
      setError(err?.message ?? "Microphone access denied");
    }
  }, [startListening]);

  // ── End the call ───────────────────────────────────────────────────────────
  const endCall = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);

    // Set ended FIRST — so the recorder's onstop → processBlob guard bails immediately
    setState("ended");

    stopTTS();

    // Cancel any in-flight RAG stream
    ragAbortRef.current?.abort();
    ragAbortRef.current = null;

    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach(t => t.stop());
    audioCtxRef.current?.close().catch(() => {});

    streamRef.current  = null;
    audioCtxRef.current = null;
    analyserRef.current = null;
    recorderRef.current = null;

    waveBarRefsRef.current.forEach(el => { if (el) el.style.height = "3px"; });
    setLiveTranscript("");
    setShowSilenceWarning(false);
    silenceWarnShownRef.current = false;

    // Fire call-ended so Chat can render the call card.
    // Use callDuration via a ref snapshot — the state variable may be stale inside
    // this callback, so we read from the timer's perspective via Date diff.
    const elapsed = Math.round((Date.now() - callStartRef.current.getTime()) / 1000);
    if (elapsed > 2) {
      window.dispatchEvent(new CustomEvent("call-ended", {
        detail: {
          duration:  elapsed,
          startedAt: callStartRef.current.toISOString(),
          convId:    convIdRef.current,
        },
      }));
    }
  }, [stopTTS]);

  // Cleanup on unmount
  useEffect(() => () => { endCall(); }, []);

  // ── Format duration ────────────────────────────────────────────────────────
  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // ── Render ─────────────────────────────────────────────────────────────────
  const isActive = !["idle", "ended", "connecting"].includes(callState);

  // hue shifts per state to drive the Orb color
  const orbHue =
    callState === "speaking"    ? 360 :   // red (distinct from idle 0)
    callState === "thinking"    ? 40  :   // amber
    callState === "transcribing"? 40  :   // amber
    callState === "detecting"   ? 150 :   // teal
    callState === "listening"   ? 150 :   // teal
    callState === "connecting"  ? 200 :
    0;

  const orbForceHover = callState === "speaking";
  const orbHoverIntensity =
    callState === "speaking"    ? 0.55 :
    callState === "thinking" || callState === "transcribing" ? 0.3 :
    isActive                    ? 0.35 : 0.18;

  // Static blue — no state switching, no flash

  return (
    <motion.div
      className="min-h-screen flex flex-col items-center justify-between px-4 py-6 select-none overflow-hidden"
      style={{ background: isDark ? "#07080f" : "#f5f4fb", transition: "background 0.15s ease" }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >

      {/* ── Vanta NET background — side-masked ── */}
      <VantaBackground isDark={isDark} orbHue={orbHue} />

      {/* ── Silence warning banner ── */}
      <AnimatePresence>
        {showSilenceWarning && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="fixed top-4 left-0 right-0 flex justify-center z-50 pointer-events-none"
          >
            <div className="pointer-events-auto inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-950 border border-red-500/50 text-red-400 text-xs font-medium shadow-sm">
              <span>😢</span>
              <span>We're having trouble hearing you — check your mic is connected and unmuted.</span>
              <button onClick={() => { silenceWarnShownRef.current = false; setShowSilenceWarning(false); }} className="ml-1 hover:text-red-300 transition-colors">
                <X className="h-3 w-3" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>


      <div className="relative z-10 w-full max-w-md flex items-center justify-between px-1">
        <button
          onClick={() => navigate(locState.convId ? "/chat" : "/")}
          className="flex items-center gap-1.5 text-foreground/30 hover:text-foreground/70 text-xs transition-colors"
        >
          <span className="text-base leading-none">←</span>
          <span>Back</span>
        </button>

        <div className="flex items-center gap-3">
          {isActive && (
            <span className="flex items-center gap-1.5 text-xs font-mono text-foreground/25">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/70 animate-pulse" />
              {fmt(callDuration)}
            </span>
          )}
          <ThemeToggle />
        </div>

      </div>

      {/* ── Orb + status ── */}
      <div className="relative z-10 flex flex-col items-center gap-8 flex-1 justify-center w-full max-w-xs">

        {/* Orb container */}
        <div className="relative flex items-center justify-center">

          {/* Outer glow ring — state-aware */}
          <AnimatePresence>
            {callState === "speaking" && (
              <>
                <motion.div key="r1"
                  className="absolute rounded-full border border-blue-500/20"
                  initial={{ width: 220, height: 220, opacity: 0.7 }}
                  animate={{ width: 320, height: 320, opacity: 0 }}
                  transition={{ duration: 2.2, repeat: Infinity, ease: "easeOut" }}
                />
                <motion.div key="r2"
                  className="absolute rounded-full border border-blue-400/10"
                  initial={{ width: 220, height: 220, opacity: 0.5 }}
                  animate={{ width: 380, height: 380, opacity: 0 }}
                  transition={{ duration: 2.2, repeat: Infinity, ease: "easeOut", delay: 0.7 }}
                />
              </>
            )}
            {(callState === "listening" || callState === "detecting") && (
              <motion.div key="lr"
                className="absolute rounded-full border border-teal-500/20"
                initial={{ width: 220, height: 220, opacity: 0.5 }}
                animate={{ width: 280, height: 280, opacity: 0 }}
                transition={{ duration: 1.6, repeat: Infinity, ease: "easeOut" }}
              />
            )}
            {(callState === "thinking" || callState === "transcribing") && (
              <motion.div key="tr"
                className="absolute rounded-full border border-amber-500/15"
                initial={{ width: 220, height: 220, opacity: 0.4 }}
                animate={{ width: 260, height: 260, opacity: 0 }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeOut" }}
              />
            )}
          </AnimatePresence>

          {/* The Orb */}
          <div style={{ width: 200, height: 200, borderRadius: "50%", overflow: "hidden" }}>
            <Orb
              key={isDark ? "dark" : "light"}
              hue={orbHue}
              hoverIntensity={orbHoverIntensity}
              rotateOnHover={false}
              forceHoverState={orbForceHover}
              backgroundColor={isDark ? "#07080f" : "#f5f4fb"}
              bgLuminanceOverride={isDark ? -1 : 1}
              satBoost={isDark ? 0.0 : 0.75}
            />
          </div>
        </div>

        {/* Agent name + state badge */}
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-foreground/90 font-semibold text-xl tracking-tight">Bimlo Copilot</h1>
          <motion.div
            key={callState}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className="flex items-center gap-2"
          >
            <span className={`text-xs font-medium px-3 py-1 rounded-full border ${
              callState === "speaking"
                ? isDark
                  ? "text-blue-300 border-blue-500/25 bg-blue-500/10"
                  : "text-blue-600 border-blue-400/40 bg-blue-100/70"
                : callState === "thinking" || callState === "transcribing"
                ? isDark
                  ? "text-amber-300 border-amber-500/25 bg-amber-500/10"
                  : "text-amber-700 border-amber-400/40 bg-amber-100/70"
                : isActive
                ? isDark
                  ? "text-teal-300 border-teal-500/25 bg-teal-500/10"
                  : "text-teal-700 border-teal-500/35 bg-teal-100/70"
                : isDark
                  ? "text-foreground/30 border-foreground/8 bg-foreground/5"
                  : "text-foreground/50 border-foreground/15 bg-foreground/5"
            }`}>
              {STATE_LABEL[callState]}
            </span>
          </motion.div>
        </div>

        {/* ── Subtitle / transcript display ── */}
        <div className="w-full min-h-[3rem] flex items-center justify-center px-2">
          <AnimatePresence mode="wait">

            {(callState === "listening" || callState === "detecting") && (
              <motion.div key="soundwave"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex items-end gap-[3px]"
                style={{ height: 28 }}
              >
                {Array.from({ length: 20 }, (_, i) => (
                  <span
                    key={i}
                    ref={el => { waveBarRefsRef.current[i] = el; }}
                    className={`rounded-full ${callState === "detecting" ? "bg-teal-400/70" : "bg-teal-400/40"}`}
                    style={{ width: 3, height: 3, display: "block", transition: "height 0.08s linear" }}
                  />
                ))}
              </motion.div>
            )}

            {(callState === "transcribing" || callState === "thinking") && liveTranscript && (
              <motion.p key="transcript"
                initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.3 }}
                className="text-sm font-medium text-foreground/80 text-center leading-relaxed"
              >
                {liveTranscript}
              </motion.p>
            )}

            {callState === "thinking" && !liveTranscript && thinkingMessage && (
              <motion.p key={thinkingMessage}
                initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.25 }}
                className="text-xs text-amber-300/50 text-center italic"
              >
                {thinkingMessage}
              </motion.p>
            )}

            {callState === "speaking" && (
              <div key="subtitles" className="w-full flex items-center justify-center">
                <AnimatePresence mode="wait">
                  <motion.p
                    key={subtitleKey}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8, transition: { duration: 0.2 } }}
                    transition={{ duration: 0.2 }}
                    className="text-sm font-light text-blue-400/80 text-center leading-relaxed px-2"
                  >
                    <AnimatePresence mode="popLayout">
                      {subtitleWords.map((word, i) => (
                        <motion.span
                          key={`${subtitleKey}-${i}`}
                          initial={{ opacity: 0, y: 4, filter: "blur(4px)" }}
                          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                          transition={{ duration: 0.16 }}
                          className="inline-block mr-[0.28em]"
                        >{word}</motion.span>
                      ))}
                    </AnimatePresence>
                  </motion.p>
                </AnimatePresence>
              </div>
            )}

          </AnimatePresence>
        </div>

        {error && (
          <motion.p
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="text-xs text-red-400/70 text-center px-6 py-2 rounded-xl bg-red-500/5 border border-red-500/10"
          >
            {error}
          </motion.p>
        )}
      </div>

      {/* ── Controls ── */}
      <div className="relative z-10 w-full max-w-xs flex flex-col items-center pb-2">

        {/* CTA + aux buttons share the same fixed-height row — no layout shift */}
        <div className="relative flex items-center justify-center w-full h-20">

          {/* Mute — springs out to the LEFT of the CTA dot */}
          <AnimatePresence>
            {isActive && (
              <motion.button
                onClick={() => setMuted(m => !m)}
                initial={{ opacity: 0, x: 0, scale: 0.6 }}
                animate={{ opacity: 1, x: -112, scale: 1 }}
                exit={{ opacity: 0, x: 0, scale: 0.6 }}
                transition={{ type: "spring", stiffness: 320, damping: 28 }}
                className={`absolute flex flex-col items-center gap-1 px-4 py-2.5 rounded-2xl border text-xs font-medium transition-colors ${
                  muted
                    ? "bg-red-500/15 text-red-400 border-red-500/25"
                    : "bg-foreground/4 text-foreground/40 border-foreground/8 hover:bg-foreground/8 hover:text-foreground/60"
                }`}
              >
                {muted ? <MicOff className="h-4 w-4 mb-0.5" /> : <Mic className="h-4 w-4 mb-0.5" />}
                {muted ? "Unmute" : "Mute"}
              </motion.button>
            )}
          </AnimatePresence>

          {/* Primary CTA — always centred, never moves */}
          {callState === "idle" || callState === "ended" ? (
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={startCall}
              className="relative w-20 h-20 rounded-full flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, #1d6cf6 0%, #7c3aed 100%)",
                boxShadow: "0 0 40px rgba(99,102,241,0.35), 0 4px 20px rgba(0,0,0,0.5)"
              }}
            >
              <Mic className="h-7 w-7 text-white" />
            </motion.button>
          ) : (
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              onClick={endCall}
              className="relative w-20 h-20 rounded-full flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, #dc2626 0%, #9f1239 100%)",
                boxShadow: "0 0 32px rgba(220,38,38,0.3), 0 4px 20px rgba(0,0,0,0.5)"
              }}
            >
              <PhoneOff className="h-7 w-7 text-white" />
            </motion.button>
          )}

          {/* Speaker — springs out to the RIGHT of the CTA dot */}
          <AnimatePresence>
            {isActive && (
              <motion.button
                onClick={() => setSpeakerOff(s => !s)}
                initial={{ opacity: 0, x: 0, scale: 0.6 }}
                animate={{ opacity: 1, x: 112, scale: 1 }}
                exit={{ opacity: 0, x: 0, scale: 0.6 }}
                transition={{ type: "spring", stiffness: 320, damping: 28, delay: 0.04 }}
                className={`absolute flex flex-col items-center gap-1 px-4 py-2.5 rounded-2xl border text-xs font-medium transition-colors ${
                  speakerOff
                    ? "bg-foreground/5 text-foreground/25 border-foreground/5"
                    : "bg-foreground/4 text-foreground/40 border-foreground/8 hover:bg-foreground/8 hover:text-foreground/60"
                }`}
              >
                {speakerOff ? <VolumeX className="h-4 w-4 mb-0.5" /> : <Volume2 className="h-4 w-4 mb-0.5" />}
                {speakerOff ? "Off" : "Speaker"}
              </motion.button>
            )}
          </AnimatePresence>
        </div>

        <p className="text-foreground/55 text-[13px] text-center mt-4 tracking-wide">
          {callState === "idle" || callState === "ended"
            ? "Tap to start · speak naturally"
            : "Tap to end call"}
        </p>

        {/* ── Voice picker — bottom of page, never overlaps orb ── */}
        <div className="relative flex flex-col items-center mt-5">
          <button
            onClick={() => setShowVoicePicker(v => !v)}
            className="flex items-center gap-1.5 text-xs text-foreground/30 hover:text-foreground/60 transition-colors capitalize tracking-wide"
          >
            <span>Voice · {selectedVoice}</span>
            <ChevronDown className={`h-3 w-3 transition-transform duration-200 ${showVoicePicker ? "rotate-180" : ""}`} />
          </button>
          <AnimatePresence>
            {showVoicePicker && (
              <motion.div
                initial={{ opacity: 0, y: 6, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 6, scale: 0.96 }}
                transition={{ duration: 0.15 }}
                className="absolute bottom-8 bg-background/95 backdrop-blur-xl border border-border rounded-2xl shadow-2xl z-50 overflow-hidden w-48"
              >
                {[
                  { id: "hannah", label: "Hannah", hint: "Warm, natural" },
                  { id: "diana",  label: "Diana",  hint: "Clear, professional" },
                  { id: "autumn", label: "Autumn", hint: "Soft, calm" },
                  { id: "austin", label: "Austin", hint: "Male, friendly" },
                  { id: "daniel", label: "Daniel", hint: "Male, clear" },
                  { id: "troy",   label: "Troy",   hint: "Male, deep" },
                ].map(v => (
                  <button
                    key={v.id}
                    onClick={() => { setSelectedVoice(v.id); setShowVoicePicker(false); }}
                    className={`w-full text-left px-3.5 py-2.5 text-xs transition-colors ${
                      selectedVoice === v.id
                        ? "bg-blue-500/15 text-blue-400"
                        : "text-foreground/50 hover:bg-foreground/5 hover:text-foreground/80"
                    }`}
                  >
                    <span className="font-medium">{v.label}</span>
                    <span className="ml-2 text-foreground/25">{v.hint}</span>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
};

// Tiny UUID v4 (no dependency)
function uuid4(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export default CallPage;