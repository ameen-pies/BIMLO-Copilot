/**
 * CallPage.tsx
 *
 * A phone-call–style conversation page that chains:
 *   mic → Whisper (/transcribe) → RAG agents (/query-stream) → Groq TTS (/tts) → speaker
 *
 * Key behaviours
 * ──────────────
 * • Voice Activity Detection (VAD) — energy + zero-crossing + sustained-duration gate.
 *   Rejects coughs, clicks, background noise; only fires on real speech.
 * • Barge-in — if you speak while the AI is talking, audio stops within ~150 ms and
 *   the system re-enters LISTENING immediately.
 * • Hands-free loop — after TTS finishes (or is interrupted) it auto-returns to LISTENING.
 * • Session memory — shares the same session_id as the main chat so history is continuous.
 * • All agents available — the /query-stream router picks RAG / report / graph etc.
 */

import React, {
  useState, useEffect, useRef, useCallback,
} from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { PhoneOff, Mic, MicOff, Volume2, VolumeX, ChevronDown } from "lucide-react";

// ── API base (same helper as Chat.tsx) ───────────────────────────────────────
const API =
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
  "http://localhost:8000";

// ── VAD tuning ────────────────────────────────────────────────────────────────
const VAD_ENERGY_THRESHOLD    = 0.012;  // RMS energy gate (0-1); raise if noisy room
const VAD_ZCR_MAX             = 0.45;   // zero-crossing rate ceiling; pure noise is high-ZCR
const VAD_HOLD_MS             = 900;    // silence after speech before we cut (ms)
const VAD_MIN_SPEECH_MS       = 350;    // minimum voiced duration to count as real speech
const VAD_BARGE_IN_ENERGY     = 0.018;  // slightly higher bar for barge-in detection
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
  const sessionRef = useRef<string>(
    // Try to inherit the chat session; otherwise create a fresh call session
    localStorage.getItem("bimlo_call_session") || `call-${uuid4()}`
  );

  // ── State ──────────────────────────────────────────────────────────────────
  const [callState, setCallState]   = useState<CallState>("idle");
  const [turns, setTurns]           = useState<Turn[]>([]);
  const [muted, setMuted]           = useState(false);
  const [speakerOff, setSpeakerOff] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState("hannah");
  const [showVoicePicker, setShowVoicePicker] = useState(false);
  const [liveTranscript, setLiveTranscript]   = useState("");
  const [waveform, setWaveform]     = useState<number[]>(Array(32).fill(0));
  const [callDuration, setCallDuration] = useState(0);
  const [error, setError]           = useState<string | null>(null);

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

  const durationTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const turnsEndRef       = useRef<HTMLDivElement>(null);

  // Keep refs in sync with state
  useEffect(() => { stateRef.current   = callState; }, [callState]);
  useEffect(() => { mutedRef.current   = muted;     }, [muted]);
  useEffect(() => { speakerOffRef.current = speakerOff; }, [speakerOff]);

  // Auto-scroll transcript
  useEffect(() => {
    turnsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, liveTranscript]);

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

    const tick = () => {
      animFrameRef.current = requestAnimationFrame(tick);

      const st = stateRef.current;
      if (st !== "listening" && st !== "detecting") return;

      analyser.getFloatTimeDomainData(buf);
      const energy = rms(buf);
      const rate   = zcr(buf);

      // ── Waveform visualisation ────────────────────────────────────────────
      const bars = Array.from({ length: 32 }, (_, i) => {
        const start = Math.floor((i / 32) * bufSize);
        const end   = Math.floor(((i + 1) / 32) * bufSize);
        const slice = buf.slice(start, end);
        return Math.min(1, rms(slice) * 8);
      });
      setWaveform(bars);

      if (mutedRef.current) return;

      const isSpeech = energy > VAD_ENERGY_THRESHOLD && rate < VAD_ZCR_MAX;

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

  // ── Stop TTS immediately ───────────────────────────────────────────────────
  const stopTTS = useCallback(() => {
    ttsAbortRef.current?.abort();
    ttsAbortRef.current = null;
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      ttsAudioRef.current.src = "";
      ttsAudioRef.current = null;
    }
  }, []);

  // ── Barge-in handler ───────────────────────────────────────────────────────
  const handleBargeIn = useCallback(() => {
    stopTTS();
    setState("detecting");
    // Re-arm recorder for new utterance
    chunksRef.current = [];
    startVAD();
  }, [stopTTS, startVAD]);

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
    if (blob.size < 500) {
      // Too small — restart listening
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

      const txRes = await fetch(`${API}/transcribe`, {
        method: "POST",
        body:   form,
      });
      if (!txRes.ok) throw new Error("Transcription failed");

      const { transcript } = (await txRes.json()) as { transcript: string };
      const text = transcript?.trim();

      if (!text || text.length < 2) {
        // Nothing heard — go back to listening
        startListening();
        return;
      }

      setLiveTranscript(text);
      addTurn("user", text);

      // ── 2. Query RAG ───────────────────────────────────────────────────
      setState("thinking");

      const qRes = await fetch(`${API}/query-stream`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          query:      `[Voice call — user spoke this aloud]\n\n${text}`,
          session_id: sessionRef.current,
          top_k:      5,
        }),
      });
      if (!qRes.ok) throw new Error("Query failed");

      let answer = "";
      const reader  = qRes.body!.getReader();
      const decoder = new TextDecoder();
      let   buf     = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "result") answer = ev.answer ?? "";
          } catch { /* skip malformed */ }
        }
      }

      answer = answer.trim();
      if (!answer) {
        startListening();
        return;
      }

      addTurn("assistant", answer);

      // ── 3. TTS ─────────────────────────────────────────────────────────
      if (speakerOffRef.current) {
        // Speaker muted — skip TTS, go straight back to listening
        startListening();
        return;
      }

      await speakText(answer);

    } catch (err) {
      console.error("[CallPage] processBlob error:", err);
      setError(String(err));
      startListening();
    }
  }, [selectedVoice]);

  // ── TTS playback ───────────────────────────────────────────────────────────
  const speakText = useCallback(async (text: string) => {
    setState("speaking");

    const clean = stripMarkdown(text);
    ttsAbortRef.current = new AbortController();

    try {
      const res = await fetch(`${API}/tts`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text: clean, voice: selectedVoice }),
        signal:  ttsAbortRef.current.signal,
      });

      if (!res.ok) throw new Error("TTS failed");

      const arrayBuf = await res.arrayBuffer();
      if (ttsAbortRef.current?.signal.aborted) return;

      const blob    = new Blob([arrayBuf], { type: "audio/mpeg" });
      const url     = URL.createObjectURL(blob);
      const audio   = new Audio(url);
      ttsAudioRef.current = audio;

      await new Promise<void>((resolve) => {
        audio.onended  = () => { URL.revokeObjectURL(url); resolve(); };
        audio.onerror  = () => { URL.revokeObjectURL(url); resolve(); };
        audio.play().catch(() => resolve());

        // While playing, run barge-in detector
        startBargeInDetector();
      });

    } catch (err: any) {
      if (err?.name === "AbortError") return; // barge-in — handled upstream
      console.warn("[CallPage] TTS error:", err);
    } finally {
      ttsAudioRef.current = null;
    }

    // TTS finished naturally — back to listening
    if (stateRef.current === "speaking") {
      startListening();
    }
  }, [selectedVoice, startBargeInDetector]);

  // ── Start (or restart) the listening state ─────────────────────────────────
  const startListening = useCallback(() => {
    if (stateRef.current === "ended") return;

    // Re-create the recorder so we get a fresh segment
    const stream = streamRef.current;
    if (!stream) return;

    chunksRef.current = [];

    const mime = mimeRef.current;
    const recorder = new MediaRecorder(stream, { mimeType: mime });
    recorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mime });
      chunksRef.current = [];
      processBlob(blob);
    };

    recorder.start(100); // collect in 100ms chunks for responsive barge-in
    setState("listening");
    startVAD();
  }, [processBlob, startVAD]);

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

      // Persist session
      localStorage.setItem("bimlo_call_session", sessionRef.current);

      startListening();

    } catch (err: any) {
      setState("idle");
      setError(err?.message ?? "Microphone access denied");
    }
  }, [startListening]);

  // ── End the call ───────────────────────────────────────────────────────────
  const endCall = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    stopTTS();

    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach(t => t.stop());
    audioCtxRef.current?.close().catch(() => {});

    streamRef.current  = null;
    audioCtxRef.current = null;
    analyserRef.current = null;
    recorderRef.current = null;

    setState("ended");
    setWaveform(Array(32).fill(0));
    setLiveTranscript("");
  }, [stopTTS]);

  // Cleanup on unmount
  useEffect(() => () => { endCall(); }, []);

  // ── Format duration ────────────────────────────────────────────────────────
  const fmt = (s: number) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // ── Render ─────────────────────────────────────────────────────────────────
  const isActive = !["idle", "ended", "connecting"].includes(callState);

  return (
    <div className="min-h-screen bg-[#0a0a0f] flex flex-col items-center justify-between px-4 py-8 select-none">

      {/* ── Top bar ── */}
      <div className="w-full max-w-sm flex items-center justify-between">
        <button
          onClick={() => navigate("/")}
          className="text-zinc-500 hover:text-zinc-300 text-xs transition-colors"
        >
          ← Back
        </button>
        <span className="text-zinc-600 text-xs font-mono">{fmt(callDuration)}</span>
        {/* Voice picker */}
        <div className="relative">
          <button
            onClick={() => setShowVoicePicker(v => !v)}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {selectedVoice.split("-")[0]}
            <ChevronDown className="h-3 w-3" />
          </button>
          <AnimatePresence>
            {showVoicePicker && (
              <motion.div
                initial={{ opacity: 0, y: -6, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.95 }}
                transition={{ duration: 0.15 }}
                className="absolute right-0 top-6 bg-zinc-900 border border-zinc-700/60 rounded-xl shadow-2xl z-50 overflow-hidden w-44"
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
                    className={`w-full text-left px-3 py-2 text-xs transition-colors ${
                      selectedVoice === v.id
                        ? "bg-primary/20 text-primary"
                        : "text-zinc-300 hover:bg-zinc-800"
                    }`}
                  >
                    <span className="font-medium">{v.label}</span>
                    <span className="ml-1.5 text-zinc-500">{v.hint}</span>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Avatar + status ── */}
      <div className="flex flex-col items-center gap-6 flex-1 justify-center w-full max-w-sm">

        {/* Outer pulse ring */}
        <div className="relative flex items-center justify-center">
          {/* Ambient ring — pulses while speaking */}
          <AnimatePresence>
            {callState === "speaking" && (
              <>
                <motion.div
                  key="ring1"
                  className="absolute rounded-full border border-blue-500/20"
                  initial={{ width: 100, height: 100, opacity: 0.6 }}
                  animate={{ width: 180, height: 180, opacity: 0 }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
                />
                <motion.div
                  key="ring2"
                  className="absolute rounded-full border border-blue-500/15"
                  initial={{ width: 100, height: 100, opacity: 0.4 }}
                  animate={{ width: 210, height: 210, opacity: 0 }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut", delay: 0.6 }}
                />
              </>
            )}
            {(callState === "listening" || callState === "detecting") && (
              <motion.div
                key="listen-ring"
                className="absolute rounded-full border border-emerald-500/25"
                initial={{ width: 100, height: 100, opacity: 0.5 }}
                animate={{ width: 150, height: 150, opacity: 0 }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
              />
            )}
          </AnimatePresence>

          {/* Avatar circle */}
          <motion.div
            className={`relative w-24 h-24 rounded-full flex items-center justify-center text-3xl
              ${callState === "speaking"
                ? "bg-gradient-to-br from-blue-600/40 to-violet-600/40 border-2 border-blue-500/50"
                : callState === "thinking" || callState === "transcribing"
                ? "bg-gradient-to-br from-amber-600/30 to-orange-600/30 border-2 border-amber-500/40"
                : isActive
                ? "bg-gradient-to-br from-emerald-600/30 to-teal-600/30 border-2 border-emerald-500/40"
                : "bg-zinc-800/60 border-2 border-zinc-700/60"
              }`}
            animate={callState === "thinking" ? { scale: [1, 1.04, 1] } : {}}
            transition={{ duration: 1.2, repeat: Infinity }}
          >
            🤖
          </motion.div>
        </div>

        {/* Agent name + status */}
        <div className="flex flex-col items-center gap-1.5">
          <h1 className="text-white font-semibold text-lg tracking-tight">Bimlo Copilot</h1>
          <motion.p
            key={callState}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className={`text-sm font-medium ${STATE_COLOR[callState]}`}
          >
            {STATE_LABEL[callState]}
          </motion.p>
        </div>

        {/* ── Live waveform bars (while listening/detecting) ── */}
        <AnimatePresence>
          {(callState === "listening" || callState === "detecting" || callState === "speaking") && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-[3px] h-10"
            >
              {waveform.map((v, i) => (
                <motion.div
                  key={i}
                  className={`w-[3px] rounded-full ${
                    callState === "speaking" ? "bg-blue-400/70" : "bg-emerald-400/70"
                  }`}
                  animate={{ height: Math.max(4, v * 36) }}
                  transition={{ duration: 0.05 }}
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Transcript feed ── */}
        <div className="w-full max-w-sm h-48 overflow-y-auto flex flex-col gap-2 px-1 scrollbar-thin scrollbar-thumb-zinc-700/40">
          {turns.map(turn => (
            <motion.div
              key={turn.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-[13px] leading-relaxed ${
                turn.role === "user"
                  ? "bg-primary/20 text-primary-foreground/90 rounded-br-sm"
                  : "bg-zinc-800/80 text-zinc-200 rounded-bl-sm"
              }`}>
                {turn.content}
              </div>
            </motion.div>
          ))}

          {/* Live transcript (user currently speaking) */}
          {liveTranscript && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-end"
            >
              <div className="max-w-[85%] rounded-2xl rounded-br-sm px-3.5 py-2 text-[13px] bg-primary/10 text-primary/60 border border-primary/20 italic">
                {liveTranscript}
              </div>
            </motion.div>
          )}

          {/* Thinking indicator */}
          {callState === "thinking" && (
            <div className="flex justify-start">
              <div className="bg-zinc-800/80 rounded-2xl rounded-bl-sm px-4 py-2.5 flex gap-1.5 items-center">
                {[0, 0.2, 0.4].map(d => (
                  <motion.div
                    key={d}
                    className="w-1.5 h-1.5 rounded-full bg-zinc-400"
                    animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
                    transition={{ duration: 0.9, repeat: Infinity, delay: d }}
                  />
                ))}
              </div>
            </div>
          )}

          <div ref={turnsEndRef} />
        </div>

        {/* Error */}
        {error && (
          <p className="text-xs text-red-400/80 text-center px-4">{error}</p>
        )}
      </div>

      {/* ── Controls ── */}
      <div className="w-full max-w-sm flex flex-col items-center gap-6">

        {/* Aux buttons (mute + speaker) — only while active */}
        {isActive && (
          <div className="flex items-center gap-6">
            <button
              onClick={() => setMuted(m => !m)}
              className={`flex flex-col items-center gap-1.5 p-3 rounded-2xl transition-all ${
                muted
                  ? "bg-red-500/20 text-red-400 border border-red-500/30"
                  : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-700/60 border border-zinc-700/40"
              }`}
            >
              {muted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
              <span className="text-[10px]">{muted ? "Unmute" : "Mute"}</span>
            </button>

            <button
              onClick={() => setSpeakerOff(s => !s)}
              className={`flex flex-col items-center gap-1.5 p-3 rounded-2xl transition-all ${
                speakerOff
                  ? "bg-zinc-700/60 text-zinc-500 border border-zinc-600/30"
                  : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-700/60 border border-zinc-700/40"
              }`}
            >
              {speakerOff ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
              <span className="text-[10px]">{speakerOff ? "Speaker off" : "Speaker"}</span>
            </button>
          </div>
        )}

        {/* Primary CTA */}
        {callState === "idle" || callState === "ended" ? (
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={startCall}
            className="w-20 h-20 rounded-full bg-emerald-500 hover:bg-emerald-400 text-white flex items-center justify-center shadow-lg shadow-emerald-500/30 transition-colors"
          >
            <Mic className="h-8 w-8" />
          </motion.button>
        ) : (
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={endCall}
            className="w-20 h-20 rounded-full bg-red-500 hover:bg-red-400 text-white flex items-center justify-center shadow-lg shadow-red-500/30 transition-colors"
          >
            <PhoneOff className="h-8 w-8" />
          </motion.button>
        )}

        <p className="text-zinc-600 text-[11px] text-center pb-2">
          {callState === "idle" || callState === "ended"
            ? "Tap to start a call"
            : "Tap to end call  ·  Speak naturally — I'll listen when you're done"}
        </p>
      </div>
    </div>
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