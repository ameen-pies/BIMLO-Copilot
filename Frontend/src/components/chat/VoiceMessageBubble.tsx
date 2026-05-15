import React, { useState, useRef, useMemo, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, ChevronDown, Phone } from "lucide-react";

interface VoiceMessageBubbleProps {
  blobUrl?: string;
  duration: number;
  transcript: string | undefined;
  isExpanded: boolean;
  onToggleTranscript: () => void;
  waveform?: number[];
  timestamp: Date;
}

export const VoiceMessageBubble: React.FC<VoiceMessageBubbleProps> = ({
  blobUrl,
  duration,
  transcript,
  isExpanded,
  onToggleTranscript,
  waveform,
  timestamp,
}) => {
  const { t } = useTranslation();
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeed] = useState<0.5 | 1 | 2 | 4>(1);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const animFrameRef = useRef<number>(0);

  const SPEEDS: Array<0.5 | 1 | 2 | 4> = [0.5, 1, 2, 4];
  const cycleSpeed = () => {
    const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length] as 0.5 | 1 | 2 | 4;
    setSpeed(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };

  const BAR_COUNT = 52;
  const bubbleBars = useMemo(() => {
    if (!waveform || waveform.length === 0) {
      return Array.from({ length: BAR_COUNT }, (_, i) => {
        const t = i / (BAR_COUNT - 1);
        return 0.15 + 0.45 * Math.sin(t * Math.PI) + 0.1 * Math.sin(t * Math.PI * 4);
      });
    }
    return Array.from({ length: BAR_COUNT }, (_, i) => {
      const start = Math.floor((i / BAR_COUNT) * waveform.length);
      const end   = Math.ceil(((i + 1) / BAR_COUNT) * waveform.length);
      const slice = waveform.slice(start, end);
      return slice.length > 0 ? Math.max(...slice) : 0;
    });
  }, [waveform]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(() => {});
    }
  };

  useEffect(() => {
    if (!blobUrl) {
      audioRef.current = null;
      setIsPlaying(false);
      setProgress(0);
      setCurrentTime(0);
      return;
    }

    const audio = new Audio(blobUrl);
    audio.playbackRate = speed;
    audioRef.current = audio;

    const tick = () => {
      if (!audioRef.current) return;
      const dur = audioRef.current.duration || duration;
      const ct  = audioRef.current.currentTime;
      setCurrentTime(ct);
      setProgress(dur > 0 ? ct / dur : 0);
      animFrameRef.current = requestAnimationFrame(tick);
    };

    const onPlay  = () => { setIsPlaying(true); tick(); };
    const onPause = () => { setIsPlaying(false); cancelAnimationFrame(animFrameRef.current); };
    const onEnded = () => { setIsPlaying(false); setProgress(0); setCurrentTime(0); cancelAnimationFrame(animFrameRef.current); };

    audio.addEventListener("play",  onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      audio.removeEventListener("play",  onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.pause();
      audio.src = "";
    };
  }, [blobUrl, speed, duration]);

  const isPending = transcript === undefined;
  const missingAudio = !blobUrl && transcript !== undefined;

  const waveformRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);

  const seekTo = (clientX: number) => {
    const el = waveformRef.current;
    const audio = audioRef.current;
    if (!el || !audio) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const dur = audio.duration || duration;
    if (isFinite(dur) && dur > 0) {
      audio.currentTime = frac * dur;
      setProgress(frac);
      setCurrentTime(frac * dur);
    }
  };

  const handleWaveMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    seekTo(e.clientX);
    const onMove = (ev: MouseEvent) => { if (isDraggingRef.current) seekTo(ev.clientX); };
    const onUp   = () => { isDraggingRef.current = false; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <div className="flex flex-col items-end gap-1 w-fit">
      <AnimatePresence>
        {isExpanded && transcript && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden w-full"
          >
            <p className="text-[11px] text-muted-foreground/60 leading-relaxed pl-3 border-l border-border/25 italic text-right">
              {transcript}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        onClick={onToggleTranscript}
        className="flex items-center gap-1.5 group/transcript w-fit pr-1"
        disabled={isPending}
      >
        {isPending ? (
          <span className="text-[11px] text-muted-foreground/40 italic leading-none flex items-center gap-1">
            <Loader2 className="h-2.5 w-2.5 animate-spin" />
            Transcribing…
          </span>
        ) : (
          <span className="text-[11px] text-muted-foreground/40 italic leading-none">
            {isExpanded ? "Hide transcript" : "Show transcript"}
          </span>
        )}
        {!isPending && (
          <motion.span
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="opacity-0 group-hover/transcript:opacity-60 transition-opacity"
          >
            <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
          </motion.span>
        )}
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-muted-foreground/30 shrink-0" />
      </button>

      <div className="bg-primary text-primary-foreground rounded-2xl rounded-br-md px-3 py-2.5 flex items-center gap-2.5 w-fit">
        <button
          onClick={togglePlay}
          disabled={!blobUrl}
          className={`h-8 w-8 rounded-full flex items-center justify-center shrink-0 transition-colors ${blobUrl ? "bg-primary-foreground/20 hover:bg-primary-foreground/30" : "bg-muted-foreground/10 cursor-not-allowed"}`}
        >
          {isPlaying ? (
            <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor">
              <rect x="0" y="0" width="4" height="14" rx="1.5" />
              <rect x="8" y="0" width="4" height="14" rx="1.5" />
            </svg>
          ) : (
            <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor">
              <path d="M1 1.5v11l10-5.5L1 1.5z" />
            </svg>
          )}
        </button>

        <div className="flex flex-col gap-1" style={{ width: "160px" }}>
          <div
            ref={waveformRef}
            onMouseDown={handleWaveMouseDown}
            className="flex items-center gap-[1.5px] h-[22px] cursor-pointer select-none"
            title={t("chat.click_drag_seek")}
          >
            {bubbleBars.map((h, i) => {
              const frac = i / (bubbleBars.length - 1);
              const isPast = frac <= progress;
              const minH = 2;
              const maxH = 20;
              const barH = minH + h * (maxH - minH);
              return (
                <div
                  key={i}
                  style={{
                    height: `${barH}px`,
                    flex: "1 1 0",
                    borderRadius: "1px",
                    background: isPast ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.32)",
                    transition: "background 0.06s ease",
                  }}
                />
              );
            })}
          </div>
          <span className="text-[10px] text-primary-foreground/55 tabular-nums">
            {isPlaying ? formatTime(currentTime) : formatTime(duration)}
          </span>
        </div>

        <button
          onClick={cycleSpeed}
          className="shrink-0 h-7 min-w-[32px] px-1 rounded-md bg-primary-foreground/15 hover:bg-primary-foreground/25 transition-colors flex items-center justify-center"
          title={t("chat.change_speed")}
        >
          <span className="text-[11px] font-semibold text-primary-foreground/80 leading-none tabular-nums">
            {speed === 0.5 ? "×½" : speed === 1 ? "×1" : speed === 2 ? "×2" : "×4"}
          </span>
        </button>
      </div>

      {missingAudio ? (
        <span className="text-[10px] text-muted-foreground/50 italic">
          Audio unavailable after reload
        </span>
      ) : (
        <span className="text-[10px] text-muted-foreground/50 tabular-nums">
          {timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      )}
    </div>
  );
};
