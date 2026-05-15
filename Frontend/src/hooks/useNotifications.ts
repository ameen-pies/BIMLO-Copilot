import { useState, useRef, useEffect, useCallback } from "react";

export function useNotifications() {
  const [notifyEnabled, setNotifyEnabled] = useState(false);
  const [showNotifyBanner, setShowNotifyBanner] = useState(false);
  const [notifyDismissed, setNotifyDismissed] = useState(false);
  const notifyEnabledRef = useRef(false);
  useEffect(() => { notifyEnabledRef.current = notifyEnabled; }, [notifyEnabled]);

  const beepAudioRef = useRef<HTMLAudioElement | null>(null);
  const audioUnlocked = useRef(false);

  const ensureAudio = useCallback(() => {
    if (beepAudioRef.current) return beepAudioRef.current;
    const audio = new Audio("/beep.wav");
    audio.preload = "auto";
    beepAudioRef.current = audio;
    if (!audioUnlocked.current) {
      audio.volume = 0;
      audio.play().then(() => {
        audio.pause();
        audio.currentTime = 0;
        audio.volume = 1;
        audioUnlocked.current = true;
      }).catch(() => {});
    }
    return audio;
  }, []);

  const playBeep = useCallback(() => {
    const audio = ensureAudio();
    audio.currentTime = 0;
    audio.volume = 1;
    audio.play().catch(() => {});
  }, [ensureAudio]);

  const fireNotification = useCallback(() => {
    if (!notifyEnabledRef.current) return;
    playBeep();
    if (document.visibilityState !== "visible" && Notification.permission === "granted") {
      new Notification("Done!", { body: "Your answer is ready.", icon: "/favicon.svg" });
    }
  }, [playBeep]);

  return {
    notifyEnabled, setNotifyEnabled,
    showNotifyBanner, setShowNotifyBanner,
    notifyDismissed, setNotifyDismissed,
    ensureAudio,
    playBeep,
    fireNotification,
  };
}
