import React, { useState, useRef } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface VoiceRecorderProps {
  onTranscription: (text: string) => void;
  disabled?: boolean;
}

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ onTranscription, disabled }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setIsProcessing(true);
        try {
          const formData = new FormData();
          formData.append("file", blob, "recording.webm");
          const res = await fetch(`${import.meta.env.VITE_API_URL || "http://localhost:8000"}/voice/transcribe`, {
            method: "POST",
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            onTranscription(data.text || "");
          }
        } catch (err) {
          console.error("Transcription failed:", err);
        } finally {
          setIsProcessing(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  return (
    <Button
      variant={isRecording ? "destructive" : "ghost"}
      size="icon"
      onClick={isRecording ? stopRecording : startRecording}
      disabled={disabled || isProcessing}
      title={isRecording ? "Stop recording" : "Start voice recording"}
    >
      {isProcessing ? <Loader2 size={18} className="animate-spin" /> : isRecording ? <Square size={18} /> : <Mic size={18} />}
    </Button>
  );
};
