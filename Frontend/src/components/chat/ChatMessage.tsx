import React from "react";
import { motion } from "framer-motion";
import { User, Bot, Copy, Check, ThumbsUp, ThumbsDown } from "lucide-react";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  onCopy?: () => void;
  onFeedback?: (positive: boolean) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  isStreaming,
  onCopy,
  onFeedback,
}) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    onCopy?.();
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 ${role === "user" ? "flex-row-reverse" : ""}`}
    >
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm ${
        role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
      }`}>
        {role === "user" ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
        role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
      }`}>
        {role === "assistant" ? (
          <MarkdownRenderer content={content} />
        ) : (
          <p className="text-sm whitespace-pre-wrap">{content}</p>
        )}
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-current animate-pulse ms-1" />
        )}
        {role === "assistant" && !isStreaming && (
          <div className="flex gap-2 mt-2 pt-2 border-t border-border/50 opacity-0 group-hover:opacity-100 transition-opacity">
            <button onClick={handleCopy} className="text-muted-foreground hover:text-foreground">
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
            {onFeedback && (
              <>
                <button onClick={() => onFeedback(true)} className="text-muted-foreground hover:text-green-500">
                  <ThumbsUp size={14} />
                </button>
                <button onClick={() => onFeedback(false)} className="text-muted-foreground hover:text-red-500">
                  <ThumbsDown size={14} />
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
};
