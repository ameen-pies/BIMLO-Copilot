import { useState, useEffect, useRef } from "react";

interface TypewriterTextProps {
  text: string;
  speed?: number;
  onComplete?: () => void;
  render?: (partial: string) => React.ReactNode;
}

/**
 * Tokenize text so we never cut inside a markdown link [text](url),
 * a citation {{fact|Source N}}, bold/italic, or inline code.
 * These are emitted as single atomic tokens.
 */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  const re = /(\{\{[^}]*\}\}|\[[^\]]*\]\([^)]*\)|[*_]{1,2}[^*_\n]+[*_]{1,2}|`[^`]+`|[\s\S])/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    tokens.push(m[1]);
  }
  return tokens;
}

const TypewriterText = ({ text, speed = 20, onComplete, render }: TypewriterTextProps) => {
  const [displayedText, setDisplayedText] = useState("");
  const [tokenIndex, setTokenIndex] = useState(0);
  const tokensRef = useRef<string[]>([]);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    tokensRef.current = tokenize(text);
    setDisplayedText("");
    setTokenIndex(0);
  }, [text]);

  useEffect(() => {
    const tokens = tokensRef.current;
    if (tokenIndex < tokens.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + tokens[tokenIndex]);
        setTokenIndex((prev) => prev + 1);
      }, speed);
      return () => clearTimeout(timeout);
    } else if (tokenIndex > 0 && tokenIndex === tokens.length) {
      onCompleteRef.current?.();
    }
  }, [tokenIndex, speed]);

  if (render) {
    return <>{render(displayedText)}</>;
  }

  return <span>{displayedText}</span>;
};

export default TypewriterText;