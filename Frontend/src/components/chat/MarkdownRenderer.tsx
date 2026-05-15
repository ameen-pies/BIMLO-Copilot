import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      className={className}
    >
      {content}
    </ReactMarkdown>
  );
};
