import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText } from "lucide-react";
import { Document } from "@/services/api";

interface AutocompletePanelProps {
  autocomplete: {
    query: string;
    triggerPos: number;
    results: Document[];
    activeIdx: number;
  } | null;
  autocompletePos: { top: number; left: number; width: number } | null;
  autocompleteRef: React.RefObject<HTMLDivElement>;
  applyAutocomplete: (doc: Document) => void;
  setAutocomplete: React.Dispatch<
    React.SetStateAction<{
      query: string;
      triggerPos: number;
      results: Document[];
      activeIdx: number;
    } | null>
  >;
}

export function AutocompletePanel({
  autocomplete,
  autocompletePos,
  autocompleteRef,
  applyAutocomplete,
  setAutocomplete,
}: AutocompletePanelProps) {
  return (
    <>
      {/* ── Filename autocomplete — fixed portal, never clipped ── */}
      <AnimatePresence>
        {autocomplete && autocomplete.results.length > 0 && autocompletePos && (
          <motion.div
            ref={autocompleteRef}
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            style={{
              position: "fixed",
              bottom: `calc(100vh - ${autocompletePos.top}px + 8px)`,
              left: autocompletePos.left,
              width: autocompletePos.width,
              zIndex: 9999,
            }}
            className="bg-card border border-border rounded-xl shadow-2xl overflow-hidden"
          >
            <div className="px-3 py-1.5 border-b border-border/60 flex items-center gap-1.5">
              <FileText className="h-3 w-3 text-primary/60" />
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
                Documents
              </span>
              <span className="text-[10px] text-muted-foreground/50 ml-auto">
                ↑↓ · Tab to select
              </span>
            </div>
            {autocomplete.results.map((doc, idx) => (
              <button
                key={doc.document_id}
                onMouseDown={(e) => {
                  e.preventDefault();
                  applyAutocomplete(doc);
                }}
                onMouseEnter={() =>
                  setAutocomplete(a => a ? { ...a, activeIdx: idx } : a)
                }
                className={`w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors ${
                  idx === autocomplete.activeIdx
                    ? "bg-primary/10 text-foreground"
                    : "hover:bg-muted text-foreground/80"
                }`}
              >
                <FileText className={`h-3.5 w-3.5 shrink-0 ${idx === autocomplete.activeIdx ? "text-primary" : "text-muted-foreground"}`} />
                <span className="text-sm flex-1 truncate">{doc.filename}</span>
                <span className="text-[10px] text-muted-foreground/50 shrink-0 capitalize">
                  {(doc.doc_type ?? "").replace(/_/g, " ")}
                </span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
