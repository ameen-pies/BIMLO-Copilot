import React, { useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { motion } from "framer-motion";
import { X, FileText, ScrollText, ImageIcon, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { XeokitViewer } from "./XeokitViewer";
import { ZoomableImage } from "./ZoomableImage";
import { PdfViewer } from "./PdfViewer";
import { renderDocumentContent, ViewerState } from "./chatRenderers";

interface DocumentViewerProps {
  state: ViewerState;
  onClose: () => void;
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({ state, onClose }) => {
  const { t } = useTranslation();
  const highlightRef = useRef<HTMLElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      if (!highlightRef.current || !scrollContainerRef.current) return;
      const container = scrollContainerRef.current;
      const el = highlightRef.current;
      const elTop = el.getBoundingClientRect().top - container.getBoundingClientRect().top;
      const targetScroll = container.scrollTop + elTop - container.clientHeight / 2 + el.offsetHeight / 2;
      container.scrollTo({ top: targetScroll, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(raf);
  }, [state.highlightKey, state.content]);

  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 480, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
      className="border-l border-border bg-background flex flex-col overflow-hidden shrink-0 h-screen"
    >
      <div className="h-14 border-b border-border flex items-center gap-3 px-4 shrink-0">
        {state.mediaType === 'image' ? <ImageIcon className="h-4 w-4 text-primary shrink-0" /> : state.mediaType === 'cad' ? <FileText className="h-4 w-4 text-primary shrink-0" /> : <ScrollText className="h-4 w-4 text-primary shrink-0" />}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-foreground truncate">{state.doc.filename}</p>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{state.doc.doc_type}</p>
        </div>
        <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground shrink-0" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {state.highlightText && state.mediaType !== 'image' && (
        <div className="px-4 py-2 bg-primary/8 border-b border-primary/20 flex items-start gap-2">
          <span className="text-[10px] font-semibold text-primary uppercase tracking-wide shrink-0 mt-0.5">{t("chat.cited_passage")}</span>
          <p className="text-[11px] text-primary/80 leading-relaxed line-clamp-2 flex-1">{state.highlightText}</p>
        </div>
      )}

      <div ref={scrollContainerRef} className={`flex-1 overflow-hidden scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40 ${state.mediaType === 'pdf' ? 'p-0' : state.mediaType === 'image' ? 'p-2 overflow-y-auto' : 'p-4 overflow-y-auto'}`}>
        {state.loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm">{t("chat.loading_document")}</p>
          </div>
        )}

        {state.error && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
            <AlertCircle className="h-8 w-8 text-destructive/70" />
            <p className="text-sm text-muted-foreground">{state.error}</p>
          </div>
        )}

        {!state.loading && !state.error && (
          state.mediaType === 'cad' ? (
            <div className="p-4 space-y-4 overflow-y-auto h-full">
              {state.cadSummary && (
                (() => {
                  const ps = state.cadSummary as any;
                  const pl = (ps.pipeline ?? '').toLowerCase();
                  const fn = (state.doc?.filename ?? ps.filename ?? 'model.ifc');
                  const ext = fn.split('.').pop()?.toLowerCase() ?? '';

                  const viewerBlobUrl = state.ifcBlobUrl ?? (pl === 'ifc' ? state.blobUrl : null);
                  const viewerFilename = state.ifcBlobUrl
                    ? fn.replace(/\.(dxf|dwg|step|stp|rvt|nwd|nwc|dgn|skp|fbx|obj|stl|sat|iges|igs|prt|sldprt|catpart|3ds|dae|rfa|rte)$/i, '.ifc') || 'model.ifc'
                    : fn;

                  if (viewerBlobUrl) {
                    return (
                      <XeokitViewer
                        blobUrl={viewerBlobUrl}
                        filename={viewerFilename}
                        pipeline={pl}
                      />
                    );
                  }

                  if (ext === 'dxf' && state.blobUrl) {
                    return (
                      <XeokitViewer
                        blobUrl={state.blobUrl}
                        filename={fn}
                        pipeline={pl}
                      />
                    );
                  }

                  return (
                    <div className="w-full rounded-lg bg-muted/30 border border-border flex flex-col items-center justify-center gap-2 p-6 text-center" style={{ height: 200 }}>
                      <svg className="h-8 w-8 text-muted-foreground/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" /></svg>
                      <p className="text-xs text-muted-foreground/60">3D preview not available for this format</p>
                      <p className="text-[10px] text-muted-foreground/40">AI analysis works via chat ↓</p>
                    </div>
                  );
                })()
              )}
              {state.cadSummary ? (() => {
                const s = state.cadSummary as any;
                const pipeline = (s.pipeline ?? 'cad').toUpperCase();
                const isIfc = pipeline === 'IFC';
                return (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-primary/15 text-primary border border-primary/30">
                        {pipeline}
                      </span>
                      {s.schema && <span className="text-[10px] text-muted-foreground">Schema: {s.schema}</span>}
                      {s.dimension && <span className="text-[10px] text-muted-foreground">{s.dimension}</span>}
                    </div>

                    {s.ux_hint && (
                      <div className="bg-primary/8 border border-primary/20 rounded-lg px-3 py-2">
                        <p className="text-xs text-primary/80 leading-relaxed">{s.ux_hint}</p>
                      </div>
                    )}

                    {s.parse_errors && s.parse_errors.length > 0 && (
                      <div className="bg-destructive/8 border border-destructive/20 rounded-lg px-3 py-2">
                        <p className="text-[10px] font-semibold text-destructive uppercase tracking-wide mb-1">Parse Warnings</p>
                        {s.parse_errors.map((e: string, i: number) => (
                          <p key={i} className="text-xs text-destructive/80">{e}</p>
                        ))}
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-2">
                      {s.total_elements != null && (
                        <div className="bg-muted/50 rounded-lg p-3">
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{t("chat.elements")}</p>
                          <p className="text-xl font-bold text-foreground">{s.total_elements.toLocaleString()}</p>
                        </div>
                      )}
                      {s.total_entities != null && (
                        <div className="bg-muted/50 rounded-lg p-3">
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{t("chat.entities")}</p>
                          <p className="text-xl font-bold text-foreground">{s.total_entities.toLocaleString()}</p>
                        </div>
                      )}
                    </div>

                    {isIfc && s.storeys && s.storeys.length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Building Storeys</p>
                        <div className="space-y-1">
                          {s.storeys.map((st: any, i: number) => (
                            <div key={i} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1.5 text-xs">
                              <span className="text-foreground font-medium">{st.name ?? `Storey ${i+1}`}</span>
                              {st.elevation != null && <span className="text-muted-foreground">{typeof st.elevation === 'number' ? st.elevation.toFixed(2) : st.elevation} m</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {s.element_counts && Object.keys(s.element_counts).length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Element Types</p>
                        <div className="space-y-1">
                          {Object.entries(s.element_counts as Record<string, number>)
                            .sort(([,a],[,b]) => b - a)
                            .map(([label, count]) => (
                              <div key={label} className="flex items-center gap-2">
                                <div className="flex-1 flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                                  <span className="text-foreground capitalize">{label}</span>
                                  <span className="text-primary font-semibold">{count}</span>
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {s.entity_counts && Object.keys(s.entity_counts).length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Entity Types</p>
                        <div className="space-y-1">
                          {Object.entries(s.entity_counts as Record<string, number>)
                            .sort(([,a],[,b]) => b - a)
                            .slice(0, 15)
                            .map(([label, count]) => (
                              <div key={label} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                                <span className="text-foreground">{label}</span>
                                <span className="text-primary font-semibold">{count}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {s.layers && s.layers.length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Layers ({s.layers.length})</p>
                        <div className="space-y-1 max-h-48 overflow-y-auto">
                          {s.layers.map((layer: any, i: number) => (
                            <div key={i} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                              <span className="text-foreground font-mono">{layer.name ?? layer}</span>
                              {layer.entity_count != null && <span className="text-muted-foreground">{layer.entity_count} entities</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {s.material_inventory && Object.keys(s.material_inventory).length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">{t("chat.materials")}</p>
                        <div className="space-y-1">
                          {Object.entries(s.material_inventory as Record<string, number>)
                            .sort(([,a],[,b]) => b - a)
                            .map(([mat, count]) => (
                              <div key={mat} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                                <span className="text-foreground">{mat}</span>
                                <span className="text-primary font-semibold">{count}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })() : (
                <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
                  <FileText className="h-10 w-10 text-muted-foreground/40" />
                  <p className="text-sm text-muted-foreground">No summary available for this file.</p>
                </div>
              )}
            </div>
          ) : state.mediaType === 'image' ? (
            <ZoomableImage src={state.blobUrl!} alt={state.doc.filename} />
          ) : state.mediaType === 'pdf' ? (
            state.blobUrl ? (
              <PdfViewer
                blobUrl={state.blobUrl}
                highlightText={state.highlightText}
                highlightLines={state.highlightLines}
                highlightKey={state.highlightKey}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
                <FileText className="h-10 w-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">PDF preview available after upload.</p>
                <p className="text-xs text-muted-foreground/60">The file could not be fetched from the server.</p>
              </div>
            )
          ) : state.content !== null ? (
            renderDocumentContent(state.content, state.highlightText, highlightRef, state.highlightKey, state.highlightLines)
          ) : null
        )}
      </div>
    </motion.aside>
  );
};
