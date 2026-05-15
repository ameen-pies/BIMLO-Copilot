import React, { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, AlertCircle } from "lucide-react";

declare global {
  interface Window {
    pdfjsLib: any;
    XKT_VERSION?: string;
  }
}

interface XeokitViewerProps {
  blobUrl: string;
  filename: string;
  pipeline: string;
}

let _xeokitPromise: Promise<{ Viewer: any; WebIFCLoaderPlugin: any; WebIFC: any }> | null = null;

function loadXeokit(): Promise<{ Viewer: any; WebIFCLoaderPlugin: any; WebIFC: any }> {
  if (_xeokitPromise) return _xeokitPromise;
  _xeokitPromise = (async () => {
    const [mod, WebIFC] = await Promise.all([
      import(/* @vite-ignore */ "https://esm.sh/@xeokit/xeokit-sdk@2.6.1"),
      import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/web-ifc-api.js"),
    ]);
    if (!mod.Viewer)             throw new Error("xeokit: Viewer not exported");
    if (!mod.WebIFCLoaderPlugin) throw new Error("xeokit: WebIFCLoaderPlugin not exported");
    return { Viewer: mod.Viewer, WebIFCLoaderPlugin: mod.WebIFCLoaderPlugin, WebIFC };
  })();
  _xeokitPromise.catch(() => { _xeokitPromise = null; });
  return _xeokitPromise;
}

export const XeokitViewer: React.FC<XeokitViewerProps> = ({ blobUrl, filename, pipeline }) => {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errMsg, setErrMsg] = useState("");

  useEffect(() => {
    let destroyed = false;

    async function init() {
      try {
        setStatus("loading");

        const { Viewer, WebIFCLoaderPlugin, WebIFC } = await loadXeokit();

        if (destroyed) return;
        if (!canvasRef.current) throw new Error("Canvas ref is null");

        const viewer = new Viewer({
          canvasElement: canvasRef.current,
          transparent: true,
        });
        viewerRef.current = viewer;

        viewer.camera.eye  = [10, 10, 10];
        viewer.camera.look = [0, 0, 0];
        viewer.camera.up   = [0, 1, 0];

        const ext = filename.split(".").pop()?.toLowerCase() ?? "";
        const isIfc = ["ifc", "ifczip"].includes(ext);

        if (isIfc) {
          const ifcAPI = new WebIFC.IfcAPI();
          ifcAPI.SetWasmPath("https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/");
          await ifcAPI.Init();

          if (destroyed) return;

          const loader = new WebIFCLoaderPlugin(viewer, {
            WebIFC,
            IfcAPI: ifcAPI,
          });
          const model = loader.load({ id: "model", src: blobUrl, edges: true, performance: true });
          model.on("loaded", () => {
            if (destroyed) return;
            viewer.cameraFlight.flyTo(model);
            setStatus("ready");
          });
          model.on("error", (e: any) => {
            if (!destroyed) { setErrMsg(typeof e === "string" ? e : (e?.message ?? "Model load error")); setStatus("error"); }
          });
        } else if (["dxf"].includes(ext)) {
          if (destroyed) return;
          const resp = await fetch(blobUrl);
          const text = await resp.text();
          if (destroyed) return;

          const lines: {x1:number,y1:number,x2:number,y2:number}[] = [];
          const entityRe = /^\s*0\s*\nLINE([\s\S]*?)(?=\n\s*0\s*\n)/gm;
          const getVal = (block: string, code: number) => {
            const m = new RegExp(`\n\s*${code}\s*\n\s*([\d.+\-eE]+)`).exec(block);
            return m ? parseFloat(m[1]) : 0;
          };
          let em: RegExpExecArray | null;
          while ((em = entityRe.exec(text)) !== null) {
            lines.push({ x1: getVal(em[1], 10), y1: getVal(em[1], 20), x2: getVal(em[1], 11), y2: getVal(em[1], 21) });
          }

          if (destroyed) return;
          const canvas = canvasRef.current!;
          const ctx = canvas.getContext("2d")!;
          canvas.width  = canvas.offsetWidth  || 600;
          canvas.height = canvas.offsetHeight || 320;

          if (lines.length === 0) {
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#6b7280";
            ctx.font = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("DXF loaded — no LINE entities to preview", canvas.width/2, canvas.height/2);
            ctx.fillText("AI analysis works via chat below", canvas.width/2, canvas.height/2 + 20);
            setStatus("ready");
            return;
          }

          const minX = Math.min(...lines.map(l=>Math.min(l.x1,l.x2)));
          const maxX = Math.max(...lines.map(l=>Math.max(l.x1,l.x2)));
          const minY = Math.min(...lines.map(l=>Math.min(l.y1,l.y2)));
          const maxY = Math.max(...lines.map(l=>Math.max(l.y1,l.y2)));
          const pad = 24;
          const scaleX = (canvas.width  - pad*2) / (maxX - minX || 1);
          const scaleY = (canvas.height - pad*2) / (maxY - minY || 1);
          const scale  = Math.min(scaleX, scaleY);
          const tx = (x: number) => pad + (x - minX) * scale;
          const ty = (y: number) => canvas.height - pad - (y - minY) * scale;

          ctx.fillStyle = "#0f0f1a";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = "#60a5fa";
          ctx.lineWidth = 0.7;
          ctx.globalAlpha = 0.8;
          ctx.beginPath();
          for (const l of lines) {
            ctx.moveTo(tx(l.x1), ty(l.y1));
            ctx.lineTo(tx(l.x2), ty(l.y2));
          }
          ctx.stroke();
          if (!destroyed) setStatus("ready");
        } else {
          throw new Error(`.${ext} files: convert to IFC for 3D view, or DXF for 2D preview. AI analysis works via chat.`);
        }
      } catch (e: any) {
        if (!destroyed) { setErrMsg(e?.message ?? String(e)); setStatus("error"); }
      }
    }

    init();

    return () => {
      destroyed = true;
      if (viewerRef.current) {
        try { viewerRef.current.destroy(); } catch {}
        viewerRef.current = null;
      }
    };
  }, [blobUrl, filename]);

  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const block = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", block, { passive: false });
    return () => el.removeEventListener("wheel", block);
  }, []);

  return (
    <div ref={containerRef} className="relative w-full bg-black/80 rounded-lg overflow-hidden" style={{ height: 320 }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        style={{ touchAction: "none" }}
      />
      {status === "loading" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/60">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <p className="text-xs text-white/70">{t("chat.loading_3d_model")}</p>
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/70 p-4 text-center">
          <AlertCircle className="h-6 w-6 text-destructive" />
          <p className="text-xs text-white/80">{errMsg || "3D viewer unavailable"}</p>
          <p className="text-[10px] text-white/40">AI analysis still works via chat ↓</p>
        </div>
      )}
      {status === "ready" && (
        <div className="absolute bottom-2 right-2 flex gap-1">
          <span className="text-[9px] text-white/30 bg-black/40 px-1.5 py-0.5 rounded">
            drag · scroll · right-drag
          </span>
        </div>
      )}
    </div>
  );
};
