import React from "react";
import { Loader2, ScrollText } from "lucide-react";
import { ReportRecord } from "@/pages/chatTypes";

export const ReportCard = ({
    report,
    inlineTitle,
    inlineMeta,
    isLoading: cardLoading = false,
    downloadingReportId,
    reports,
    onActivate,
    onDownload,
}: {
    report: ReportRecord | null;
    inlineTitle?: string | null;
    inlineMeta?: { word_count: number; section_count: number; source_docs: string[]; version: number } | null;
    isLoading?: boolean;
    downloadingReportId?: string | null;
    reports?: ReportRecord[];
    onActivate?: (report: ReportRecord | null) => void;
    onDownload?: (report: ReportRecord, fmt: "pdf" | "md") => void;
}) => {
    const isDownloading = report ? downloadingReportId === report.report_id : false;

    const title = report?.title ?? inlineTitle ?? null;
    const wordCount = report ? report.content.split(/\s+/).length : (inlineMeta?.word_count ?? 0);
    const sectionCount = report ? (report.content.match(/^#{1,2}\s/gm) || []).length : (inlineMeta?.section_count ?? 0);
    const sourceDocs = report?.source_docs ?? inlineMeta?.source_docs ?? [];
    const version = report?.version ?? inlineMeta?.version ?? 1;

    if (cardLoading && !title) {
        return (
            <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg border border-border bg-muted/30">
                <Loader2 className="h-3.5 w-3.5 text-primary animate-spin shrink-0" />
                <span className="text-xs text-muted-foreground italic">Building report…</span>
            </div>
        );
    }

    return (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl border border-border bg-muted/20 hover:bg-muted/40 hover:border-primary/30 transition-all duration-150 group">
            <ScrollText className="h-5 w-5 text-primary/70 shrink-0" />

            <div
                className="flex-1 min-w-0 cursor-pointer"
                onClick={() => {
                    if (report) {
                        onActivate?.(report);
                    } else if (inlineTitle) {
                        const fullReport = reports?.find(r => r.title === inlineTitle) ?? null;
                        if (fullReport) onActivate?.(fullReport);
                    }
                }}
            >
                <span className="text-sm font-semibold text-foreground truncate block group-hover:text-primary transition-colors">
                    {title}
                </span>
                <span className="text-xs text-muted-foreground/70 mt-0.5 block">
                    {sectionCount > 0 && `${sectionCount} sections · `}{wordCount > 0 ? `${wordCount.toLocaleString()} words` : ""}
                    {version > 1 && ` · v${version}`}
                    {sourceDocs.length > 0 && ` · ${sourceDocs.length} source${sourceDocs.length === 1 ? "" : "s"}`}
                </span>
            </div>

            <div className="flex items-center gap-1.5 shrink-0">
                <button
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded transition-colors disabled:opacity-40"
                    onClick={e => { e.stopPropagation(); if (report) onDownload?.(report, "md"); }}
                    disabled={!report || isDownloading}
                    title={report ? "Download Markdown" : "Report loading…"}
                >
                    MD
                </button>
                <button
                    className="text-xs font-medium text-primary hover:text-primary/70 px-2 py-1 rounded transition-colors disabled:opacity-40 flex items-center gap-1"
                    onClick={e => { e.stopPropagation(); if (report) onDownload?.(report, "pdf"); }}
                    disabled={!report || isDownloading}
                    title={report ? "Download PDF" : "Report loading…"}
                >
                    {isDownloading && <Loader2 className="h-3 w-3 animate-spin" />}
                    PDF
                </button>
            </div>
        </div>
    );
};
