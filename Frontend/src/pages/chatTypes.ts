import { Source } from "@/services/api";

export interface ThinkingStep { node: string; icon: string; message: string; ts: number; }

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rawAnswer?: string;
  sources?: Source[];
  confidence?: number;
  timestamp: Date;
  thinkingSteps?: ThinkingStep[];
  voiceBlobUrl?: string;
  voiceDuration?: number;
  voiceTranscript?: string;
  voiceWaveform?: number[];
  interrupted?: true;
  callCard?: { duration: number; startedAt: Date };
  attachedDocIds?: string[];
  analytics?: Record<string, any> | null;
  reportId?: string | null;
  reportTitle?: string | null;
  reportMeta?: { word_count: number; section_count: number; source_docs: string[]; version: number } | null;
  reportGenerating?: boolean;
  navAction?: { path: string; label: string; icon: string } | null;
  clarificationOptions?: string[];
  route?: string;
  isRateLimit?: boolean;
}

export interface FactChip {
  label: string;
  value: string;
  raw_line: string;
  is_numeric: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  initialTitle?: string;
  titleLocked?: boolean;
  preview: string;
  timestamp: Date;
  messages: Message[];
}

export interface ChartRecord {
  section_id:     string;
  chart_id:       string;
  chart_js:       Record<string, unknown>;
  title:          string;
  description:    string;
  interpretation: string;
}

export interface VersionInfo {
  version:     number;
  title:       string;
  instruction: string;
  created_at:  string;
}

export interface ReportRecord {
  report_id:   string;
  title:       string;
  content:     string;
  summary?:    string;
  charts:      ChartRecord[];
  source_docs: string[];
  created_at:  string;
  updated_at:  string;
  version:     number;
  versions:    VersionInfo[];
}

export type ModelProvider = string;

export const IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.gif'];
export const CAD_EXTS = ['.ifc', '.ifczip', '.dxf', '.dwg', '.step', '.stp', '.rvt', '.nwd', '.nwc', '.dgn', '.skp', '.3dm', '.fbx', '.obj', '.stl', '.sat', '.iges', '.igs', '.prt', '.sldprt', '.catpart', '.3ds', '.dae', '.rfa', '.rte'];

export const createUniqueId = (prefix = "id-") =>
  crypto.randomUUID?.() ?? `${prefix}${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
