/**
 * newsCache.ts — thin client-side cache for the /api/news briefing.
 *
 * prefetchNews(force?)
 *   force = false (default) → server returns its 6-hour cache
 *   force = true            → server re-scrapes immediately (used by Refresh button)
 */

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const CLIENT_TTL_MS = 5 * 60 * 1000; // 5-min client-side cache (avoids hammering on re-render)

export interface NewsItem {
  id:           string;
  title:        string;
  source:       string;
  source_url:   string;
  article_url:  string;
  image_url?:   string | null;
  raw_summary:  string;
  ai_impact:    string;
  category:     string;
  published_at: string;
  scraped_at:   string;
}

export interface NewsBriefing {
  generated_at: string;
  count:        number;
  items:        NewsItem[];
  errors:       string[];
}

// ── In-memory client cache ─────────────────────────────────────────────────

let _briefing:   NewsBriefing | null = null;
let _fetchedAt:  number              = 0;

export function getCachedBriefing(): NewsBriefing | null {
  if (_briefing && Date.now() - _fetchedAt < CLIENT_TTL_MS) return _briefing;
  return null;
}

// ── Fetch ──────────────────────────────────────────────────────────────────

/**
 * @param force  true → bypass the server-side 6-hour cache (used by Refresh button)
 */
export async function prefetchNews(force = false): Promise<NewsBriefing> {
  // Return client cache unless force=true
  const cached = getCachedBriefing();
  if (cached && !force) return cached;

  // Hit the backend — force=true triggers a full re-scrape on the server
  const url = `${API_BASE}/api/news${force ? "?force=true" : ""}`;
  const res  = await fetch(url);

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`News fetch failed (${res.status}): ${text}`);
  }

  const data: NewsBriefing = await res.json();

  // Update client cache
  _briefing  = data;
  _fetchedAt = Date.now();

  return data;
}