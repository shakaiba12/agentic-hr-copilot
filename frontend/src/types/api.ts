export interface RouteDecision {
  category: 'DATA_QUERY' | 'RAG_KNOWLEDGE' | 'CASUAL' | 'GENERAL' | 'OUT_OF_SCOPE' | 'INVALID' | string;
  target?: string | null;
  allowed: boolean;
  confidence: number;
  reason: string;
}

export interface RAGSource {
  chunk_id?: string;
  document_name?: string;
  section_title?: string;
  page_number?: number | string;
  token_count?: number;
  score?: number;
  text?: string;
  [key: string]: unknown;
}

export interface RAGResult {
  success: boolean;
  sources: RAGSource[];
  chunks_count: number;
  grounded: boolean;
  error?: string | null;
}

export interface SQLResult {
  success: boolean;
  generated_sql?: string | null;
  rows: Array<Record<string, unknown>>;
  row_count: number;
  message: string;
  error?: string | null;
  was_truncated?: boolean;
}

export interface ChatResponse {
  query: string;
  response: string;
  source: 'master' | 'rag' | 'sql' | string;
  allowed: boolean;
  decision: RouteDecision;
  rag_result?: RAGResult | null;
  sql_result?: SQLResult | null;
  error?: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  source?: 'master' | 'rag' | 'sql' | string;
  allowed?: boolean;
  decision?: RouteDecision;
  rag_result?: RAGResult | null;
  sql_result?: SQLResult | null;
  error?: string | null;
  isStreaming?: boolean;
}

export interface HealthStatus {
  status: string;
  app: string;
  environment: string;
  provider: string;
  model: string;
  database: string;
  langsmith_tracing: boolean;
  documents_count: number;
}

export interface PolicyDocument {
  filename: string;
  title: string;
  size_bytes: number;
}
