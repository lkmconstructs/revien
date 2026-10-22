-- hand-built v0.2.x-shaped schema (nodes/edges before the confidence layer, no audit_log, no user_version): the repository history starts at 0.3.0, so no 0.2.x code exists to generate one
BEGIN TRANSACTION;
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY, edge_type TEXT NOT NULL, source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL, weight REAL DEFAULT 0.5, created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE);
INSERT INTO "edges" VALUES('v2-e1','mentioned_by','v2-ent','v2-ctx',0.5,'2026-07-05T10:00:00+00:00','{}');
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL,
    content TEXT NOT NULL, source_id TEXT DEFAULT '', created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL, access_count INTEGER DEFAULT 0, metadata TEXT DEFAULT '{}');
INSERT INTO "nodes" VALUES('v2-ctx','context','We chose PostgreSQL for billing.','User: We chose PostgreSQL for billing.
Assistant: Noted.','claude-code:proj-x:s1','2026-07-05T10:00:00+00:00','2026-07-05T10:00:00+00:00',0,'{}');
INSERT INTO "nodes" VALUES('v2-ent','entity','PostgreSQL','Technology/tool: PostgreSQL','claude-code:proj-x:s1','2026-07-05T10:00:00+00:00','2026-07-05T10:00:00+00:00',0,'{}');
INSERT INTO "nodes" VALUES('v2-fact','fact','billing uses PostgreSQL','billing uses PostgreSQL','mcp','2026-07-05T10:00:00+00:00','2026-07-05T10:00:00+00:00',0,'{}');
CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_source ON nodes(source_id);
CREATE INDEX idx_edges_source ON edges(source_node_id);
CREATE INDEX idx_edges_target ON edges(target_node_id);
COMMIT;
PRAGMA user_version = 0;
