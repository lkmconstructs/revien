-- generated from origin/main 03217f4 (user_version 4)
BEGIN TRANSACTION;
CREATE TABLE audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                op TEXT NOT NULL,
                actor TEXT DEFAULT '',
                ts TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT
            );
INSERT INTO "audit_log" VALUES(1,'4ff1eec4-4c63-4457-9780-52d21a8e271d','create','','2026-10-16T22:25:57.832714+00:00',NULL,'{"node_id": "4ff1eec4-4c63-4457-9780-52d21a8e271d", "node_type": "fact", "label": "Uses PostgreSQL", "content": "The team chose PostgreSQL for storage.", "source_id": "claude-code:proj-x:sess-1", "created_at": "2026-10-16T22:25:57.832300Z", "last_accessed": "2026-10-16T22:25:57.832303Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:25:57.832305Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(2,'c2866f44-d49a-4000-b41b-6e5d7a355100','create','','2026-10-16T22:25:57.834016+00:00',NULL,'{"node_id": "c2866f44-d49a-4000-b41b-6e5d7a355100", "node_type": "context", "label": "Session one", "content": "User: hello\nAssistant: hi", "source_id": "claude-code:proj-x:sess-1", "created_at": "2026-10-16T22:25:57.832332Z", "last_accessed": "2026-10-16T22:25:57.832332Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:25:57.832333Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(3,'80d35fb0-91a4-494d-afb0-8ee99e818177','create','','2026-10-16T22:25:57.834828+00:00',NULL,'{"node_id": "80d35fb0-91a4-494d-afb0-8ee99e818177", "node_type": "fact", "label": "Prefers tabs", "content": "The user prefers tabs over spaces.", "source_id": "mcp", "created_at": "2026-10-16T22:25:57.832347Z", "last_accessed": "2026-10-16T22:25:57.832348Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:25:57.832349Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(4,'4ff1eec4-4c63-4457-9780-52d21a8e271d','create','fixture','2026-10-16T22:25:57.836550+00:00',NULL,NULL);
CREATE TABLE distill_manifest (
                note_stem TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                anchor_node_id TEXT NOT NULL,
                current_node_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                section TEXT DEFAULT '',
                PRIMARY KEY (note_stem, anchor_node_id)
            );
CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                confidence REAL DEFAULT 0.5,
                confidence_set_at TEXT,
                confidence_set_by TEXT DEFAULT '',
                source_context TEXT DEFAULT '',
                invalidated_at TEXT,
                FOREIGN KEY (source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            );
INSERT INTO "edges" VALUES('4cc5de99-6cd6-467f-954a-56a848c179f1','related_to','4ff1eec4-4c63-4457-9780-52d21a8e271d','c2866f44-d49a-4000-b41b-6e5d7a355100',0.5,'2026-10-16T22:25:57.835724+00:00','{}',0.5,'2026-10-16T22:25:57.835727+00:00','','',NULL);
CREATE TABLE nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                content TEXT NOT NULL,
                source_id TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                source_type TEXT DEFAULT 'inferred',
                confidence REAL DEFAULT 0.5,
                pinned INTEGER DEFAULT 0,
                confidence_set_at TEXT,
                confidence_set_by TEXT DEFAULT '',
                source_context TEXT DEFAULT '',
                last_referenced TEXT,
                invalidated_at TEXT,
                source_modality TEXT DEFAULT 'text',
                answerable_by_text INTEGER DEFAULT 1,
                vision_processed INTEGER DEFAULT 0,
                recorded_at TEXT,
                event_time_start TEXT,
                event_time_end TEXT,
                event_time_granularity TEXT,
                event_time_confidence REAL,
                event_time_text TEXT DEFAULT '',
                valid_from TEXT,
                valid_until TEXT,
                origin_runtime TEXT,
                origin_source TEXT,
                project_key TEXT,
                session_key TEXT
            );
INSERT INTO "nodes" VALUES('4ff1eec4-4c63-4457-9780-52d21a8e271d','fact','Uses PostgreSQL','The team chose PostgreSQL for storage.','claude-code:proj-x:sess-1','2026-10-16T22:25:57.832300+00:00','2026-10-16T22:25:57.832303+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:25:57.832305+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "nodes" VALUES('c2866f44-d49a-4000-b41b-6e5d7a355100','context','Session one','User: hello
Assistant: hi','claude-code:proj-x:sess-1','2026-10-16T22:25:57.832332+00:00','2026-10-16T22:25:57.832332+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:25:57.832333+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "nodes" VALUES('80d35fb0-91a4-494d-afb0-8ee99e818177','fact','Prefers tabs','The user prefers tabs over spaces.','mcp','2026-10-16T22:25:57.832347+00:00','2026-10-16T22:25:57.832348+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:25:57.832349+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
CREATE TABLE pending_embeds (node_id TEXT PRIMARY KEY, queued_at TEXT NOT NULL);
INSERT INTO "pending_embeds" VALUES('4ff1eec4-4c63-4457-9780-52d21a8e271d','2026-10-16T22:25:57.832638+00:00');
INSERT INTO "pending_embeds" VALUES('c2866f44-d49a-4000-b41b-6e5d7a355100','2026-10-16T22:25:57.833969+00:00');
INSERT INTO "pending_embeds" VALUES('80d35fb0-91a4-494d-afb0-8ee99e818177','2026-10-16T22:25:57.834797+00:00');
CREATE TABLE supersession_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                existing_node_id TEXT NOT NULL,
                new_node_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT DEFAULT '',
                trace TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                resolution TEXT
            );
CREATE TABLE sync_cursors (
                adapter_name TEXT PRIMARY KEY,
                cursor TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
INSERT INTO "sync_cursors" VALUES('claude-code','2026-07-01T00:00:00+00:00','2026-10-16T22:25:57.836199+00:00');
CREATE INDEX idx_nodes_type ON nodes(node_type);
CREATE INDEX idx_nodes_source ON nodes(source_id);
CREATE INDEX idx_nodes_label ON nodes(label);
CREATE INDEX idx_edges_source ON edges(source_node_id);
CREATE INDEX idx_edges_target ON edges(target_node_id);
CREATE INDEX idx_edges_type ON edges(edge_type);
CREATE INDEX idx_nodes_confidence ON nodes(confidence);
CREATE INDEX idx_edges_confidence ON edges(confidence);
CREATE INDEX idx_audit_node ON audit_log(node_id);
CREATE INDEX idx_audit_ts ON audit_log(ts);
CREATE INDEX idx_candidates_unresolved
                ON supersession_candidates(resolved_at);
CREATE INDEX idx_nodes_modality ON nodes(source_modality);
CREATE INDEX idx_nodes_origin_runtime ON nodes(origin_runtime, created_at DESC);
CREATE INDEX idx_nodes_origin_source ON nodes(origin_source);
CREATE INDEX idx_nodes_project ON nodes(project_key, created_at DESC);
CREATE INDEX idx_nodes_session ON nodes(session_key, recorded_at, created_at);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('audit_log',4);
COMMIT;
PRAGMA user_version = 4;
