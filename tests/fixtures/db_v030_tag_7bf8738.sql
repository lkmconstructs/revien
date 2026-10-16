-- generated from tag v0.3.0 (7bf8738) (user_version 0)
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
INSERT INTO "audit_log" VALUES(1,'88b24693-c0ab-4f7f-8f1e-e86061ca94b0','create','','2026-10-16T22:26:33.236042+00:00',NULL,'{"node_id": "88b24693-c0ab-4f7f-8f1e-e86061ca94b0", "node_type": "fact", "label": "Uses PostgreSQL", "content": "The team chose PostgreSQL for storage.", "source_id": "claude-code:proj-x:sess-1", "created_at": "2026-10-16T22:26:33.235469Z", "last_accessed": "2026-10-16T22:26:33.235471Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:26:33.235474Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null}');
INSERT INTO "audit_log" VALUES(2,'c667e108-bb18-4fff-afb9-9056962d634f','create','','2026-10-16T22:26:33.236746+00:00',NULL,'{"node_id": "c667e108-bb18-4fff-afb9-9056962d634f", "node_type": "context", "label": "Session one", "content": "User: hello\nAssistant: hi", "source_id": "claude-code:proj-x:sess-1", "created_at": "2026-10-16T22:26:33.235498Z", "last_accessed": "2026-10-16T22:26:33.235498Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:26:33.235499Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null}');
INSERT INTO "audit_log" VALUES(3,'bc7afef8-7b70-4ccd-b953-f23316641f82','create','','2026-10-16T22:26:33.237725+00:00',NULL,'{"node_id": "bc7afef8-7b70-4ccd-b953-f23316641f82", "node_type": "fact", "label": "Prefers tabs", "content": "The user prefers tabs over spaces.", "source_id": "mcp", "created_at": "2026-10-16T22:26:33.235508Z", "last_accessed": "2026-10-16T22:26:33.235508Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:26:33.235509Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null}');
INSERT INTO "audit_log" VALUES(4,'88b24693-c0ab-4f7f-8f1e-e86061ca94b0','create','fixture','2026-10-16T22:26:33.238734+00:00',NULL,NULL);
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
                FOREIGN KEY (source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            );
INSERT INTO "edges" VALUES('7c98151e-5963-47c7-88b4-441b136a69ea','related_to','88b24693-c0ab-4f7f-8f1e-e86061ca94b0','c667e108-bb18-4fff-afb9-9056962d634f',0.5,'2026-10-16T22:26:33.238234+00:00','{}',0.5,'2026-10-16T22:26:33.238236+00:00','','');
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
                valid_until TEXT
            );
INSERT INTO "nodes" VALUES('88b24693-c0ab-4f7f-8f1e-e86061ca94b0','fact','Uses PostgreSQL','The team chose PostgreSQL for storage.','claude-code:proj-x:sess-1','2026-10-16T22:26:33.235469+00:00','2026-10-16T22:26:33.235471+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:26:33.235474+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL);
INSERT INTO "nodes" VALUES('c667e108-bb18-4fff-afb9-9056962d634f','context','Session one','User: hello
Assistant: hi','claude-code:proj-x:sess-1','2026-10-16T22:26:33.235498+00:00','2026-10-16T22:26:33.235498+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:26:33.235499+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL);
INSERT INTO "nodes" VALUES('bc7afef8-7b70-4ccd-b953-f23316641f82','fact','Prefers tabs','The user prefers tabs over spaces.','mcp','2026-10-16T22:26:33.235508+00:00','2026-10-16T22:26:33.235508+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:26:33.235509+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL);
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
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('audit_log',4);
COMMIT;
PRAGMA user_version = 0;
