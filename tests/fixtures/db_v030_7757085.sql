-- generated from 7757085 (Revien 0.3.0) (user_version 0)
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
INSERT INTO "audit_log" VALUES(1,'bbd0abf8-324b-4040-978e-620c8fe97b7c','create','','2026-10-16T22:25:58.014611+00:00',NULL,'{"node_id": "bbd0abf8-324b-4040-978e-620c8fe97b7c", "node_type": "fact", "label": "Uses PostgreSQL", "content": "The team chose PostgreSQL for storage.", "source_id": "claude-code:proj-x:sess-1", "created_at": "2026-10-16T22:25:58.014105Z", "last_accessed": "2026-10-16T22:25:58.014107Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:25:58.014109Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null}');
INSERT INTO "audit_log" VALUES(2,'77fa31b9-94f9-4953-a3fd-e78e5744677f','create','','2026-10-16T22:25:58.015309+00:00',NULL,'{"node_id": "77fa31b9-94f9-4953-a3fd-e78e5744677f", "node_type": "context", "label": "Session one", "content": "User: hello\nAssistant: hi", "source_id": "claude-code:proj-x:sess-1", "created_at": "2026-10-16T22:25:58.014131Z", "last_accessed": "2026-10-16T22:25:58.014132Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:25:58.014133Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null}');
INSERT INTO "audit_log" VALUES(3,'6dfede93-d2f3-41e5-b84a-c407822797b9','create','','2026-10-16T22:25:58.016237+00:00',NULL,'{"node_id": "6dfede93-d2f3-41e5-b84a-c407822797b9", "node_type": "fact", "label": "Prefers tabs", "content": "The user prefers tabs over spaces.", "source_id": "mcp", "created_at": "2026-10-16T22:25:58.014141Z", "last_accessed": "2026-10-16T22:25:58.014142Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:25:58.014142Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null}');
INSERT INTO "audit_log" VALUES(4,'bbd0abf8-324b-4040-978e-620c8fe97b7c','create','fixture','2026-10-16T22:25:58.017244+00:00',NULL,NULL);
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
INSERT INTO "edges" VALUES('4ffe3ea0-ac87-4261-813f-cf079d65c4e7','related_to','bbd0abf8-324b-4040-978e-620c8fe97b7c','77fa31b9-94f9-4953-a3fd-e78e5744677f',0.5,'2026-10-16T22:25:58.016701+00:00','{}',0.5,'2026-10-16T22:25:58.016704+00:00','','');
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
INSERT INTO "nodes" VALUES('bbd0abf8-324b-4040-978e-620c8fe97b7c','fact','Uses PostgreSQL','The team chose PostgreSQL for storage.','claude-code:proj-x:sess-1','2026-10-16T22:25:58.014105+00:00','2026-10-16T22:25:58.014107+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:25:58.014109+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL);
INSERT INTO "nodes" VALUES('77fa31b9-94f9-4953-a3fd-e78e5744677f','context','Session one','User: hello
Assistant: hi','claude-code:proj-x:sess-1','2026-10-16T22:25:58.014131+00:00','2026-10-16T22:25:58.014132+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:25:58.014133+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL);
INSERT INTO "nodes" VALUES('6dfede93-d2f3-41e5-b84a-c407822797b9','fact','Prefers tabs','The user prefers tabs over spaces.','mcp','2026-10-16T22:25:58.014141+00:00','2026-10-16T22:25:58.014142+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:25:58.014142+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL);
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
