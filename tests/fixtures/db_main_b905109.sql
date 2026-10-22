-- generated from origin/main b905109 with real flows: ChatGPT and Readwise importers, claude-code adapter sync, mcp ingest, usage ledger, receipt, candidate, sync reset (user_version 5)
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
INSERT INTO "audit_log" VALUES(1,'2839f8d7-4a99-48fc-b053-98fa6c03cd03','create','','2026-10-19T18:54:26.100074+00:00',NULL,'{"node_id": "2839f8d7-4a99-48fc-b053-98fa6c03cd03", "node_type": "context", "label": "Hello, Mara here. We chose Postgres for the orders service.", "content": "User: Hello, Mara here. We chose Postgres for the orders service.\nAssistant: Noted, Postgres for the orders service.", "source_id": "chatgpt:conversation:conv-mara-1", "created_at": "2026-10-19T18:54:26.098758Z", "last_accessed": "2026-10-19T18:54:26.098758Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Fernweh-Core planning", "messages": 2}, "ingest_key": "chatgpt:conversation:conv-mara-1", "ingest_content_hash": "06c09a834b7bf61e06b6b03e045040ad110dd5a3f330ce5403509db2d3498dd8"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.098942Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2023-11-14T22:13:21Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "chatgpt", "origin_source": "import", "project_key": null, "session_key": "conv-mara-1"}');
INSERT INTO "audit_log" VALUES(2,'592c836a-fb58-48cb-a947-8b45480b926b','create','','2026-10-19T18:54:26.101828+00:00',NULL,'{"node_id": "592c836a-fb58-48cb-a947-8b45480b926b", "node_type": "entity", "label": "Postgres", "content": "Technology/tool: Postgres", "source_id": "chatgpt:conversation:conv-mara-1", "created_at": "2026-10-19T18:54:26.098758Z", "last_accessed": "2026-10-19T18:54:26.098758Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.098972Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2023-11-14T22:13:21Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "chatgpt", "origin_source": "import", "project_key": null, "session_key": "conv-mara-1"}');
INSERT INTO "audit_log" VALUES(3,'93982e82-82e7-4eef-8a18-f617d5e5e248','create','','2026-10-19T18:54:26.103274+00:00',NULL,'{"node_id": "93982e82-82e7-4eef-8a18-f617d5e5e248", "node_type": "entity", "label": "orders service", "content": "Entity: orders service", "source_id": "chatgpt:conversation:conv-mara-1", "created_at": "2026-10-19T18:54:26.098758Z", "last_accessed": "2026-10-19T18:54:26.098758Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.099251Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2023-11-14T22:13:21Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "chatgpt", "origin_source": "import", "project_key": null, "session_key": "conv-mara-1"}');
INSERT INTO "audit_log" VALUES(4,'dd59291f-d31f-41ae-8381-7e38230ffcc5','create','','2026-10-19T18:54:26.105162+00:00',NULL,'{"node_id": "dd59291f-d31f-41ae-8381-7e38230ffcc5", "node_type": "topic", "label": "orders service", "content": "Topic: orders service (mentioned 2 times)", "source_id": "chatgpt:conversation:conv-mara-1", "created_at": "2026-10-19T18:54:26.098758Z", "last_accessed": "2026-10-19T18:54:26.098758Z", "access_count": 0, "metadata": {"frequency": 2, "recorded_at_source": "content", "origin_declared": true}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.099485Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2023-11-14T22:13:21Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "chatgpt", "origin_source": "import", "project_key": null, "session_key": "conv-mara-1"}');
INSERT INTO "audit_log" VALUES(5,'bd8a8a07-4e61-4579-8328-2102d837310a','create','','2026-10-19T18:54:26.109296+00:00',NULL,'{"node_id": "bd8a8a07-4e61-4579-8328-2102d837310a", "node_type": "context", "label": "Memory is a graph.", "content": "Memory is a graph.\n\nNote: n", "source_id": "readwise:notes:de966d153f14", "created_at": "2026-10-19T18:54:26.108854Z", "last_accessed": "2026-10-19T18:54:26.108854Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Notes", "author": "Mara", "location": "12", "tags": []}, "ingest_key": "readwise:notes:de966d153f14", "ingest_content_hash": "4f8c8d2ae93cf8cf2e172725fe272b1285d8e9cc4a3a84d0ac4bb3915967ef06"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.108910Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-07-01T10:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "readwise", "origin_source": "import", "project_key": null, "session_key": "notes"}');
INSERT INTO "audit_log" VALUES(6,'ca59317f-5772-4e1c-88bc-56e83a72304a','create','','2026-10-19T18:54:26.110414+00:00',NULL,'{"node_id": "ca59317f-5772-4e1c-88bc-56e83a72304a", "node_type": "entity", "label": "Notes", "content": "Notes", "source_id": "readwise:notes:de966d153f14", "created_at": "2026-10-19T18:54:26.110296Z", "last_accessed": "2026-10-19T18:54:26.110299Z", "access_count": 0, "metadata": {"recorded_at_source": "content"}, "source_type": "extracted", "confidence": 0.8, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.110301Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-07-01T10:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "readwise", "origin_source": "import", "project_key": null, "session_key": "notes"}');
INSERT INTO "audit_log" VALUES(7,'9b5a62d0-8bbc-4fcf-85df-3d39e0942093','create','','2026-10-19T18:54:26.115021+00:00',NULL,'{"node_id": "9b5a62d0-8bbc-4fcf-85df-3d39e0942093", "node_type": "context", "label": "We chose PostgreSQL for billing.", "content": "User: We chose PostgreSQL for billing.\nAssistant: Noted, PostgreSQL for billing.", "source_id": "claude-code:proj-x:s1", "created_at": "2026-10-19T18:54:26.114488Z", "last_accessed": "2026-10-19T18:54:26.114488Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true, "caller": {"adapter": "claude_code", "project": "proj-x", "session_file": "s1.jsonl", "path": "/tmp/tmpwzefsus2/cc/projects/proj-x/s1.jsonl"}, "ingest_key": "claude-code:proj-x:s1", "ingest_content_hash": "f43d0b4bd9cf3554727e6250cd8889d70c527d425e1449e74d3f419b5726913d"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.114529Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-03-05T10:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "claude-code", "origin_source": "live", "project_key": "proj-x", "session_key": "s1"}');
INSERT INTO "audit_log" VALUES(8,'c09f4927-cca6-4dd2-8f23-ecff1ebc15f6','create','','2026-10-19T18:54:26.120674+00:00',NULL,'{"node_id": "c09f4927-cca6-4dd2-8f23-ecff1ebc15f6", "node_type": "context", "label": "I prefer tabs over spaces in Python.", "content": "I prefer tabs over spaces in Python.", "source_id": "mcp", "created_at": "2026-10-19T18:54:26.120301Z", "last_accessed": "2026-10-19T18:54:26.120301Z", "access_count": 0, "metadata": {}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.120347Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(9,'1126b155-6338-4142-898a-084bc609bf75','create','','2026-10-19T18:54:26.122757+00:00',NULL,'{"node_id": "1126b155-6338-4142-898a-084bc609bf75", "node_type": "entity", "label": "Python", "content": "Technology/tool: Python", "source_id": "mcp", "created_at": "2026-10-19T18:54:26.120301Z", "last_accessed": "2026-10-19T18:54:26.120301Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.120374Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(10,'f00150b6-20f2-4d5d-8273-6461dadc54c4','create','','2026-10-19T18:54:26.124979+00:00',NULL,'{"node_id": "f00150b6-20f2-4d5d-8273-6461dadc54c4", "node_type": "preference", "label": "tabs over spaces in Python", "content": "I prefer tabs over spaces in Python.", "source_id": "mcp", "created_at": "2026-10-19T18:54:26.120301Z", "last_accessed": "2026-10-19T18:54:26.120301Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-19T18:54:26.120467Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
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
INSERT INTO "edges" VALUES('a70d8efd-775b-46ed-b592-b820b4838aac','mentioned_by','592c836a-fb58-48cb-a947-8b45480b926b','2839f8d7-4a99-48fc-b053-98fa6c03cd03',0.5,'2026-10-19T18:54:26.106078+00:00','{}',0.5,'2026-10-19T18:54:26.106081+00:00','','',NULL);
INSERT INTO "edges" VALUES('9ee43494-2a45-4f33-ad95-412574813089','mentioned_by','93982e82-82e7-4eef-8a18-f617d5e5e248','2839f8d7-4a99-48fc-b053-98fa6c03cd03',0.5,'2026-10-19T18:54:26.106702+00:00','{}',0.5,'2026-10-19T18:54:26.106705+00:00','','',NULL);
INSERT INTO "edges" VALUES('fd171cc7-cb6f-42fb-be51-37492a1a326c','related_to','dd59291f-d31f-41ae-8381-7e38230ffcc5','2839f8d7-4a99-48fc-b053-98fa6c03cd03',0.5,'2026-10-19T18:54:26.107162+00:00','{}',0.5,'2026-10-19T18:54:26.107164+00:00','','',NULL);
INSERT INTO "edges" VALUES('10f3b847-4b5a-47cc-9a10-4c899a3e73af','related_to','592c836a-fb58-48cb-a947-8b45480b926b','93982e82-82e7-4eef-8a18-f617d5e5e248',0.3,'2026-10-19T18:54:26.107703+00:00','{}',0.5,'2026-10-19T18:54:26.107704+00:00','','',NULL);
INSERT INTO "edges" VALUES('7a4eadf4-7219-47c8-a11a-c7560365c498','related_to','bd8a8a07-4e61-4579-8328-2102d837310a','ca59317f-5772-4e1c-88bc-56e83a72304a',0.8,'2026-10-19T18:54:26.112259+00:00','{}',0.5,'2026-10-19T18:54:26.112263+00:00','','',NULL);
INSERT INTO "edges" VALUES('10536942-05d5-48e3-8c47-8215826dcff9','mentioned_by','592c836a-fb58-48cb-a947-8b45480b926b','9b5a62d0-8bbc-4fcf-85df-3d39e0942093',0.5,'2026-10-19T18:54:26.118625+00:00','{}',0.5,'2026-10-19T18:54:26.118628+00:00','','',NULL);
INSERT INTO "edges" VALUES('fed7dab1-cc9c-4797-8827-274511120a41','mentioned_by','1126b155-6338-4142-898a-084bc609bf75','c09f4927-cca6-4dd2-8f23-ecff1ebc15f6',0.5,'2026-10-19T18:54:26.126136+00:00','{}',0.5,'2026-10-19T18:54:26.126140+00:00','','',NULL);
INSERT INTO "edges" VALUES('32154a5a-0229-485b-87e6-5d7103045592','related_to','f00150b6-20f2-4d5d-8273-6461dadc54c4','c09f4927-cca6-4dd2-8f23-ecff1ebc15f6',0.5,'2026-10-19T18:54:26.126796+00:00','{}',0.5,'2026-10-19T18:54:26.126799+00:00','','',NULL);
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
INSERT INTO "nodes" VALUES('2839f8d7-4a99-48fc-b053-98fa6c03cd03','context','Hello, Mara here. We chose Postgres for the orders service.','User: Hello, Mara here. We chose Postgres for the orders service.
Assistant: Noted, Postgres for the orders service.','chatgpt:conversation:conv-mara-1','2026-10-19T18:54:26.098758+00:00','2026-10-19T18:54:26.098758+00:00',0,'{"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Fernweh-Core planning", "messages": 2}, "ingest_key": "chatgpt:conversation:conv-mara-1", "ingest_content_hash": "06c09a834b7bf61e06b6b03e045040ad110dd5a3f330ce5403509db2d3498dd8"}','extracted',1.0,0,'2026-10-19T18:54:26.098942+00:00','','',NULL,NULL,'text',1,0,'2023-11-14T22:13:21+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'chatgpt','import',NULL,'conv-mara-1');
INSERT INTO "nodes" VALUES('592c836a-fb58-48cb-a947-8b45480b926b','entity','Postgres','Technology/tool: Postgres','chatgpt:conversation:conv-mara-1','2026-10-19T18:54:26.098758+00:00','2026-10-19T18:54:26.118005+00:00',2,'{"recorded_at_source": "content", "origin_declared": true}','inferred',0.5,0,'2026-10-19T18:54:26.098972+00:00','','',NULL,NULL,'text',1,0,'2023-11-14T22:13:21+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'chatgpt','import',NULL,'conv-mara-1');
INSERT INTO "nodes" VALUES('93982e82-82e7-4eef-8a18-f617d5e5e248','entity','orders service','Entity: orders service','chatgpt:conversation:conv-mara-1','2026-10-19T18:54:26.098758+00:00','2026-10-19T18:54:26.098758+00:00',0,'{"recorded_at_source": "content", "origin_declared": true}','inferred',0.5,0,'2026-10-19T18:54:26.099251+00:00','','',NULL,NULL,'text',1,0,'2023-11-14T22:13:21+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'chatgpt','import',NULL,'conv-mara-1');
INSERT INTO "nodes" VALUES('dd59291f-d31f-41ae-8381-7e38230ffcc5','topic','orders service','Topic: orders service (mentioned 2 times)','chatgpt:conversation:conv-mara-1','2026-10-19T18:54:26.098758+00:00','2026-10-19T18:54:26.098758+00:00',0,'{"frequency": 2, "recorded_at_source": "content", "origin_declared": true}','inferred',0.5,0,'2026-10-19T18:54:26.099485+00:00','','',NULL,NULL,'text',1,0,'2023-11-14T22:13:21+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'chatgpt','import',NULL,'conv-mara-1');
INSERT INTO "nodes" VALUES('bd8a8a07-4e61-4579-8328-2102d837310a','context','Memory is a graph.','Memory is a graph.

Note: n','readwise:notes:de966d153f14','2026-10-19T18:54:26.108854+00:00','2026-10-19T18:54:26.108854+00:00',0,'{"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Notes", "author": "Mara", "location": "12", "tags": []}, "ingest_key": "readwise:notes:de966d153f14", "ingest_content_hash": "4f8c8d2ae93cf8cf2e172725fe272b1285d8e9cc4a3a84d0ac4bb3915967ef06"}','extracted',1.0,0,'2026-10-19T18:54:26.108910+00:00','','',NULL,NULL,'text',1,0,'2026-07-01T10:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'readwise','import',NULL,'notes');
INSERT INTO "nodes" VALUES('ca59317f-5772-4e1c-88bc-56e83a72304a','entity','Notes','Notes','readwise:notes:de966d153f14','2026-10-19T18:54:26.110296+00:00','2026-10-19T18:54:26.110299+00:00',0,'{"recorded_at_source": "content"}','extracted',0.8,0,'2026-10-19T18:54:26.110301+00:00','','',NULL,NULL,'text',1,0,'2026-07-01T10:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'readwise','import',NULL,'notes');
INSERT INTO "nodes" VALUES('9b5a62d0-8bbc-4fcf-85df-3d39e0942093','context','We chose PostgreSQL for billing.','User: We chose PostgreSQL for billing.
Assistant: Noted, PostgreSQL for billing.','claude-code:proj-x:s1','2026-10-19T18:54:26.114488+00:00','2026-10-19T18:54:26.114488+00:00',0,'{"recorded_at_source": "content", "origin_declared": true, "caller": {"adapter": "claude_code", "project": "proj-x", "session_file": "s1.jsonl", "path": "/tmp/tmpwzefsus2/cc/projects/proj-x/s1.jsonl"}, "ingest_key": "claude-code:proj-x:s1", "ingest_content_hash": "f43d0b4bd9cf3554727e6250cd8889d70c527d425e1449e74d3f419b5726913d"}','extracted',1.0,0,'2026-10-19T18:54:26.114529+00:00','','',NULL,NULL,'text',1,0,'2026-03-05T10:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'claude-code','live','proj-x','s1');
INSERT INTO "nodes" VALUES('c09f4927-cca6-4dd2-8f23-ecff1ebc15f6','context','I prefer tabs over spaces in Python.','I prefer tabs over spaces in Python.','mcp','2026-10-19T18:54:26.120301+00:00','2026-10-19T18:54:26.120301+00:00',0,'{}','extracted',1.0,0,'2026-10-19T18:54:26.120347+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "nodes" VALUES('1126b155-6338-4142-898a-084bc609bf75','entity','Python','Technology/tool: Python','mcp','2026-10-19T18:54:26.120301+00:00','2026-10-19T18:54:26.120301+00:00',0,'{}','inferred',0.5,0,'2026-10-19T18:54:26.120374+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "nodes" VALUES('f00150b6-20f2-4d5d-8273-6461dadc54c4','preference','tabs over spaces in Python','I prefer tabs over spaces in Python.','mcp','2026-10-19T18:54:26.120301+00:00','2026-10-19T18:54:26.120301+00:00',0,'{}','inferred',0.5,0,'2026-10-19T18:54:26.120467+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
CREATE TABLE pending_embeds (node_id TEXT PRIMARY KEY, queued_at TEXT NOT NULL);
INSERT INTO "pending_embeds" VALUES('2839f8d7-4a99-48fc-b053-98fa6c03cd03','2026-10-19T18:54:26.099965+00:00');
INSERT INTO "pending_embeds" VALUES('592c836a-fb58-48cb-a947-8b45480b926b','2026-10-19T18:54:26.101762+00:00');
INSERT INTO "pending_embeds" VALUES('93982e82-82e7-4eef-8a18-f617d5e5e248','2026-10-19T18:54:26.103233+00:00');
INSERT INTO "pending_embeds" VALUES('dd59291f-d31f-41ae-8381-7e38230ffcc5','2026-10-19T18:54:26.105091+00:00');
INSERT INTO "pending_embeds" VALUES('bd8a8a07-4e61-4579-8328-2102d837310a','2026-10-19T18:54:26.109250+00:00');
INSERT INTO "pending_embeds" VALUES('ca59317f-5772-4e1c-88bc-56e83a72304a','2026-10-19T18:54:26.110379+00:00');
INSERT INTO "pending_embeds" VALUES('9b5a62d0-8bbc-4fcf-85df-3d39e0942093','2026-10-19T18:54:26.114969+00:00');
INSERT INTO "pending_embeds" VALUES('c09f4927-cca6-4dd2-8f23-ecff1ebc15f6','2026-10-19T18:54:26.120635+00:00');
INSERT INTO "pending_embeds" VALUES('1126b155-6338-4142-898a-084bc609bf75','2026-10-19T18:54:26.122720+00:00');
INSERT INTO "pending_embeds" VALUES('f00150b6-20f2-4d5d-8273-6461dadc54c4','2026-10-19T18:54:26.124939+00:00');
CREATE TABLE recall_receipts (
            recall_id TEXT PRIMARY KEY,
            runtime TEXT NOT NULL DEFAULT '',
            query TEXT NOT NULL DEFAULT '',
            node_ids TEXT NOT NULL DEFAULT '[]',
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            closed_at TEXT
        );
INSERT INTO "recall_receipts" VALUES('rc-1','claude-code','what db','["1126b155-6338-4142-898a-084bc609bf75"]','2026-10-19T18:54:26.129335+00:00','2026-10-19T18:54:26.129338+00:00','open',NULL);
CREATE TABLE store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
INSERT INTO "store_meta" VALUES('daemon_heartbeat:99999','{"features": ["sync_reset_fence"], "pid": 99999, "time": "2020-01-01T00:00:00+00:00"}');
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
INSERT INTO "supersession_candidates" VALUES(1,'1126b155-6338-4142-898a-084bc609bf75','c09f4927-cca6-4dd2-8f23-ecff1ebc15f6','candidate','c','{}','2026-10-19T18:54:26.130185+00:00',NULL,NULL);
CREATE TABLE sync_cursors (
                adapter_name TEXT PRIMARY KEY,
                cursor TEXT NOT NULL,
                updated_at TEXT NOT NULL
            , generation INTEGER NOT NULL DEFAULT 0);
INSERT INTO "sync_cursors" VALUES('claude-code','','2026-10-19T18:54:26.131398+00:00',1);
CREATE TABLE use_judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_key TEXT NOT NULL,
            node_id TEXT NOT NULL,
            used INTEGER NOT NULL,
            reason TEXT DEFAULT '',
            shared_terms TEXT DEFAULT '[]',
            similarity REAL,
            query TEXT DEFAULT '',
            mode TEXT NOT NULL,
            applied INTEGER NOT NULL DEFAULT 0,
            judged_at TEXT NOT NULL,
            UNIQUE (episode_key, node_id)
        );
INSERT INTO "use_judgments" VALUES(1,'s#t1','1126b155-6338-4142-898a-084bc609bf75',1,'r','["postgres"]',0.8,'what db','shadow',0,'2026-10-19T18:54:26.127560+00:00');
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
CREATE INDEX idx_use_judgments_node
            ON use_judgments(node_id);
CREATE INDEX idx_recall_receipts_status
            ON recall_receipts(status, expires_at);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('audit_log',10);
INSERT INTO "sqlite_sequence" VALUES('use_judgments',1);
INSERT INTO "sqlite_sequence" VALUES('supersession_candidates',1);
COMMIT;
PRAGMA user_version = 5;
