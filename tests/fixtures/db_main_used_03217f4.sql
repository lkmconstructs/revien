-- generated from origin/main 03217f4 with real flows: ChatGPT/Claude.ai/Readwise importers, claude-code adapter sync, mcp ingest, recall (user_version 4)
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
INSERT INTO "audit_log" VALUES(1,'4eca7581-a23f-4e4f-975b-676bb479e788','create','','2026-10-16T22:35:01.605324+00:00',NULL,'{"node_id": "4eca7581-a23f-4e4f-975b-676bb479e788", "node_type": "context", "label": "Hello, Mara here.", "content": "User: Hello, Mara here.\nAssistant: New reply \u2014 the current branch.\nAssistant: You''re welcome, Theo will see this too.", "source_id": "chatgpt:conversation:conv-mara-1", "created_at": "2026-10-16T22:35:01.603839Z", "last_accessed": "2026-10-16T22:35:01.603839Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Fernweh-Core planning", "messages": 3}, "ingest_key": "chatgpt:conversation:conv-mara-1", "ingest_content_hash": "0ce6af7617fbdfab28a5f790ba13db773f08dfb4eecaab726d4bd15b2976cca3"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.604055Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2023-11-14T22:13:30Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "chatgpt", "origin_source": "import", "project_key": null, "session_key": "conv-mara-1"}');
INSERT INTO "audit_log" VALUES(2,'3c90296b-7ba6-418a-852d-e0a5b47b7179','create','','2026-10-16T22:35:01.608048+00:00',NULL,'{"node_id": "3c90296b-7ba6-418a-852d-e0a5b47b7179", "node_type": "context", "label": "What did we decide about Fernweh-Core''s rollout?", "content": "User: What did we decide about Fernweh-Core''s rollout?\nAssistant: We staged it behind a flag for Sam''s team.", "source_id": "claude:conversation:conv-theo-1", "created_at": "2026-10-16T22:35:01.607362Z", "last_accessed": "2026-10-16T22:35:01.607362Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Sam''s onboarding notes", "messages": 2}, "ingest_key": "claude:conversation:conv-theo-1", "ingest_content_hash": "51ab3376a08bee6183ca4c652a44cc15a2e297cd7ec8e5f91a05df345ad48daf"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.607407Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-01-05T12:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "claude", "origin_source": "import", "project_key": null, "session_key": "conv-theo-1"}');
INSERT INTO "audit_log" VALUES(3,'36d6ae80-2cac-48cf-9f60-b20815e16e14','create','','2026-10-16T22:35:01.609801+00:00',NULL,'{"node_id": "36d6ae80-2cac-48cf-9f60-b20815e16e14", "node_type": "decision", "label": "about Fernweh-Core''s rollout?", "content": "we decide about Fernweh-Core''s rollout?", "source_id": "claude:conversation:conv-theo-1", "created_at": "2026-10-16T22:35:01.607362Z", "last_accessed": "2026-10-16T22:35:01.607362Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.607556Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-01-05T12:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "claude", "origin_source": "import", "project_key": null, "session_key": "conv-theo-1"}');
INSERT INTO "audit_log" VALUES(4,'318b49ab-0318-48cf-b9e4-9157ee35a114','create','','2026-10-16T22:35:01.612595+00:00',NULL,'{"node_id": "318b49ab-0318-48cf-b9e4-9157ee35a114", "node_type": "context", "label": "Memory is a graph", "content": "Memory is a graph\n\nNote: n", "source_id": "readwise:notes:bcb473067a1b", "created_at": "2026-10-16T22:35:01.612290Z", "last_accessed": "2026-10-16T22:35:01.612290Z", "access_count": 0, "metadata": {"origin_declared": true, "caller": {"title": "Notes", "author": "Mara", "location": "12", "tags": []}, "ingest_key": "readwise:notes:bcb473067a1b", "ingest_content_hash": "195fe944f4a398a2a5a33aaa1bde03b8706b214e90e6f21e7e76eeab77c59f2c"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.612327Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "readwise", "origin_source": "import", "project_key": null, "session_key": "notes"}');
INSERT INTO "audit_log" VALUES(5,'5f3cbb81-b8b3-40cd-9d64-0cc09b293a51','create','','2026-10-16T22:35:01.613789+00:00',NULL,'{"node_id": "5f3cbb81-b8b3-40cd-9d64-0cc09b293a51", "node_type": "entity", "label": "Notes", "content": "Notes", "source_id": "readwise:notes:bcb473067a1b", "created_at": "2026-10-16T22:35:01.613662Z", "last_accessed": "2026-10-16T22:35:01.613665Z", "access_count": 0, "metadata": {}, "source_type": "extracted", "confidence": 0.8, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.613669Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "readwise", "origin_source": "import", "project_key": null, "session_key": "notes"}');
INSERT INTO "audit_log" VALUES(6,'4db89123-1d02-4b9b-87c7-79ef3b7d94a4','create','','2026-10-16T22:35:01.617353+00:00',NULL,'{"node_id": "4db89123-1d02-4b9b-87c7-79ef3b7d94a4", "node_type": "context", "label": "We chose PostgreSQL for billing.", "content": "User: We chose PostgreSQL for billing.\nAssistant: Noted, PostgreSQL for billing.", "source_id": "claude-code:proj-x:s1", "created_at": "2026-10-16T22:35:01.616807Z", "last_accessed": "2026-10-16T22:35:01.616807Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true, "caller": {"adapter": "claude_code", "project": "proj-x", "session_file": "s1.jsonl", "path": "/tmp/tmpm1lqe_56/projects/proj-x/s1.jsonl"}, "ingest_key": "claude-code:proj-x:s1", "ingest_content_hash": "f43d0b4bd9cf3554727e6250cd8889d70c527d425e1449e74d3f419b5726913d"}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.616844Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-03-05T10:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "claude-code", "origin_source": "live", "project_key": "proj-x", "session_key": "s1"}');
INSERT INTO "audit_log" VALUES(7,'dd425246-a2bc-47e1-9f1a-f9a34e29c63d','create','','2026-10-16T22:35:01.620600+00:00',NULL,'{"node_id": "dd425246-a2bc-47e1-9f1a-f9a34e29c63d", "node_type": "entity", "label": "PostgreSQL", "content": "Technology/tool: PostgreSQL", "source_id": "claude-code:proj-x:s1", "created_at": "2026-10-16T22:35:01.616807Z", "last_accessed": "2026-10-16T22:35:01.616807Z", "access_count": 0, "metadata": {"recorded_at_source": "content", "origin_declared": true}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.616888Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": "2026-03-05T10:00:00Z", "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": "claude-code", "origin_source": "live", "project_key": "proj-x", "session_key": "s1"}');
INSERT INTO "audit_log" VALUES(8,'58ff1529-f28f-47cc-acb8-b2b5b204caab','create','','2026-10-16T22:35:01.624326+00:00',NULL,'{"node_id": "58ff1529-f28f-47cc-acb8-b2b5b204caab", "node_type": "context", "label": "I prefer tabs over spaces in Python.", "content": "I prefer tabs over spaces in Python.", "source_id": "mcp", "created_at": "2026-10-16T22:35:01.623998Z", "last_accessed": "2026-10-16T22:35:01.623998Z", "access_count": 0, "metadata": {}, "source_type": "extracted", "confidence": 1.0, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.624039Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(9,'0ecf48f1-356f-492a-a400-b87afbb0154d','create','','2026-10-16T22:35:01.626098+00:00',NULL,'{"node_id": "0ecf48f1-356f-492a-a400-b87afbb0154d", "node_type": "entity", "label": "Python", "content": "Technology/tool: Python", "source_id": "mcp", "created_at": "2026-10-16T22:35:01.623998Z", "last_accessed": "2026-10-16T22:35:01.623998Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.624062Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
INSERT INTO "audit_log" VALUES(10,'25b5e56a-7f49-41d1-a27b-df44aa96ce63','create','','2026-10-16T22:35:01.628461+00:00',NULL,'{"node_id": "25b5e56a-7f49-41d1-a27b-df44aa96ce63", "node_type": "preference", "label": "tabs over spaces in Python", "content": "I prefer tabs over spaces in Python.", "source_id": "mcp", "created_at": "2026-10-16T22:35:01.623998Z", "last_accessed": "2026-10-16T22:35:01.623998Z", "access_count": 0, "metadata": {}, "source_type": "inferred", "confidence": 0.5, "pinned": false, "confidence_set_at": "2026-10-16T22:35:01.624157Z", "confidence_set_by": "", "source_context": "", "last_referenced": null, "invalidated_at": null, "source_modality": "text", "answerable_by_text": true, "vision_processed": false, "recorded_at": null, "event_time_start": null, "event_time_end": null, "event_time_granularity": null, "event_time_confidence": null, "event_time_text": "", "valid_from": null, "valid_until": null, "origin_runtime": null, "origin_source": null, "project_key": null, "session_key": null}');
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
INSERT INTO "edges" VALUES('89471879-cab6-4eb2-87b6-1bc92a786df8','decided_in','36d6ae80-2cac-48cf-9f60-b20815e16e14','3c90296b-7ba6-418a-852d-e0a5b47b7179',0.5,'2026-10-16T22:35:01.610917+00:00','{}',0.5,'2026-10-16T22:35:01.610920+00:00','','',NULL);
INSERT INTO "edges" VALUES('b8217ba5-82b9-406b-b3cd-55212b4fe6c9','related_to','318b49ab-0318-48cf-b9e4-9157ee35a114','5f3cbb81-b8b3-40cd-9d64-0cc09b293a51',0.8,'2026-10-16T22:35:01.614805+00:00','{}',0.5,'2026-10-16T22:35:01.614808+00:00','','',NULL);
INSERT INTO "edges" VALUES('8ad89c91-77e5-442b-952d-f56ca4620116','mentioned_by','5f3cbb81-b8b3-40cd-9d64-0cc09b293a51','4db89123-1d02-4b9b-87c7-79ef3b7d94a4',0.5,'2026-10-16T22:35:01.621633+00:00','{}',0.5,'2026-10-16T22:35:01.621636+00:00','','',NULL);
INSERT INTO "edges" VALUES('b985e758-b015-4bb0-a45e-ddbd5526e29c','mentioned_by','dd425246-a2bc-47e1-9f1a-f9a34e29c63d','4db89123-1d02-4b9b-87c7-79ef3b7d94a4',0.5,'2026-10-16T22:35:01.622263+00:00','{}',0.5,'2026-10-16T22:35:01.622264+00:00','','',NULL);
INSERT INTO "edges" VALUES('4b01d72b-f8fd-4301-b6bf-b36771a79dae','related_to','5f3cbb81-b8b3-40cd-9d64-0cc09b293a51','dd425246-a2bc-47e1-9f1a-f9a34e29c63d',0.3,'2026-10-16T22:35:01.622758+00:00','{}',0.5,'2026-10-16T22:35:01.622760+00:00','','',NULL);
INSERT INTO "edges" VALUES('34b47ff4-3cee-499f-88c9-1e11250b67e9','mentioned_by','0ecf48f1-356f-492a-a400-b87afbb0154d','58ff1529-f28f-47cc-acb8-b2b5b204caab',0.5,'2026-10-16T22:35:01.629354+00:00','{}',0.5,'2026-10-16T22:35:01.629357+00:00','','',NULL);
INSERT INTO "edges" VALUES('2a231178-fadd-4a32-99ef-60dea7bff499','related_to','25b5e56a-7f49-41d1-a27b-df44aa96ce63','58ff1529-f28f-47cc-acb8-b2b5b204caab',0.5,'2026-10-16T22:35:01.629889+00:00','{}',0.5,'2026-10-16T22:35:01.629892+00:00','','',NULL);
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
INSERT INTO "nodes" VALUES('4eca7581-a23f-4e4f-975b-676bb479e788','context','Hello, Mara here.','User: Hello, Mara here.
Assistant: New reply — the current branch.
Assistant: You''re welcome, Theo will see this too.','chatgpt:conversation:conv-mara-1','2026-10-16T22:35:01.603839+00:00','2026-10-16T22:35:01.603839+00:00',0,'{"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Fernweh-Core planning", "messages": 3}, "ingest_key": "chatgpt:conversation:conv-mara-1", "ingest_content_hash": "0ce6af7617fbdfab28a5f790ba13db773f08dfb4eecaab726d4bd15b2976cca3"}','extracted',1.0,0,'2026-10-16T22:35:01.604055+00:00','','',NULL,NULL,'text',1,0,'2023-11-14T22:13:30+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'chatgpt','import',NULL,'conv-mara-1');
INSERT INTO "nodes" VALUES('3c90296b-7ba6-418a-852d-e0a5b47b7179','context','What did we decide about Fernweh-Core''s rollout?','User: What did we decide about Fernweh-Core''s rollout?
Assistant: We staged it behind a flag for Sam''s team.','claude:conversation:conv-theo-1','2026-10-16T22:35:01.607362+00:00','2026-10-16T22:35:01.607362+00:00',0,'{"recorded_at_source": "content", "origin_declared": true, "caller": {"title": "Sam''s onboarding notes", "messages": 2}, "ingest_key": "claude:conversation:conv-theo-1", "ingest_content_hash": "51ab3376a08bee6183ca4c652a44cc15a2e297cd7ec8e5f91a05df345ad48daf"}','extracted',1.0,0,'2026-10-16T22:35:01.607407+00:00','','',NULL,NULL,'text',1,0,'2026-01-05T12:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'claude','import',NULL,'conv-theo-1');
INSERT INTO "nodes" VALUES('36d6ae80-2cac-48cf-9f60-b20815e16e14','decision','about Fernweh-Core''s rollout?','we decide about Fernweh-Core''s rollout?','claude:conversation:conv-theo-1','2026-10-16T22:35:01.607362+00:00','2026-10-16T22:35:01.607362+00:00',0,'{"recorded_at_source": "content", "origin_declared": true}','inferred',0.5,0,'2026-10-16T22:35:01.607556+00:00','','',NULL,NULL,'text',1,0,'2026-01-05T12:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'claude','import',NULL,'conv-theo-1');
INSERT INTO "nodes" VALUES('318b49ab-0318-48cf-b9e4-9157ee35a114','context','Memory is a graph','Memory is a graph

Note: n','readwise:notes:bcb473067a1b','2026-10-16T22:35:01.612290+00:00','2026-10-16T22:35:01.612290+00:00',0,'{"origin_declared": true, "caller": {"title": "Notes", "author": "Mara", "location": "12", "tags": []}, "ingest_key": "readwise:notes:bcb473067a1b", "ingest_content_hash": "195fe944f4a398a2a5a33aaa1bde03b8706b214e90e6f21e7e76eeab77c59f2c"}','extracted',1.0,0,'2026-10-16T22:35:01.612327+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,'readwise','import',NULL,'notes');
INSERT INTO "nodes" VALUES('5f3cbb81-b8b3-40cd-9d64-0cc09b293a51','entity','Notes','Notes','readwise:notes:bcb473067a1b','2026-10-16T22:35:01.613662+00:00','2026-10-16T22:35:01.619158+00:00',1,'{}','extracted',0.8,0,'2026-10-16T22:35:01.613669+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,'readwise','import',NULL,'notes');
INSERT INTO "nodes" VALUES('4db89123-1d02-4b9b-87c7-79ef3b7d94a4','context','We chose PostgreSQL for billing.','User: We chose PostgreSQL for billing.
Assistant: Noted, PostgreSQL for billing.','claude-code:proj-x:s1','2026-10-16T22:35:01.616807+00:00','2026-10-16T22:35:01.616807+00:00',0,'{"recorded_at_source": "content", "origin_declared": true, "caller": {"adapter": "claude_code", "project": "proj-x", "session_file": "s1.jsonl", "path": "/tmp/tmpm1lqe_56/projects/proj-x/s1.jsonl"}, "ingest_key": "claude-code:proj-x:s1", "ingest_content_hash": "f43d0b4bd9cf3554727e6250cd8889d70c527d425e1449e74d3f419b5726913d"}','extracted',1.0,0,'2026-10-16T22:35:01.616844+00:00','','',NULL,NULL,'text',1,0,'2026-03-05T10:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'claude-code','live','proj-x','s1');
INSERT INTO "nodes" VALUES('dd425246-a2bc-47e1-9f1a-f9a34e29c63d','entity','PostgreSQL','Technology/tool: PostgreSQL','claude-code:proj-x:s1','2026-10-16T22:35:01.616807+00:00','2026-10-16T22:35:01.616807+00:00',0,'{"recorded_at_source": "content", "origin_declared": true}','inferred',0.5,0,'2026-10-16T22:35:01.616888+00:00','','',NULL,NULL,'text',1,0,'2026-03-05T10:00:00+00:00',NULL,NULL,NULL,NULL,'',NULL,NULL,'claude-code','live','proj-x','s1');
INSERT INTO "nodes" VALUES('58ff1529-f28f-47cc-acb8-b2b5b204caab','context','I prefer tabs over spaces in Python.','I prefer tabs over spaces in Python.','mcp','2026-10-16T22:35:01.623998+00:00','2026-10-16T22:35:01.623998+00:00',0,'{}','extracted',1.0,0,'2026-10-16T22:35:01.624039+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "nodes" VALUES('0ecf48f1-356f-492a-a400-b87afbb0154d','entity','Python','Technology/tool: Python','mcp','2026-10-16T22:35:01.623998+00:00','2026-10-16T22:35:01.623998+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:35:01.624062+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "nodes" VALUES('25b5e56a-7f49-41d1-a27b-df44aa96ce63','preference','tabs over spaces in Python','I prefer tabs over spaces in Python.','mcp','2026-10-16T22:35:01.623998+00:00','2026-10-16T22:35:01.623998+00:00',0,'{}','inferred',0.5,0,'2026-10-16T22:35:01.624157+00:00','','',NULL,NULL,'text',1,0,NULL,NULL,NULL,NULL,NULL,'',NULL,NULL,NULL,NULL,NULL,NULL);
CREATE TABLE pending_embeds (node_id TEXT PRIMARY KEY, queued_at TEXT NOT NULL);
INSERT INTO "pending_embeds" VALUES('4eca7581-a23f-4e4f-975b-676bb479e788','2026-10-16T22:35:01.605224+00:00');
INSERT INTO "pending_embeds" VALUES('3c90296b-7ba6-418a-852d-e0a5b47b7179','2026-10-16T22:35:01.607986+00:00');
INSERT INTO "pending_embeds" VALUES('36d6ae80-2cac-48cf-9f60-b20815e16e14','2026-10-16T22:35:01.609760+00:00');
INSERT INTO "pending_embeds" VALUES('318b49ab-0318-48cf-b9e4-9157ee35a114','2026-10-16T22:35:01.612550+00:00');
INSERT INTO "pending_embeds" VALUES('5f3cbb81-b8b3-40cd-9d64-0cc09b293a51','2026-10-16T22:35:01.613752+00:00');
INSERT INTO "pending_embeds" VALUES('4db89123-1d02-4b9b-87c7-79ef3b7d94a4','2026-10-16T22:35:01.617312+00:00');
INSERT INTO "pending_embeds" VALUES('dd425246-a2bc-47e1-9f1a-f9a34e29c63d','2026-10-16T22:35:01.620559+00:00');
INSERT INTO "pending_embeds" VALUES('58ff1529-f28f-47cc-acb8-b2b5b204caab','2026-10-16T22:35:01.624291+00:00');
INSERT INTO "pending_embeds" VALUES('0ecf48f1-356f-492a-a400-b87afbb0154d','2026-10-16T22:35:01.626060+00:00');
INSERT INTO "pending_embeds" VALUES('25b5e56a-7f49-41d1-a27b-df44aa96ce63','2026-10-16T22:35:01.628427+00:00');
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
INSERT INTO "sync_cursors" VALUES('claude-code','2026-10-16T22:35:01.616171+00:00','2026-10-16T22:35:01.623246+00:00');
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
INSERT INTO "sqlite_sequence" VALUES('audit_log',10);
COMMIT;
PRAGMA user_version = 4;
