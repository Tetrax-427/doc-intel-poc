## Table `documents`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `name` | `text` |  Nullable |
| `created_at` | `timestamp` |  Nullable |
| `summary` | `text` |  Nullable |
| `summary_short` | `text` |  Nullable |
| `doc_type` | `text` |  Nullable |
| `classification_confidence` | `float8` |  Nullable |
| `classification_data` | `jsonb` |  Nullable |
| `requires_review` | `bool` |  Nullable |
| `user_id` | `text` |  Nullable |
| `org_id` | `uuid` |  Nullable |
| `team_id` | `uuid` |  Nullable |
| `visibility` | `text` |  Nullable |

## Table `chunks`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `document_id` | `uuid` |  Nullable |
| `content` | `text` |  Nullable |
| `embedding` | `vector` |  Nullable |
| `metadata` | `jsonb` |  Nullable |
| `created_at` | `timestamp` |  Nullable |

## Table `chats`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `document_id` | `uuid` |  Nullable |
| `role` | `text` |  Nullable |
| `content` | `text` |  Nullable |
| `sources` | `jsonb` |  Nullable |
| `created_at` | `timestamp` |  Nullable |
| `user_id` | `text` |  Nullable |

## Table `usage_logs`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `call_type` | `text` |  Nullable |
| `model` | `text` |  Nullable |
| `prompt_chars` | `int4` |  Nullable |
| `response_chars` | `int4` |  Nullable |
| `estimated_tokens` | `int4` |  Nullable |
| `latency_ms` | `int4` |  Nullable |
| `created_at` | `timestamp` |  Nullable |

## Table `api_keys`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `name` | `text` |  |
| `key_hash` | `text` |  Unique |
| `key_prefix` | `text` |  |
| `is_active` | `bool` |  Nullable |
| `rate_limit` | `int4` |  Nullable |
| `calls_today` | `int4` |  Nullable |
| `last_reset` | `date` |  Nullable |
| `created_at` | `timestamp` |  Nullable |
| `user_id` | `text` |  Nullable |
| `status` | `text` |  |
| `grace_expires_at` | `timestamptz` |  Nullable |
| `org_id` | `uuid` |  Nullable |
| `scope` | `text` |  Nullable |

## Table `webhooks`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `name` | `text` |  |
| `url` | `text` |  |
| `events` | `_text` |  Nullable |
| `is_active` | `bool` |  Nullable |
| `secret` | `text` |  Nullable |
| `last_triggered` | `timestamp` |  Nullable |
| `fail_count` | `int4` |  Nullable |
| `created_at` | `timestamp` |  Nullable |
| `user_id` | `text` |  Nullable |

## Table `webhook_logs`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `webhook_id` | `uuid` |  Nullable |
| `event` | `text` |  Nullable |
| `payload` | `jsonb` |  Nullable |
| `response_status` | `int4` |  Nullable |
| `success` | `bool` |  Nullable |
| `created_at` | `timestamp` |  Nullable |

## Table `review_corrections`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `document_id` | `uuid` |  Nullable |
| `doc_type` | `text` |  Nullable |
| `field_name` | `text` |  Nullable |
| `original_value` | `text` |  Nullable |
| `corrected_value` | `text` |  Nullable |
| `action` | `text` |  Nullable |
| `evidence_used` | `text` |  Nullable |
| `reviewer_note` | `text` |  Nullable |
| `created_at` | `timestamp` |  Nullable |

## Table `lineage_logs`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `document_id` | `text` |  |
| `user_id` | `text` |  |
| `event_type` | `text` |  |
| `event_data` | `jsonb` |  |
| `duration_ms` | `int4` |  Nullable |
| `status` | `text` |  |
| `error_message` | `text` |  Nullable |
| `created_at` | `timestamptz` |  |
| `org_id` | `uuid` |  Nullable |
| `team_id` | `uuid` |  Nullable |

## Table `extraction_results`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `document_id` | `text` |  |
| `user_id` | `text` |  |
| `template_id` | `text` |  |
| `result` | `jsonb` |  |
| `created_at` | `timestamptz` |  |

## Table `llm_calls`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `user_id` | `text` |  |
| `document_id` | `text` |  Nullable |
| `session_id` | `text` |  Nullable |
| `call_type` | `text` |  |
| `provider` | `text` |  |
| `model` | `text` |  |
| `is_override` | `bool` |  |
| `is_stream` | `bool` |  |
| `used_fallback` | `bool` |  |
| `system_text` | `text` |  Nullable |
| `user_text` | `text` |  Nullable |
| `prompt_hash` | `text` |  Nullable |
| `success` | `bool` |  |
| `response_text` | `text` |  Nullable |
| `response_model_name` | `text` |  Nullable |
| `error_message` | `text` |  Nullable |
| `error_code` | `text` |  Nullable |
| `prompt_tokens` | `int4` |  Nullable |
| `completion_tokens` | `int4` |  Nullable |
| `total_tokens` | `int4` |  Nullable |
| `estimated_cost_usd` | `numeric` |  Nullable |
| `cache_hit` | `bool` |  |
| `cache_layer` | `text` |  Nullable |
| `latency_ms` | `int4` |  |
| `created_at` | `timestamptz` |  |
| `org_id` | `uuid` |  Nullable |
| `team_id` | `uuid` |  Nullable |

## Table `llm_cache`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `user_id` | `text` |  |
| `document_id` | `text` |  Nullable |
| `call_type` | `text` |  |
| `provider` | `text` |  |
| `model` | `text` |  |
| `cache_key` | `text` |  |
| `system_text` | `text` |  Nullable |
| `user_text` | `text` |  Nullable |
| `response_text` | `text` |  |
| `response_model_name` | `text` |  Nullable |
| `original_provider_call_id` | `uuid` |  Nullable |
| `original_cost_usd` | `numeric` |  Nullable |
| `hit_count` | `int4` |  |
| `last_hit_at` | `timestamptz` |  Nullable |
| `created_at` | `timestamptz` |  |
| `org_id` | `uuid` |  Nullable |
| `team_id` | `uuid` |  Nullable |

## Table `orgs`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `name` | `text` |  |
| `slug` | `text` |  Unique |
| `created_by` | `text` |  |
| `created_at` | `timestamptz` |  Nullable |
| `updated_at` | `timestamptz` |  Nullable |

## Table `teams`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `org_id` | `uuid` |  |
| `name` | `text` |  |
| `created_by` | `text` |  |
| `created_at` | `timestamptz` |  Nullable |
| `updated_at` | `timestamptz` |  Nullable |

## Table `org_members`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `org_id` | `uuid` |  |
| `user_id` | `text` |  |
| `role` | `text` |  |
| `can_read_team_documents` | `bool` |  Nullable |
| `can_read_all_usage` | `bool` |  Nullable |
| `status` | `text` |  Nullable |
| `joined_at` | `timestamptz` |  Nullable |
| `updated_at` | `timestamptz` |  Nullable |

## Table `team_members`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `team_id` | `uuid` |  |
| `org_id` | `uuid` |  |
| `user_id` | `text` |  |
| `role` | `text` |  |
| `can_read_member_personal_documents` | `bool` |  Nullable |
| `joined_at` | `timestamptz` |  Nullable |
| `updated_at` | `timestamptz` |  Nullable |

## Table `audit_logs`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `actor_id` | `text` |  |
| `actor_role` | `text` |  |
| `action` | `text` |  |
| `resource_type` | `text` |  |
| `resource_id` | `text` |  |
| `org_id` | `uuid` |  Nullable |
| `details` | `jsonb` |  Nullable |
| `ip_address` | `text` |  Nullable |
| `created_at` | `timestamptz` |  Nullable |

## Table `quotas`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `user_id` | `text` |  Nullable |
| `team_id` | `uuid` |  Nullable |
| `org_id` | `uuid` |  Nullable |
| `quota_type` | `text` |  |
| `limit_value` | `numeric` |  |
| `is_hard_limit` | `bool` |  Nullable |
| `set_by` | `text` |  |
| `created_at` | `timestamptz` |  Nullable |
| `updated_at` | `timestamptz` |  Nullable |

## Table `agent_runs`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `agent_name` | `text` |  |
| `task` | `text` |  |
| `input_data` | `jsonb` |  Nullable |
| `status` | `text` |  |
| `current_stage` | `text` |  Nullable |
| `state` | `jsonb` |  |
| `pending_questions` | `jsonb` |  Nullable |
| `result` | `jsonb` |  Nullable |
| `error` | `text` |  Nullable |
| `user_id` | `uuid` |  |
| `created_at` | `timestamptz` |  |
| `started_at` | `timestamptz` |  Nullable |
| `completed_at` | `timestamptz` |  Nullable |
| `name` | `text` |  Nullable |

## Table `agent_chat_messages`

### Columns

| Name | Type | Constraints |
|------|------|-------------|
| `id` | `uuid` | Primary |
| `run_id` | `uuid` |  |
| `role` | `text` |  |
| `content` | `text` |  |
| `user_id` | `uuid` |  |
| `created_at` | `timestamptz` |  |

## RLS Policies

### `lineage_logs`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can only see own lineage logs` | ALL | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

### `chats`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `allow_all` | ALL | public | PERMISSIVE | `true` | `true` |

### `usage_logs`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `allow_all` | ALL | public | PERMISSIVE | `true` | `true` |

### `api_keys`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `API key visibility policy` | SELECT | public | PERMISSIVE | `((user_id = (auth.uid())::text) OR ((scope = 'org'::text) AND (EXISTS ( SELECT 1    FROM org_members om   WHERE ((om.org_id = api_keys.org_id) AND (om.user_id = (auth.uid())::text) AND (om.role = 'org_admin'::text) AND (om.status = 'active'::text))))))` | — |
| `Users can only see own API keys` | ALL | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

### `webhooks`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `allow_all` | ALL | public | PERMISSIVE | `true` | `true` |

### `webhook_logs`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `allow_all` | ALL | public | PERMISSIVE | `true` | `true` |

### `review_corrections`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can only see own corrections` | ALL | public | PERMISSIVE | `((document_id)::text IN ( SELECT (documents.id)::text AS id    FROM documents   WHERE (documents.user_id = (auth.uid())::text)))` | — |

### `audit_logs`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users and org admins can see audit logs` | SELECT | public | PERMISSIVE | `((actor_id = (auth.uid())::text) OR (org_id IN ( SELECT org_members.org_id    FROM org_members   WHERE ((org_members.user_id = (auth.uid())::text) AND (org_members.role = 'org_admin'::text) AND (org_members.status = 'active'::text)))))` | — |

### `extraction_results`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Extraction results follow document visibility` | SELECT | public | PERMISSIVE | `(EXISTS ( SELECT 1    FROM documents d   WHERE (((d.id)::text = extraction_results.document_id) AND ((d.user_id = (auth.uid())::text) OR ((d.visibility = 'team'::text) AND (EXISTS ( SELECT 1            FROM team_members tm           WHERE ((tm.team_id = d.team_id) AND (tm.user_id = (auth.uid())::text))))) OR ((d.visibility = 'team'::text) AND (EXISTS ( SELECT 1            FROM org_members om           WHERE ((om.org_id = d.org_id) AND (om.user_id = (auth.uid())::text) AND (om.role = 'org_admin'::text) AND (om.can_read_team_documents = true) AND (om.status = 'active'::text))))) OR ((d.visibility = 'org'::text) AND (EXISTS ( SELECT 1            FROM org_members om           WHERE ((om.org_id = d.org_id) AND (om.user_id = (auth.uid())::text) AND (om.status = 'active'::text)))))))))` | — |
| `Users can only see own extraction results` | ALL | public | PERMISSIVE | `(document_id IN ( SELECT (documents.id)::text AS id    FROM documents   WHERE (documents.user_id = (auth.uid())::text)))` | — |

### `llm_calls`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can view their own llm_calls` | SELECT | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

### `quotas`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can see relevant quotas` | SELECT | public | PERMISSIVE | `((user_id = (auth.uid())::text) OR (team_id IN ( SELECT team_members.team_id    FROM team_members   WHERE (team_members.user_id = (auth.uid())::text))) OR (org_id IN ( SELECT org_members.org_id    FROM org_members   WHERE ((org_members.user_id = (auth.uid())::text) AND (org_members.status = 'active'::text)))))` | — |

### `llm_cache`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can view their own llm_cache entries` | SELECT | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

### `agent_runs`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can insert their own agent runs` | INSERT | public | PERMISSIVE | — | `(auth.uid() = user_id)` |
| `Users can update their own agent runs` | UPDATE | public | PERMISSIVE | `(auth.uid() = user_id)` | — |
| `Users can view their own agent runs` | SELECT | public | PERMISSIVE | `(auth.uid() = user_id)` | — |

### `agent_chat_messages`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Users can insert their own agent chat messages` | INSERT | public | PERMISSIVE | — | `(auth.uid() = user_id)` |
| `Users can view their own agent chat messages` | SELECT | public | PERMISSIVE | `(auth.uid() = user_id)` | — |

### `documents`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Document visibility policy` | SELECT | public | PERMISSIVE | `((user_id = (auth.uid())::text) OR ((visibility = 'team'::text) AND (EXISTS ( SELECT 1    FROM team_members tm   WHERE ((tm.team_id = documents.team_id) AND (tm.user_id = (auth.uid())::text))))) OR ((visibility = 'team'::text) AND (EXISTS ( SELECT 1    FROM org_members om   WHERE ((om.org_id = documents.org_id) AND (om.user_id = (auth.uid())::text) AND (om.role = 'org_admin'::text) AND (om.can_read_team_documents = true) AND (om.status = 'active'::text))))) OR ((visibility = 'org'::text) AND (EXISTS ( SELECT 1    FROM org_members om   WHERE ((om.org_id = documents.org_id) AND (om.user_id = (auth.uid())::text) AND (om.status = 'active'::text))))))` | — |
| `documents_delete` | DELETE | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |
| `documents_insert` | INSERT | public | PERMISSIVE | — | `(user_id = (auth.uid())::text)` |
| `documents_write` | UPDATE | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

### `chunks`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Chunks follow document visibility` | SELECT | public | PERMISSIVE | `(EXISTS ( SELECT 1    FROM documents d   WHERE ((d.id = chunks.document_id) AND ((d.user_id = (auth.uid())::text) OR ((d.visibility = 'team'::text) AND (EXISTS ( SELECT 1            FROM team_members tm           WHERE ((tm.team_id = d.team_id) AND (tm.user_id = (auth.uid())::text))))) OR ((d.visibility = 'team'::text) AND (EXISTS ( SELECT 1            FROM org_members om           WHERE ((om.org_id = d.org_id) AND (om.user_id = (auth.uid())::text) AND (om.role = 'org_admin'::text) AND (om.can_read_team_documents = true) AND (om.status = 'active'::text))))) OR ((d.visibility = 'org'::text) AND (EXISTS ( SELECT 1            FROM org_members om           WHERE ((om.org_id = d.org_id) AND (om.user_id = (auth.uid())::text) AND (om.status = 'active'::text)))))))))` | — |
| `Users can only see chunks of own documents` | ALL | public | PERMISSIVE | `((document_id)::text IN ( SELECT (documents.id)::text AS id    FROM documents   WHERE (documents.user_id = (auth.uid())::text)))` | — |

### `orgs`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Org members can see their org` | SELECT | public | PERMISSIVE | `(EXISTS ( SELECT 1    FROM org_members om   WHERE ((om.org_id = orgs.id) AND (om.user_id = (auth.uid())::text) AND (om.status = 'active'::text))))` | — |

### `teams`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `Org members can see org teams` | SELECT | public | PERMISSIVE | `(EXISTS ( SELECT 1    FROM org_members om   WHERE ((om.org_id = teams.org_id) AND (om.user_id = (auth.uid())::text) AND (om.status = 'active'::text))))` | — |

### `org_members`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `org_members_select` | SELECT | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

### `team_members`

| Policy | Command | Roles | Action | USING | WITH CHECK |
|--------|---------|-------|--------|-------|------------|
| `team_members_select` | SELECT | public | PERMISSIVE | `(user_id = (auth.uid())::text)` | — |

