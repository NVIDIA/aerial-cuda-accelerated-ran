# Aerial Report Refactor Mapping -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Purpose: Traceability map from legacy report set to consolidated `-codex` reports.

## Archive Policy
All legacy reports were preserved unchanged in:
- `archive/aerial-reports-v1.1-2026-02-11/`

Original files also remain in repository root.

## New Consolidated Reports
- `architecture-overview-codex.md`
- `control-plane-runtime-codex.md`
- `downlink-pdsch-pipeline-codex.md`
- `uplink-pusch-prach-pipeline-codex.md`
- `parallelism-threading-model-codex.md`
- `roles-responsibilities-codex.md`
- `migration-checklist-25-3-codex.md`

## Legacy -> Consolidated Mapping
## `architecture_analysis.md`
- primary destination: `architecture-overview-codex.md`
- secondary coverage: `control-plane-runtime-codex.md`, `parallelism-threading-model-codex.md`

## `MAC_ARCHITECTURE.md`
- primary destination: `architecture-overview-codex.md`
- secondary coverage: `control-plane-runtime-codex.md`

## `MULTITHREAD_PARALLEL_ARCHITECTURE-claude.md`
- primary destination: `parallelism-threading-model-codex.md`
- secondary coverage: `architecture-overview-codex.md`

## `multi_task_parallel_architecture.md`
- primary destination: `parallelism-threading-model-codex.md`
- secondary coverage: `architecture-overview-codex.md`

## `multi_task_parallel_architecture-v0.md`
- primary destination: `parallelism-threading-model-codex.md`
- secondary coverage: `architecture-overview-codex.md`

## `CUPHYCONTROLLER_ANALYSIS.md`
- primary destination: `control-plane-runtime-codex.md`
- secondary coverage: `roles-responsibilities-codex.md`

## `CUPHYCONTROLLER_ROLES.md`
- primary destination: `roles-responsibilities-codex.md`
- secondary coverage: `control-plane-runtime-codex.md`

## `CUPHYDRIVER_ANALYSIS.md`
- primary destination: `architecture-overview-codex.md`
- secondary coverage: `parallelism-threading-model-codex.md`

## `CUPHYDRIVER_ROLES.md`
- primary destination: `roles-responsibilities-codex.md`
- secondary coverage: `architecture-overview-codex.md`

## `PDSCH_PIPELINE.md`
- primary destination: `downlink-pdsch-pipeline-codex.md`
- secondary coverage: `architecture-overview-codex.md`

## `pdsch_processing_report.md`
- primary destination: `downlink-pdsch-pipeline-codex.md`

## `pdsch_precoding_analysis.md`
- primary destination: `downlink-pdsch-pipeline-codex.md`

## `pdsch_iq_bfp_report.md`
- primary destination: `downlink-pdsch-pipeline-codex.md`

## `UPLINK_PUSCH_PROCESSING_ARCHITECTURE.md`
- primary destination: `uplink-pusch-prach-pipeline-codex.md`
- secondary coverage: `parallelism-threading-model-codex.md`

## `PRACH_PROCESSING_CHAIN.md`
- primary destination: `uplink-pusch-prach-pipeline-codex.md`

## `UPLINK_MULTITHREAD_ARCHITECTURE.md`
- primary destination: `parallelism-threading-model-codex.md`
- secondary coverage: `uplink-pusch-prach-pipeline-codex.md`

## `UPGRADE_25_3_MIGRATION_CHECKLIST.md`
- primary destination: `migration-checklist-25-3-codex.md`

## Consolidation Rationale
1. Remove repeated architecture explanations scattered across many files.
2. Keep one canonical deep-dive per concern area.
3. Preserve all legacy context in archive and maintain direct traceability.
4. Improve maintainability for future version updates.

## Maintenance Guidance
When updating docs in future:
1. Update consolidated `-codex` reports first.
2. Keep this mapping file synchronized with any new split/merge decisions.
3. Preserve previous consolidated versions in archive for reproducible historical review.
