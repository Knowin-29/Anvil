# Post-release weight cleanup, prepared but not executed

Cleanup requires a completed accepted coordinate-v2 release, explicit native/public/general quality approval, all35 key checkpoints remotely verified, exact pinned repository inventory, idle campaign jobs and writers, and unchanged local payload identities. It refuses unaccepted or incomplete releases, active readers, symlink escapes, missing checkpoint state, altered evidence and ambiguous missing files.

The default is a dry run. Only named model/optimizer/scheduler/RNG payloads from completed campaign stages and uploaded package/merged copies are eligible. Unuploaded non-key saves are eligible only when their completed stage's final checkpoint is remotely verified. Metadata, logs, data, bases, teachers, source and Codex history remain. Each deletion is journaled. The prepared commands and source hashes are in cleanup_policy.json; no real cleanup has been executed.

Nineteen CPU tests use temporary files only, including live reader/writer refusal. The initial test fixture incorrectly included the real running publisher; the guard correctly refused cleanup. The fixture now isolates its control directory and independently tests that guard.
