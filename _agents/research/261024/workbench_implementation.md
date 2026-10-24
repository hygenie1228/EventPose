# Browser research workspace implementation log

## 2026-10-24

The user requested a server-hosted browser research environment with a left file explorer, a middle viewer/editor, and multiple Codex conversations. Implemented it in `apps/research-workbench/`, independently of motion training/evaluation. The server binds `0.0.0.0:18080` and runs as the user's `eventpose-workbench.service`.

The backend uses the signed-in Codex CLI through JSONL subprocess events, preserves each independent Codex thread, and resumes the matching thread for subsequent messages. The file API supports optimistic save conflicts, read-only datasets, symlink-aware path bounds, and byte-range media streaming. Existing training remains untouched.

Initial backend tests passed, including login, path bounds, save conflicts, dataset protection, byte ranges, independent chat persistence, and origin checking. Browser tests exercised file editing, Markdown, images, video play/seek, real Codex messages, and conversation resume. A separate real CLI tool invocation created a disposable fixture successfully.

An external-connectivity report revealed that server-local HTTP 200 did not prove remote access. UFW is enabled with DROP as the input default, while this account lacks passwordless sudo. The user was given the scoped `sudo ufw allow 18080/tcp` command. Do not claim external-client reachability without evidence.

The user then requested password login, larger text and close buttons, light/dark themes, two simultaneous chats, independent scrolling, visible message composers, and loading indicators. The clarified default layout is a full-height Codex panel on the right and a second Codex panel below the viewer, rather than two stacked in the right column. The bottom panel can be resized up to 50% of the workspace height. One/two-panel selection and theme selection use top-bar buttons. The configured password is stored only as a salted scrypt hash in private runtime state; the old session/access token was rotated.

The user requires all Agent records under `_agents/research/`, and reports under `_agents/report/`. Browser user, assistant, and tool events are now written to `_agents/research/YYMMDD/agent_sessions/` as full JSONL and readable Markdown, including backfill of retained browser conversations. This file records implementation decisions for the primary agent. Training/evaluation progress checks remain limited to once every 30 minutes.

### Wider Codex panel and animated loading

The user requested more width for the right Codex pane and explicit loading animations. Increased the default right pane to 60% of the viewport and removed the former hard-coded 750px resize ceiling; the available workspace now determines the maximum. File/chat loading uses an animated spinner. Active Codex work adds a spinner, animated dots, and a pulsing status badge. No additional model request is needed to verify these presentation changes.
