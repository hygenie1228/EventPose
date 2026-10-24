# Browser research workspace implementation log

## 2026-10-24

The user requested a server-hosted browser research environment with a left file explorer, a middle viewer/editor, and multiple Codex conversations. Implemented it in `apps/research-workbench/`, independently of motion training/evaluation. The server binds `0.0.0.0:18080` and runs as the user's `eventpose-workbench.service`.

The backend uses the signed-in Codex CLI through JSONL subprocess events, preserves each independent Codex thread, and resumes the matching thread for subsequent messages. The file API supports optimistic save conflicts, read-only datasets, symlink-aware path bounds, and byte-range media streaming. Existing training remains untouched.

Initial backend tests passed, including login, path bounds, save conflicts, dataset protection, byte ranges, independent chat persistence, and origin checking. Browser tests exercised file editing, Markdown, images, video play/seek, real Codex messages, and conversation resume. A separate real CLI tool invocation created a disposable fixture successfully.

An external-connectivity report revealed that server-local HTTP 200 did not prove remote access. UFW is enabled with DROP as the input default, while this account lacks passwordless sudo. The user was given the scoped `sudo ufw allow 18080/tcp` command. Do not claim external-client reachability without evidence.

The user then requested password login, larger text and close buttons, light/dark themes, two simultaneous chats, independent scrolling, visible message composers, and loading indicators. The clarified default layout is a full-height Codex panel on the right and a second Codex panel below the viewer, rather than two stacked in the right column. The bottom panel can be resized up to 50% of the workspace height. One/two-panel selection and theme selection use top-bar buttons. The configured password is stored only as a salted scrypt hash in private runtime state; the old session/access token was rotated.

The user requires all Agent records under `_agents/research/`, and reports under `_agents/report/`. Browser user, assistant, and tool events are now written to `_agents/research/YYMMDD/` as full JSONL and readable Markdown, including backfill of retained browser conversations. This file records implementation decisions for the primary agent. Training/evaluation progress checks remain limited to once every 30 minutes.

### Wider Codex panel and animated loading

The user requested more width for the right Codex pane and explicit loading animations. Increased the default right pane to 60% of the viewport and removed the former hard-coded 750px resize ceiling; the available workspace now determines the maximum. File/chat loading uses an animated spinner. Active Codex work adds a spinner, animated dots, and a pulsing status badge. No additional model request is needed to verify these presentation changes.

### Native goals, permissions, telemetry, and handoff

Moved the web application to `src/apps/research-workbench/`. Chat and AI Agent now have pane-owned tabs and both use real workspace-write CLI permissions. A real Chat file-write smoke test succeeded. Right-click file/directory rename and recoverable delete, chat renaming, hidden dotfiles, date-prefixed Korean reports, flat date-scoped Agent records, and newest-first recent reports were verified. The old nested agent_sessions folder was removed with records preserved.

Native goals use Codex app-server thread/goal APIs. A verification goal created/read a fixture and reached native `complete` with persisted usage accounting. Goal checkbox, state, pause/resume/clear, and service-restart continuation are connected. Account quota is fetched from the signed-in CLI without model inference; GPU telemetry is likewise direct system data. GPUSTAT follows compact gpustat-style rows without GPU names/temperatures. Refresh defaults the right Chat to 30%, while dragging can exceed half the viewport.

The user explicitly requested migration of the ongoing NLF comparison goal to the web AI Agent after goal verification. Full operating state and completion criteria are recorded in `nlf_goal_handoff.md`; training is adopted rather than restarted.

### Compact Goal controls and query-key login

Replaced the persistent Goal status area with a small status icon immediately left of the Goal checkbox. It opens a native dialog with objective, status, usage, and pause/resume/clear controls. The dialog closes with its close button, Escape, or an outside click; incoming goal events update the content without opening it. Browser verification used the existing handed-off NLF goal without pausing or modifying that goal.

Added the requested `?key=...` initial login alongside fragment login. The login secret is exchanged for the existing session cookie, removed from browser history/address, and other parameters such as the target AI Agent session are preserved. Browser verification confirmed login and authenticated API access with `?key=cvlab`.

### Laptop density and phone/one-panel mode switching

Reduced the top chrome to 32–40px and scaled typography, tabs, toolbars, margins and composers for smaller desktop widths/heights. On <=767px screens only one conversation area is visible, with Chat / AI Agent tabs; explorer, viewer, top bar and footer are hidden. The user's later request to switch Chat/AI Agent on phones supersedes the earlier Chat-only phone layout. In desktop one-panel mode the same tabs switch between independent panes. Two-panel mode restores Chat on the right and AI Agent below the viewer. Drafts are kept per pane and chat, including across iframe reparenting. Removed the agent-info banner rather than merely hiding its text.

Browser verification passed at 1600×1000, 1440×900, 1280×800, 1024×650, 390×844 and 375×667. Both phone modes and composers remained usable. A separate test verified one-panel switching, draft preservation and two-panel restoration without JavaScript errors. No model calls or training interruptions were needed for these UI tests.
