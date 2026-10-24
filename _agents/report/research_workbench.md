# Research Workbench verification

- Address: `http://147.46.89.70:18080/` (server listens on `0.0.0.0`). Password login is configured in private runtime state. Do not use the old fragment access-key link.
- Layout: file explorer on the left, file/media viewer in the middle, Codex on the right, and a second independent Codex below the viewer by default. The bottom boundary is draggable up to half of the workspace height; the top toolbar switches one/two panels.
- Appearance: light theme by default, sun/moon theme controls synchronized across the two panes, 15–16px primary text, and 30px tab close controls.
- Interaction: independent file/chat scrolling, visible composers, file/chat loading indicators, Markdown preview, code editing/save, image/video preview, and stale-chat recovery.
- Real browser verification passed: password login, bottom-pane position, both composers visible, 16px chat inputs, synchronized themes, layout switching, large close controls, viewer/chat scrolling, and simultaneous real Codex replies from both panes. No browser JavaScript errors were observed.
- Isolated backend tests passed: authentication, path restrictions, save conflicts, dataset write protection, media byte ranges, independent chat persistence, and origin checks.
- Complete browser Agent user/assistant/tool events are recorded in `_agents/research/YYMMDD/agent_sessions/` as JSONL and readable Markdown. Implementation notes are in `_agents/research/261024/workbench_implementation.md`.
- Evidence: `exp/261024/research_workbench/usability-test.json`, `workbench-light.png`, `workbench-dark.png`, and earlier `browser-test.json`.
- Network limit: server-side IP HTTP responses and browser tests are verified. External-client access depends on UFW/network policy. UFW is enabled with DROP as the input default; opening TCP 18080 requires an administrator because this account has no passwordless sudo. The user was given `sudo ufw allow 18080/tcp`.

Motion training/evaluation continues independently and is inspected only every 30 minutes.
