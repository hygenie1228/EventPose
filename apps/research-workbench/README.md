# EventPose Research Workbench

A browser workspace with a file explorer, editable file/media viewer, and multiple independent Codex conversations. The UI and backend run on this server. It uses the existing `codex` executable and its signed-in account; no API key is entered in the browser.

## Run

Requires Node.js 22+, npm, and a working `codex login status` for the server user.

```bash
cd apps/research-workbench
npm ci
npm run build
HOST=0.0.0.0 PORT=18080 npm start
```

Open `http://SERVER_IP:18080/`. The first start generates an access key at `.experiment_storage/research_workbench/access-key` under the workspace. Paste that key into the login screen, or open `http://SERVER_IP:18080/#key=YOUR_KEY`. The fragment is exchanged for an HttpOnly session cookie and removed from browser history. Credentials used by Codex remain on the server.

The deployed user service is `eventpose-workbench.service`:

```bash
systemctl --user status eventpose-workbench
systemctl --user restart eventpose-workbench
journalctl --user -u eventpose-workbench -n 30
```

Its configuration is installed in `~/.config/systemd/user/eventpose-workbench.service`. `HOST`, `PORT` (>= 10000), `WORKBENCH_ROOT`, `WORKBENCH_STATE`, and `CODEX_BIN` are configurable. Avoid browser-blocked ports such as 10080; the deployment uses **18080**. Port access from other machines also depends on the network's routing/firewall.

## Workspace

- Left: lazily loaded server file tree, including the project's data/experiment symlinks. Refresh and filter loaded file names; expand directories to browse further.
- Middle: multiple file tabs, CodeMirror editing and save (`Ctrl/Cmd+S`), Markdown source/preview toggle, images, video/audio with seeking, PDF, downloads. Saving uses a content hash to detect edits made elsewhere.
- Right: create independent chats with **+ 새 채팅**, switch tabs, rename with a tab's context menu, stream replies/tool results, stop a running request, and attach the selected file as context. Up to four chats may run concurrently. Each chat resumes its own Codex thread, and history persists across browser/server restarts.
- Drag column boundaries to resize. `Ctrl/Cmd+K` focuses the explorer filter.
- `dataset/`, `orig_dataset/`, `etri_processed/`, and `data/` are read-only through the editor. External experiment symlinks can be viewed/downloaded but are not edited through the file API. Git internals, Codex credentials and private workbench state are not exposed through the explorer.
- Browser-supported media is streamed as-is with byte ranges. Playback depends on the source codec and browser (e.g. H.264 MP4 and WebM); scientific arrays/checkpoints are offered as downloads.
- The existing CLI executes from the workspace with `workspace-write` and `approval_policy=never`, network access enabled, and the resolved experiment output directory added as a writable root. It can read project files and use tools; it does not create GPU work until instructed. Ongoing training/evaluation should be checked only every 30 minutes, per `_agents/instruct.md`.

## Implementation and verification

`server.mjs` provides authenticated file/chat/media APIs, persistent histories and per-chat CLI processes. `client/app.js` is bundled to `public/app.js`; static dependencies are served locally. Fonts have system fallbacks.

The integration follows [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive/): `codex exec --json` starts a thread; `codex exec ... resume THREAD_ID -` continues that exact conversation. The installed CLI's `--help` is checked against the local version.

```bash
npm test                         # isolated backend tests; no Codex tokens
node test/browser.mjs           # deployed browser E2E; uses actual CLI chats
```

Browser tests require Chromium (`node_modules/.bin/playwright install chromium`) and the running server. Test fixtures are tiny generated files in `test/fixtures/` (ignored by Git); see the fixture preparation in `test/prepare-fixtures.mjs`. Browser screenshots and results go to `exp/261024/research_workbench/`. The Playwright development dependency is pinned to a Chromium version compatible with this server's Ubuntu 18.04.
