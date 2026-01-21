# AI Agent Guidelines

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

**Note for AI models**: The `bd` CLI is unavailable in this environment. AI models must **manually edit** the `.beads/issues.jsonl` file directly to add, update, or close issues, keeping it in sync with code changes.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start (AI models: manual edits)

Since `bd` is unavailable for models, edit `.beads/issues.jsonl` directly (one valid JSON object per line):
- **Check for ready work**: open `.beads/issues.jsonl` and look for issues without blockers and with a status of `ready`.
- **Create new issues**: add a new JSONL entry with the appropriate `type`, `priority`, and `deps` (e.g., `discovered-from:bd-123`).
- **Claim and update**: update the issue entry with `status: in_progress` or adjust `priority` as needed.
- **Complete work**: update the issue entry with `status: closed` and a `reason`.

Example:
{"id":"bd-123","type":"task","status":"ready","priority":2,"deps":[],"title":"..."}

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: open `.beads/issues.jsonl` and find issues with `status: ready` and no blockers
2. **Claim your task**: edit the issue entry to set `status: in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Add a new JSONL entry with `deps: ["discovered-from:<parent-id>"]`
5. **Complete**: update the issue entry with `status: closed` and a `reason`
6. **Commit together**: Always commit the `.beads/issues.jsonl` file together with the code changes so issue state stays in sync with code state

### Auto-Sync

bd automatically syncs with git when available, but since AI models cannot run `bd`, treat `.beads/issues.jsonl` as the source of truth and edit it directly.

### GitHub Copilot Integration

If using GitHub Copilot, also create `.github/copilot-instructions.md` for automatic instruction loading.
Run `bd onboard` to get the content (human developers), or see step 2 of the onboard instructions.
If `bd` isn’t available, copy the onboarding text from the instructions directly.

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, install the beads MCP server:

```bash
pip install beads-mcp
```

Add to MCP config (e.g., `~/.config/claude/config.json`):
```json
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

Then use `mcp__beads__*` functions instead of CLI commands.

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

**Best Practice: Use a dedicated directory for these ephemeral files**

**Recommended approach:**
- Create a `history/` directory in the project root
- Store ALL AI-generated planning/design docs in `history/`
- Keep the repository root clean and focused on permanent project files
- Only access `history/` when explicitly asked to review past planning

**Example .gitignore entry (optional):**
```
# AI planning documents (ephemeral)
history/
```

**Benefits:**
- ✅ Clean repository root
- ✅ Clear separation between ephemeral and permanent documentation
- ✅ Easy to exclude from version control if desired
- ✅ Preserves planning history for archeological research
- ✅ Reduces noise when browsing the project

### CLI Help

Run `bd <command> --help` to see all available flags for any command.
For example: `bd create --help` shows `--parent`, `--deps`, `--assignee`, etc.

### Important Rules

- ✅ Use beads issues for ALL task tracking
- ✅ **AI models** must edit `.beads/issues.jsonl` directly (no `bd` CLI)
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `.beads/issues.jsonl` before asking "what should I work on?"
- ✅ Store AI planning docs in `history/` directory
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems
- ❌ Do NOT clutter repo root with planning documents

For more details, see README.md and QUICKSTART.md.
