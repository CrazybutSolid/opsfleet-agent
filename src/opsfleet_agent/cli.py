"""Interactive CLI: ``opsfleet --user alice``.

Slash commands are handled locally (no LLM). Everything else is a chat turn.
The loop catches every exception, so a failure prints a message, never a stack trace.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
import warnings

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import AgentService, TurnResult
from .config import load_settings
from .observability.tracing import aggregate
from .security.scope import UnknownUser, load_users
from .storage.preferences import PreferenceError

HELP = """\
**Ask anything about sales, products, customers and orders**, e.g.
- *What data do you have?*
- *Compare Calvin Klein and Levi's. Why do they perform differently?*
- *Monthly revenue for the last 6 months* → *and why did it jump in May?*
- *Create a Q3 report with insights and action items for Q4*
- *Delete all reports mentioning Levi's* / *Delete the reports we made in this conversation*

**Commands**
| command | what it does |
|---|---|
| `/reports [text]` | list your saved reports (optionally filtered) |
| `/report <id>` | show one saved report |
| `/prefs` · `/prefs set <key> <value>` | show / set preferences (format, depth, visuals, focus) |
| `/trace [id\\|last] [--full]` | inspect the trace of a turn (default: last) |
| `/metrics [--all]` | agent metrics over your turns (or everyone's) |
| `/feedback good\\|bad [comment]` | rate the last answer (good answers become golden-trio candidates) |
| `/audit` | your delete audit log |
| `/new` | start a new conversation · `/whoami` · `/help` · `/quit` |
"""


def _fmt_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


class ChatCLI:
    def __init__(self, service: AgentService, console: Console | None = None):
        self.svc = service
        self.console = console or Console()

    # -- rendering -------------------------------------------------------------

    def render_turn(self, result: TurnResult) -> None:
        style = {"refused": "yellow", "error": "red"}.get(result.outcome, "")
        if style:
            self.console.print(Panel(result.text, border_style=style))
        else:
            self.console.print(Markdown(result.text))
        if result.pending_deletion:
            plan = result.pending_deletion
            table = Table(title=f"About to permanently delete {len(plan.reports)} report(s): {plan.criteria}",
                          title_style="bold red", show_lines=False)
            table.add_column("id", justify="right")
            table.add_column("title")
            table.add_column("created")
            for r in plan.reports:
                table.add_row(str(r.id), r.title, _fmt_ts(r.created_at))
            self.console.print(table)
            secs = int(plan.expires_at - self.svc.reports.clock())
            self.console.print(
                f"[bold red]Type [white on red] confirm [/] within {secs}s to delete these. Anything else cancels.[/]"
            )
        t = self.svc.last_trace
        if t is not None:
            sql = f" · {t.sql_attempts} SQL" + (f" ({t.sql_failures} self-corrected)" if t.sql_failures else "") if t.sql_attempts else ""
            fb = " · fallback model" if t.fallback_used else ""
            self.console.print(
                f"[dim]trace {t.trace_id} · {t.outcome}{sql} · {t.latency_ms / 1000:.1f}s · {t.tokens['total']:,} tokens{fb}[/]"
            )

    # -- commands --------------------------------------------------------------

    async def command(self, line: str) -> bool:
        """Handle a slash command. Returns False to quit."""
        parts = line.strip().split()
        cmd, args = parts[0].lower(), parts[1:]
        svc, out = self.svc, self.console
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            out.print(Markdown(HELP))
        elif cmd == "/whoami":
            u = svc.user
            out.print(f"{u.display_name} ({u.role}) · scope: {u.describe()} · conversation {svc.session_id} · "
                      f"persona {svc.persona.version}")
        elif cmd == "/new":
            sid = await svc.new_conversation()
            out.print(f"[green]New conversation {sid}.[/]")
        elif cmd == "/reports":
            reports = svc.reports.list(svc.user.user_id, mentioning=" ".join(args) or None)
            table = Table(title=f"Your reports ({len(reports)})")
            for col in ("id", "title", "created", "conversation"):
                table.add_column(col)
            for r in reports:
                table.add_row(str(r.id), r.title, _fmt_ts(r.created_at),
                              r.session_id + (" (this)" if r.session_id == svc.session_id else ""))
            out.print(table)
        elif cmd == "/report" and args and args[0].isdigit():
            r = svc.reports.get(svc.user.user_id, int(args[0]))
            out.print(Markdown(f"# {r.title}\n\n{r.body}") if r else "[yellow]No such report in your library.[/]")
        elif cmd == "/prefs":
            if len(args) == 3 and args[0] == "set":
                try:
                    svc.prefs.set(svc.user.user_id, args[1], args[2])
                    out.print(f"[green]Saved {args[1]} = {args[2]}[/]")
                except PreferenceError as e:
                    out.print(f"[yellow]{e}[/]")
            else:
                stored = svc.prefs.get_all(svc.user.user_id)
                for k, v in svc.prefs.effective(svc.user.user_id).items():
                    src = stored.get(k, {}).get("source", "default")
                    out.print(f"{k} = [bold]{v}[/] [dim]({src})[/]")
        elif cmd == "/trace":
            full = "--full" in args
            ids = [a for a in args if not a.startswith("--")]
            t = svc.tracer.get(ids[0] if ids else "last", svc.user.user_id)
            if not t:
                out.print("[yellow]No trace found.[/]")
            else:
                self._print_trace(t, full)
        elif cmd == "/metrics":
            traces = svc.tracer.load(None if "--all" in args else svc.user.user_id)
            out.print_json(json.dumps(aggregate(traces)))
        elif cmd == "/feedback" and args and args[0] in ("good", "bad"):
            msg = svc.record_feedback(args[0], " ".join(args[1:]))
            out.print(f"[green]{msg}[/]")
        elif cmd == "/audit":
            table = Table(title="Report audit log (yours)")
            for col in ("time", "action", "reports", "detail"):
                table.add_column(col)
            for e in svc.reports.audit_log(svc.user.user_id)[-20:]:
                table.add_row(_fmt_ts(e["ts"]), e["action"], ",".join(map(str, e["report_ids"])), json.dumps(e["detail"]))
            out.print(table)
        else:
            out.print("[yellow]Unknown command. Type /help.[/]")
        return True

    def _print_trace(self, t: dict, full: bool) -> None:
        if full:
            self.console.print_json(json.dumps(t, default=str))
            return
        summary = {k: t.get(k) for k in ("trace_id", "outcome", "error", "user_message", "guard", "golden", "context",
                                         "latency_ms", "tokens", "llm_retries", "fallback_used", "sql_attempts",
                                         "sql_failures", "self_corrected", "pii_redactions")}
        summary["model_calls"] = [
            {k: c.get(k) for k in ("model", "attempt", "status", "latency_ms", "prompt_tokens", "output_tokens",
                                   "error_code", "response")} for c in t.get("model_calls", [])
        ]
        summary["tool_calls"] = t.get("tool_calls", [])
        summary["system_prompt_chars"] = (t.get("prompt") or {}).get("chars")
        self.console.print_json(json.dumps(summary, default=str))
        self.console.print("[dim]/trace <id> --full for the complete record (incl. system prompt and governed SQL).[/]")

    # -- loop ------------------------------------------------------------------

    async def run(self) -> None:
        await self.svc.new_conversation()
        u = self.svc.user
        self.console.print(Panel(
            f"[bold]Opsfleet Insights[/] · signed in as [bold]{u.display_name}[/] ({u.role})\n"
            f"Data scope: {u.describe()} · model {self.svc.settings.model} "
            f"(fallback {self.svc.settings.fallback_model}) · /help for commands",
            border_style="cyan",
        ))
        while True:
            try:
                line = await asyncio.to_thread(self.console.input, "[bold cyan]you ›[/] ")
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not await self.command(line):
                        break
                    continue
                with self.console.status("Analysing…", spinner="dots"):
                    result = await self.svc.chat(line)
                self.render_turn(result)
            except KeyboardInterrupt:
                self.console.print("[yellow]Interrupted.[/]")
            except Exception as e:  # the CLI never crashes
                self.console.print(f"[red]Unexpected error: {type(e).__name__}: {e}. You can keep chatting.[/]")
        self.console.print("Bye.")


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    settings = load_settings()
    users = sorted(load_users(settings.users_path))
    parser = argparse.ArgumentParser(prog="opsfleet", description="Opsfleet retail analytics chat agent")
    parser.add_argument("--user", "-u", required=True, help=f"who is chatting: {', '.join(users)}")
    parser.add_argument("--debug", action="store_true", help="verbose logs to stderr")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.ERROR, stream=sys.stderr)
    try:
        service = AgentService(settings, args.user)
    except UnknownUser as e:
        print(e, file=sys.stderr)
        return 2
    try:
        asyncio.run(ChatCLI(service).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
