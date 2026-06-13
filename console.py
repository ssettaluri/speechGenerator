"""
Rich console output for the speech-generator harness.
Import and call these instead of bare print() throughout the codebase.
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box
from contextlib import contextmanager

console = Console()


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def print_guardrails_pass():
    console.print("  [bold green]✓[/] Guardrails [green]passed[/]")

def print_guardrails_block(rule: str, reason: str):
    console.print(Panel(
        f"[bold]{reason}[/]\n[dim]Rule: {rule}[/]",
        title="[bold red]⛔ Guardrails Blocked[/]",
        border_style="red",
    ))


# ---------------------------------------------------------------------------
# Corpus tool calls
# ---------------------------------------------------------------------------

def print_corpus_query(tool_input: dict, result_count: int):
    args = "  ".join(f"[cyan]{k}[/]=[yellow]{v!r}[/]" for k, v in tool_input.items())
    console.print(f"  [dim]📚 corpus query:[/] {args}  → [green]{result_count} result(s)[/]")

def print_corpus_cap_reached(limit: int):
    console.print(f"  [dim yellow]⚠ corpus query cap reached ({limit})[/]")


# ---------------------------------------------------------------------------
# Iteration progress
# ---------------------------------------------------------------------------

def print_iteration(iteration: int, max_iter: int, word_count: int,
                    desired: int, confidence: float, threshold: float,
                    corpus_queries: int, input_tokens: int,
                    output_tokens: int, latency_ms: float):

    conf_color = "green" if confidence >= threshold else "yellow"
    word_color = "green" if abs(word_count - desired) / desired < 0.10 else "yellow"

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("words",          f"[{word_color}]{word_count}[/] / {desired}")
    table.add_row("confidence",     f"[{conf_color}]{confidence:.2f}[/] (threshold {threshold})")
    table.add_row("corpus queries", str(corpus_queries))
    table.add_row("tokens",         f"{input_tokens} in / {output_tokens} out")
    table.add_row("latency",        f"{latency_ms / 1000:.1f}s")

    console.print(Panel(
        table,
        title=f"[bold]Iteration {iteration} / {max_iter}[/]",
        border_style="blue",
    ))


# ---------------------------------------------------------------------------
# Exit condition
# ---------------------------------------------------------------------------

def print_exit(reason: str):
    icon = "✅" if reason == "confidence_met" else "⏹"
    labels = {
        "confidence_met":       "[green]confidence threshold met[/]",
        "iterations_exhausted": "[yellow]max iterations reached[/]",
        "blocked":              "[red]blocked[/]",
    }
    console.print(f"\n  {icon}  Exit: {labels.get(reason, reason)}\n")


# ---------------------------------------------------------------------------
# Final speech
# ---------------------------------------------------------------------------

def print_speech(speech: str, word_count: int, truncated: bool):
    title = f"[bold green]Final Speech[/]  [dim]{word_count} words"
    if truncated:
        title += "  [yellow](truncated to length cap)[/]"
    console.print(Panel(speech, title=title, border_style="green", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Observability report
# ---------------------------------------------------------------------------

def print_report(report):
    # Header
    console.print(Rule(f"[bold]Observability Report[/]  [dim]run {report.run_id}[/]"))

    # Summary table
    summary = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    summary.add_column(style="bold dim", width=24)
    summary.add_column()

    success_str = "[green]yes[/]" if report.success else "[red]no[/]"
    summary.add_row("success",          success_str)
    summary.add_row("exit reason",      report.exit_reason)
    summary.add_row("alignment",        report.alignment)
    summary.add_row("topic",            report.topic)
    summary.add_row("desired length",   f"{report.desired_length_words} words")
    summary.add_row("final length",     f"{report.final_word_count} words")
    summary.add_row("final confidence", f"{report.final_confidence:.3f}")
    summary.add_row("truncated",        str(report.truncated))
    summary.add_row("iterations",       f"{report.iterations_used} / {report.max_iterations}")
    summary.add_row("total tokens",     f"{report.total_tokens}  ({report.total_input_tokens} in / {report.total_output_tokens} out)")
    summary.add_row("corpus queries",   str(report.total_corpus_queries))
    summary.add_row("corpus IDs used",  str(report.unique_corpus_ids))
    summary.add_row("total latency",    f"{report.total_latency_ms / 1000:.1f}s")
    summary.add_row("avg / iteration",  f"{report.avg_latency_per_iter_ms / 1000:.1f}s")
    console.print(summary)

    # Per-iteration breakdown
    if report.iterations:
        iter_table = Table(
            box=box.SIMPLE_HEAD,
            title="Per-iteration breakdown",
            title_style="bold",
        )
        iter_table.add_column("iter",       justify="right",  style="dim")
        iter_table.add_column("words",      justify="right")
        iter_table.add_column("confidence", justify="right")
        iter_table.add_column("corpus q",   justify="right")
        iter_table.add_column("tokens in",  justify="right")
        iter_table.add_column("tokens out", justify="right")
        iter_table.add_column("latency",    justify="right")

        for it in report.iterations:
            conf_style = "green" if it.confidence >= 0.80 else "yellow"
            iter_table.add_row(
                str(it.iteration),
                str(it.word_count),
                f"[{conf_style}]{it.confidence:.2f}[/]",
                str(it.corpus_queries),
                str(it.input_tokens),
                str(it.output_tokens),
                f"{it.latency_ms / 1000:.1f}s",
            )

        console.print(iter_table)

    console.print(Rule(style="dim"))


# ---------------------------------------------------------------------------
# Spinner context manager (wraps blocking calls)
# ---------------------------------------------------------------------------

@contextmanager
def spinner(message: str):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(message)
        yield
