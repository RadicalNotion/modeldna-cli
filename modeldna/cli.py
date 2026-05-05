"""
modeldna CLI — `modeldna scan <model_id>`
"""
from __future__ import annotations
import argparse, json, sys

from .scan import scan


def _format_rich(result: dict) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
    except ImportError:
        print(json.dumps(result, indent=2, default=str))
        return

    console = Console()

    if "error" in result:
        console.print(f"[bold red]❌ Scan failed:[/bold red] {result['error']}")
        return

    v = result["verdict"]
    e = result["evidence"]
    m = result["metadata"]
    flags = v.get("flags", [])

    conf_color = {"HIGH": "green", "MODERATE": "yellow", "NONE": "red"}.get(v["confidence"], "red")
    console.print(Panel(
        f"[bold]{v['architecture']}[/bold]\n"
        f"Confidence: [{conf_color}]{v['confidence']}[/{conf_color}]  |  "
        f"📥 {m['downloads']:,} downloads  |  "
        f"Scanned in {result.get('elapsed_s', '?')}s",
        title=f"🧬 [bold cyan]{result['model_id']}[/bold cyan]",
        border_style="cyan",
    ))

    if e.get("base_matches"):
        t = Table("Base", "Confidence", "Score", "Evidence", box=box.SIMPLE)
        for bm in e["base_matches"][:3]:
            t.add_row(bm["name"], bm["confidence"], str(bm["score"]),
                      "; ".join(bm.get("evidence", [])))
        console.print(t)

    if flags:
        for f in flags:
            console.print(Panel(
                f["explanation"],
                title=f"[bold red]⚠ {f['type']}[/bold red]",
                border_style="red",
            ))
    else:
        console.print("[green]✅ No flags — no suspicious claims detected.[/green]")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="modeldna",
        description="The DNA test for AI models — verify provenance before you download.",
    )
    sub = p.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan", help="Scan a HuggingFace model")
    scan_p.add_argument("model_id", help="HuggingFace model ID or URL")
    scan_p.add_argument("--json", dest="as_json", action="store_true",
                        help="Output raw JSON")

    args = p.parse_args()

    if args.command == "scan":
        result = scan(args.model_id)
        if args.as_json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _format_rich(result)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
