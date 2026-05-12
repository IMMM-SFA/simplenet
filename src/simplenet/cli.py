"""Command-line interface mirroring the MATLAB ``reduction_test.m`` workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from simplenet.case import PowerCase
from simplenet.io.csv_loader import load_excluded_nodes
from simplenet.io.matpower import load_m
from simplenet.io.xlsx import dump_xlsx, load_xlsx
from simplenet.pipeline import reduce_network


def _load_case(path: Path) -> PowerCase:
    """Auto-dispatch by extension."""

    suffix = path.suffix.lower()
    if suffix == ".m":
        return load_m(path)
    if suffix in (".xlsx", ".xlsm"):
        return load_xlsx(path)
    if suffix == ".json":
        data = json.loads(path.read_text())
        return PowerCase.from_pypower(data)
    raise click.UsageError(f"Unsupported case file extension: {suffix}")


@click.group()
@click.version_option()
def main() -> None:
    """simplenet - DC modified-Ward network reduction CLI."""


@main.command()
@click.argument("case_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("excluded_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o", "--output", "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write the reduced multi-sheet xlsx.",
)
@click.option(
    "--summary-txt",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to also dump the summary diary as a .txt file.",
)
@click.option(
    "--pf/--no-pf",
    default=True,
    help="Solve a DC PF on the full model before redistributing loads (default: --pf, matches Pf_flag=1).",
)
@click.option(
    "--gentype-xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="If the case file is a MATPOWER .m that delegates Gen/Bus/etc. data to an xlsx, point to it here.",
)
def reduce(
    case_path: Path,
    excluded_path: Path,
    output_path: Path,
    summary_txt: Path | None,
    pf: bool,
    gentype_xlsx: Path | None,
) -> None:
    """Run network reduction on CASE_PATH using buses listed in EXCLUDED_PATH.

    \b
    Examples
        simplenet reduce case9.m excluded.csv -o reduced.xlsx
        simplenet reduce matlab2_WECC.xlsx excluded_550.csv -o result.xlsx
    """

    case = _load_case(case_path)
    if gentype_xlsx is not None:
        sup = load_xlsx(gentype_xlsx)
        if sup.bus.size:
            case.bus = sup.bus
        if sup.gen.size:
            case.gen = sup.gen
        if sup.branch.size:
            case.branch = sup.branch
        if sup.gencost is not None:
            case.gencost = sup.gencost
        if sup.gentype is not None:
            case.gentype = sup.gentype
        if sup.genfuel is not None:
            case.genfuel = sup.genfuel
        if sup.bus_name is not None:
            case.bus_name = sup.bus_name

    excluded = load_excluded_nodes(excluded_path)

    click.echo(f"Loaded {case.n_bus()} buses, {case.n_branch()} branches, {case.n_gen()} gens", err=True)
    click.echo(f"Excluding {excluded.size} buses", err=True)

    result = reduce_network(case, excluded, pf_flag=pf)

    dump_xlsx(result.reduced_case, output_path, summary=result.summary)
    if summary_txt is not None:
        summary_txt.write_text(result.summary)

    click.echo(result.summary)


@main.command()
@click.argument("case_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(case_path: Path) -> None:
    """Print summary statistics for CASE_PATH."""

    case = _load_case(case_path)
    click.echo(f"baseMVA: {case.base_mva}")
    click.echo(f"buses:   {case.n_bus()}")
    click.echo(f"gens:    {case.n_gen()}")
    click.echo(f"branches:{case.n_branch()}")
    if case.gencost is not None:
        click.echo(f"gencost: {case.gencost.shape}")
    if case.gentype is not None:
        click.echo(f"gentype: {len(case.gentype)} entries")
    if case.genfuel is not None:
        click.echo(f"genfuel: {len(case.genfuel)} entries")
    if case.bus_name is not None:
        click.echo(f"bus_name: {len(case.bus_name)} entries")
    if case.dcline is not None:
        click.echo(f"dcline: {case.dcline.shape}")


if __name__ == "__main__":
    sys.exit(main())
