"""One daemon job owns the full official curation stage sequence."""


def cmd_curation_stage(extra: list[str]) -> None:
    from scripts.curation.runner import main

    main(extra)
