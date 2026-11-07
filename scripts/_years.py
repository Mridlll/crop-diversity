"""
_years.py

The one place the analysis window is set. Every multi-year district average on the
site is taken over these years and no others.

1997-98, the first year in the raw APY file, is left out. That year is not usable:
West Bengal reports 50.8 million hectares of rabi niger seed, Odisha reports no kharif
season, Chhattisgarh's kharif area is 0.48 million hectares against 4.56 the next year,
several states (Tamil Nadu among them) are missing, and twelve states report crop lists
about half as long as in any later year (outputs/revision_nov8/d0_results.md).
2020-21 is a partial stub and is left out too (script 76, defect 1).

Usage, after `year_start` has been parsed from the APY `year` column:

    from _years import keep_years, FIRST_YEAR, LAST_YEAR, N_YEARS, WINDOW_LABEL
    df = keep_years(df)
"""
FIRST_YEAR = 1998          # 1998-99
LAST_YEAR = 2019           # 2019-20
N_YEARS = LAST_YEAR - FIRST_YEAR + 1
WINDOW_LABEL = "{}-{} to {}-{}".format(FIRST_YEAR, str(FIRST_YEAR + 1)[-2:],
                                       LAST_YEAR, str(LAST_YEAR + 1)[-2:])


def keep_years(df, col="year_start"):
    """Rows whose agricultural year starts in FIRST_YEAR..LAST_YEAR."""
    return df[df[col].between(FIRST_YEAR, LAST_YEAR)]
