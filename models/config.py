"""Shared feature and role definitions for LCK S15 models."""

ROLE_MAP = {"TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "SUPPORT": 4}
ROLES = ["top", "jng", "mid", "adc", "sup"]
N_ROLES = len(ROLES)

INDIVIDUAL_STATS = [
    "level",
    "kills",
    "deaths",
    "assists",
    "cs",
    "golds",
    "vision_score",
    "solo_kills",
    "double_kills",
    "triple_kills",
    "quadra_kills",
    "penta_kills",
    "gd_at_15",
    "csd_at_15",
    "xpd_at_15",
    "objectives_stolen",
    "damage_dealt_to_buildings",
    "total_heal",
    "total_heals_on_teammates",
    "damage_self_mitigated",
    "total_damage_shielded_on_teammates",
    "total_time_cc_dealt",
    "total_damage_taken",
    "total_time_spent_dead",
    "shutdown_bounty_collected",
    "shutdown_bounty_lost",
    "total_damage_to_champion",
]
