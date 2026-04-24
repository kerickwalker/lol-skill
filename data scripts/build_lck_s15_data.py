from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from urllib.parse import quote
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3


BASE_URL = "https://gol.gg"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://gol.gg/"}

TOURNAMENTS = [
    "LCK_2025_Rounds_1-2",
    "LCK_2025_Rounds_3-5",
    "LCK_2025_Season_Playoffs",
    "LCK_Cup_2025",
]

OUT_ROOT = Path("data/LCK_S15_games")
OUT_BLOCKED = Path("data/lck_s15_games_blocked.csv")
OUT_PLAYER_AGG_CSV = Path("data/lck_s15_player_aggregated.csv")


def tournament_display_name(folder_name: str) -> str:
    return folder_name.replace("_", " ")


def get_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=20, verify=False)
    response.raise_for_status()
    return response.text


def parse_players_for_tournament(tournament_name: str) -> pd.DataFrame:
    url = (
        f"{BASE_URL}/players/list/season-ALL/split-ALL/"
        f"tournament-{quote(tournament_name)}/"
    )
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    seen = set()
    role_lookup: dict[str, str] = {}

    try:
        tables = pd.read_html(StringIO(html))
        if tables:
            player_table = max(tables, key=lambda table: (len(table), len(table.columns))).copy()
            normalized = {str(column).strip().lower(): column for column in player_table.columns}
            player_column = normalized.get("player")
            role_column = normalized.get("role")
            if player_column is not None and role_column is not None:
                role_lookup = (
                    player_table[[player_column, role_column]]
                    .dropna()
                    .assign(**{player_column: lambda frame: frame[player_column].astype(str).str.strip()})
                    .drop_duplicates(subset=[player_column], keep="first")
                    .set_index(player_column)[role_column]
                    .astype(str)
                    .str.strip()
                    .to_dict()
                )
    except ValueError:
        role_lookup = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        match = re.search(r"(?:^|/|\.\/)player-stats/(\d+)/", href)
        if not match:
            continue
        player_id = int(match.group(1))
        player_name = anchor.get_text(" ", strip=True)
        if not player_name:
            continue
        key = (player_id, player_name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "role": role_lookup.get(player_name, ""),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["player_id", "player_name", "role"])
    return pd.DataFrame(rows).drop_duplicates()


def read_match_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()
    return max(tables, key=lambda table: (len(table), len(table.columns))).copy()


def fetch_player_tournament_matches(
    player_id: int, player_name: str, role: str, tournament_name: str
) -> pd.DataFrame:
    url = (
        f"{BASE_URL}/players/player-matchlist/{player_id}/"
        f"season-ALL/split-ALL/tournament-{quote(tournament_name)}/"
    )
    try:
        html = get_html(url)
    except requests.RequestException:
        return pd.DataFrame()

    df = read_match_table(html)
    if df.empty:
        return df

    needed = ["Champion", "Result", "Duration", "KDA", "CSM", "DPM", "KP%", "Date", "Game", "Tournament"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df[needed].copy()
    out["player_id"] = player_id
    out["player_name"] = player_name
    out["role"] = role
    return out


def parse_duration_minutes(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) + int(seconds) / 60.0


def make_excel_safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    safe_df["KDA"] = safe_df["KDA"].astype(str).str.strip().map(lambda value: f'="{value}"')
    return safe_df


def build_block_spaced_sheet(df: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for _, group in df.groupby("game_block_id", sort=True):
        blocks.append(group)
        blocks.append(pd.DataFrame([{column: "" for column in df.columns}]))
    return pd.concat(blocks, ignore_index=True)


def build_player_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    kda_parts = df["KDA"].astype(str).str.strip().str.split("/", expand=True)
    kda_parts.columns = ["kills", "deaths", "assists"]
    enriched = df.copy()
    enriched[["kills", "deaths", "assists"]] = kda_parts.astype(float)
    enriched["is_win"] = enriched["Result"].eq("Victory").astype(int)
    enriched["kp_pct"] = enriched["KP%"].astype(str).str.rstrip("%").astype(float) / 100.0
    enriched["duration_mins"] = enriched["Duration"].astype(str).map(parse_duration_minutes)
    enriched["kda_ratio_calc"] = (
        (enriched["kills"] + enriched["assists"]) / enriched["deaths"].replace(0, pd.NA)
    )
    enriched["series_name"] = enriched["Game"].astype(str).str.strip()
    if "Date" in enriched.columns:
        enriched["Date"] = pd.to_datetime(enriched["Date"], errors="coerce")

    grouped = (
        enriched.groupby(["player_id", "player_name", "role"], as_index=False)
        .agg(
            games=("game_block_id", "count"),
            wins=("is_win", "sum"),
            avg_kills=("kills", "mean"),
            avg_deaths=("deaths", "mean"),
            avg_assists=("assists", "mean"),
            avg_kda_ratio=("kda_ratio_calc", "mean"),
            avg_csm=("CSM", "mean"),
            avg_dpm=("DPM", "mean"),
            avg_kp=("kp_pct", "mean"),
            first_game_date=("Date", "min"),
            last_game_date=("Date", "max"),
            unique_tournaments=("Tournament", "nunique"),
        )
        .copy()
    )

    grouped["losses"] = grouped["games"] - grouped["wins"]
    grouped["win_rate"] = grouped["wins"] / grouped["games"]

    ordered_columns = [
        "player_id",
        "player_name",
        "role",
        "games",
        "wins",
        "losses",
        "win_rate",
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_kda_ratio",
        "avg_csm",
        "avg_dpm",
        "avg_kp",
        "first_game_date",
        "last_game_date",
        "unique_tournaments",
    ]
    grouped = grouped[ordered_columns].sort_values(["games", "win_rate"], ascending=[False, False])

    for column in [
        "win_rate",
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_kda_ratio",
        "avg_csm",
        "avg_dpm",
        "avg_kp",
    ]:
        grouped[column] = grouped[column].round(4)

    return grouped.reset_index(drop=True)


def main() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for folder_name in TOURNAMENTS:
        tournament_name = tournament_display_name(folder_name)
        tournament_dir = OUT_ROOT / folder_name
        tournament_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {tournament_name} ===")

        players = parse_players_for_tournament(tournament_name)
        print(f"Players found: {len(players)}")
        if players.empty:
            continue

        jobs = [
            (int(row.player_id), str(row.player_name), str(row.role), tournament_name)
            for row in players.itertuples(index=False)
        ]
        tournament_frames: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(fetch_player_tournament_matches, player_id, player_name, role, requested_tournament): (
                    player_id,
                    player_name,
                )
                for player_id, player_name, role, requested_tournament in jobs
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                df = future.result()
                if not df.empty:
                    tournament_frames.append(df)
                if done % 15 == 0:
                    print(f"  fetched {done}/{len(jobs)} players")

        if not tournament_frames:
            print("  no rows fetched")
            continue

        tournament_df = pd.concat(tournament_frames, ignore_index=True)
        tournament_df["Date"] = pd.to_datetime(tournament_df["Date"], errors="coerce")
        tournament_df = tournament_df.drop_duplicates(
            subset=["player_id", "Date", "Game", "Tournament", "Duration", "Champion", "KDA"],
            keep="first",
        ).copy()

        game_key = ["Date", "Tournament", "Game", "Duration"]
        tournament_df["game_block_id_local"] = (
            tournament_df.groupby(game_key, dropna=False).ngroup().add(1).astype(int)
        )

        kept_blocks = 0
        dropped_blocks = 0
        for gid, block in tournament_df.groupby("game_block_id_local", sort=True):
            if len(block) != 10:
                dropped_blocks += 1
                continue
            kept_blocks += 1
            block_out = block[
                ["Game", "Duration", "player_id", "player_name", "Champion", "Result", "KDA", "CSM", "DPM", "KP%"]
            ].copy()
            out_file = tournament_dir / f"game_{gid:04d}.csv"
            block_out.to_csv(out_file, index=False)

        print(f"  complete games saved: {kept_blocks}")
        print(f"  incomplete games dropped: {dropped_blocks}")

        complete = tournament_df.groupby("game_block_id_local").filter(lambda group: len(group) == 10).copy()
        if not complete.empty:
            complete["tournament_folder"] = folder_name
            all_frames.append(complete)

    if not all_frames:
        raise RuntimeError("No complete games produced.")

    merged = pd.concat(all_frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.sort_values(["Date", "Tournament", "Game", "Duration", "player_name"]).reset_index(drop=True)

    game_key = ["Date", "Tournament", "Game", "Duration"]
    merged["game_block_id"] = merged.groupby(game_key, dropna=False).ngroup().add(1).astype(int)

    final = merged[
        ["game_block_id", "Game", "Duration", "player_id", "player_name", "role", "Champion", "Result", "KDA", "CSM", "DPM", "KP%"]
    ].copy()
    final.to_csv(OUT_BLOCKED, index=False)

    final = make_excel_safe_copy(final)
    final.to_csv(OUT_BLOCKED, index=False)

    player_aggregate = build_player_aggregate(merged)
    player_aggregate.to_csv(OUT_PLAYER_AGG_CSV, index=False)

    counts = merged.groupby("game_block_id").size()
    print("\n=== Final ===")
    print(f"Saved blocked CSV: {OUT_BLOCKED}")
    print(f"Saved player aggregate CSV: {OUT_PLAYER_AGG_CSV}")
    print(f"Game blocks: {counts.shape[0]}")
    print(f"Rows: {len(merged)}")
    print(f"All blocks have 10 rows: {bool((counts == 10).all())}")


if __name__ == "__main__":
    main()
