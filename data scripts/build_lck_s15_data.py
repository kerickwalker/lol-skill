from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from urllib.parse import quote, urljoin
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3


BASE_URL = "https://gol.gg"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://gol.gg/"}

TOURNAMENTS = [
    "LCK_2025_Rounds_1-2",
    "LCK_2025_Rounds_3-5",
    "LCK_2025_Road_to_MSI",
    "LCK_2025_Season_Play-In",
    "LCK_2025_Season_Playoffs",
    "LCK_Cup_2025",
]

OUT_ROOT = Path("data/LCK_S15_games")
OUT_BLOCKED = Path("data/lck_s15_games_blocked.csv")
OUT_PLAYER_AGG_CSV = Path("data/lck_s15_player_aggregated.csv")


def tournament_display_name(folder_name: str) -> str:
    return folder_name.replace("_", " ")


def get_html(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20, verify=False)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def parse_kda_series(series: pd.Series) -> pd.DataFrame:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace('^="', "", regex=True)
        .str.replace('"$', "", regex=True)
    )
    parts = cleaned.str.split("/", expand=True)
    parts.columns = ["kills", "deaths", "assists"]
    return parts.astype(float)


def normalize_stat_name(label: str) -> str:
    normalized = str(label).strip().lower()
    replacements = {
        "%": "pct",
        "@": "_at_",
        "+": "_plus_",
        "/": "_",
        "'": "",
        ".": "",
        "-": "_",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


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


def parse_match_table(html: str, page_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = None
    for candidate in soup.find_all("table"):
        caption = candidate.find("caption")
        caption_text = caption.get_text(" ", strip=True).lower() if caption else ""
        if "recent games" in caption_text:
            table = candidate
            break
    if table is None:
        return pd.DataFrame()

    rows: list[dict[str, str | int | None]] = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 11:
            continue

        game_link = tds[9].find("a", href=True)
        game_href = game_link["href"] if game_link else ""
        game_match = re.search(r"game/stats/(\d+)/", game_href)
        game_id = int(game_match.group(1)) if game_match else None

        rows.append(
            {
                "Champion": tds[0].get_text(" ", strip=True),
                "Result": tds[1].get_text(" ", strip=True),
                "Duration": tds[2].get_text(" ", strip=True),
                "KDA": tds[3].get_text(" ", strip=True),
                "CSM": tds[4].get_text(" ", strip=True),
                "DPM": tds[5].get_text(" ", strip=True),
                "KP%": tds[6].get_text(" ", strip=True),
                "Date": tds[8].get_text(" ", strip=True),
                "Game": tds[9].get_text(" ", strip=True),
                "Tournament": tds[10].get_text(" ", strip=True),
                "game_id": game_id,
                "game_url": urljoin(page_url, game_href) if game_href else "",
            }
        )

    return pd.DataFrame(rows)


def fetch_fullstats_for_game(game_id: int) -> pd.DataFrame:
    url = f"{BASE_URL}/game/stats/{game_id}/page-fullstats/"
    try:
        html = get_html(url)
    except requests.RequestException:
        return pd.DataFrame()

    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    table = max(tables, key=lambda candidate: (len(candidate), len(candidate.columns))).copy()
    if table.shape[1] < 11 or table.shape[0] < 3:
        return pd.DataFrame()

    labels = table.iloc[:, 0].astype(str).str.strip()
    player_names = table.iloc[0, 1:11].astype(str).str.strip().tolist()
    roles = table.iloc[1, 1:11].astype(str).str.strip().tolist()
    stat_rows = table.iloc[2:, :].reset_index(drop=True)

    records: list[dict[str, str | int]] = []
    for player_idx, player_name in enumerate(player_names, start=1):
        record: dict[str, str | int] = {
            "game_id": game_id,
            "player_name": player_name,
            "fullstats_role": roles[player_idx - 1],
        }
        for row_idx in range(len(stat_rows)):
            stat_name = normalize_stat_name(labels.iloc[row_idx + 2])
            record[stat_name] = stat_rows.iat[row_idx, player_idx]
        records.append(record)

    return pd.DataFrame(records)


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

    df = parse_match_table(html, url)
    if df.empty:
        return df

    needed = [
        "Champion",
        "Result",
        "Duration",
        "KDA",
        "CSM",
        "DPM",
        "KP%",
        "Date",
        "Game",
        "Tournament",
        "game_id",
        "game_url",
    ]
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
    enriched = df.copy()
    enriched[["kills", "deaths", "assists"]] = parse_kda_series(enriched["KDA"])
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
        tournament_df["game_id"] = pd.to_numeric(tournament_df["game_id"], errors="coerce").astype("Int64")
        tournament_df["CSM"] = pd.to_numeric(tournament_df["CSM"], errors="coerce")
        tournament_df["DPM"] = pd.to_numeric(tournament_df["DPM"], errors="coerce")
        tournament_df = tournament_df.drop_duplicates(
            subset=["player_id", "Date", "Game", "Tournament", "Duration", "Champion", "KDA", "game_id"],
            keep="first",
        ).copy()

        fullstats_frames: list[pd.DataFrame] = []
        unique_game_ids = [
            int(game_id)
            for game_id in sorted(tournament_df["game_id"].dropna().unique())
        ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(fetch_fullstats_for_game, game_id): game_id
                for game_id in unique_game_ids
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                fullstats_df = future.result()
                if not fullstats_df.empty:
                    fullstats_frames.append(fullstats_df)
                if done % 25 == 0:
                    print(f"  fetched full stats {done}/{len(unique_game_ids)} games")

        if fullstats_frames:
            fullstats = pd.concat(fullstats_frames, ignore_index=True)
            tournament_df = tournament_df.merge(
                fullstats,
                on=["game_id", "player_name"],
                how="left",
            )
            tournament_df["role"] = (
                tournament_df["role"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .fillna(tournament_df["fullstats_role"])
            )

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
                ["game_id", "Date", "Tournament", "Game", "Duration", "player_id", "player_name", "role", "Champion", "Result", "KDA", "CSM", "DPM", "KP%"]
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
        ["game_block_id", "game_id", "Date", "Tournament", "Game", "Duration", "player_id", "player_name", "role", "Champion", "Result", "KDA", "CSM", "DPM", "KP%"]
    ].copy()
    extra_columns = [
        column
        for column in merged.columns
        if column not in final.columns and column not in {"game_block_id_local", "tournament_folder", "game_url", "fullstats_role"}
    ]
    if extra_columns:
        final = pd.concat([final, merged[extra_columns]], axis=1)
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
