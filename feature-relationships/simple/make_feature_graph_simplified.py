from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from html import escape

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "lck_s15_games_MODEL-READY.csv"
OUT_GROUPED_HTML = BASE / "lck_s15_feature_relationship_graph_simplified_grouped.html"
OUT_FLOWY_HTML = BASE / "lck_s15_feature_relationship_graph_simplified_flowy.html"
OUT_EDGES = BASE / "lck_s15_feature_relationship_edges_simplified.csv"
OUT_NODES = BASE / "lck_s15_feature_relationship_nodes_simplified.csv"
OUT_DISCUSSION = BASE / "feature_relationship_scale_discussion_points.md"

EXCLUDED_COLUMNS = {"lvld_at_15"}
DISPLAY_RENAMES = {
    "kills_diff_vs_role_opp": "kills_diff",
    "deaths_diff_vs_role_opp": "deaths_diff",
    "assists_diff_vs_role_opp": "assists_diff",
    "cs_diff_vs_role_opp": "cs_diff",
    "golds_diff_vs_role_opp": "golds_diff",
    "vision_diff_vs_role_opp": "vision_diff",
    "damage_diff_vs_role_opp": "damage_diff",
}


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))

columns = read_header(CSV_PATH)
start_idx = columns.index("Result")
visible_columns = [c for c in columns[start_idx:] if c not in EXCLUDED_COLUMNS]
visible_set = set(visible_columns)

CORE_RAW_STATS = {"total_damage_to_champion", "level", "kills", "deaths", "assists", "cs", "golds", "vision_score"}
MULTI_KILLS = {"double_kills", "triple_kills", "quadra_kills", "penta_kills"}
TEAM_CONTEXT = {"team_kills", "team_deaths", "team_assists", "team_cs", "team_golds", "team_vision_score", "team_total_damage_to_champion"}
OPPONENT_DIFF = {
    "gd_at_15", "csd_at_15", "xpd_at_15",
    "kills_diff_vs_role_opp", "deaths_diff_vs_role_opp", "assists_diff_vs_role_opp",
    "cs_diff_vs_role_opp", "golds_diff_vs_role_opp", "vision_diff_vs_role_opp", "damage_diff_vs_role_opp",
}

GROUP_STYLES = {
    "game_info": {"background": "#fdae61", "border": "#d73027", "font": "#111111", "size": 24},
    "raw_stats": {"background": "#fee090", "border": "#e6ab02", "font": "#111111", "size": 21},
    "additional_stats": {"background": "#f1eef6", "border": "#8856a7", "font": "#111111", "size": 18},
    "multi_kills": {"background": "#fbb4ae", "border": "#e41a1c", "font": "#111111", "size": 19},
    "team_context": {"background": "#abd9e9", "border": "#2c7bb6", "font": "#111111", "size": 21},
    "opponent_diff": {"background": "#d9ef8b", "border": "#66a61e", "font": "#111111", "size": 21},
}
GROUP_CENTERS = {
    "game_info": (-560, -340),
    "raw_stats": (-560, 0),
    "additional_stats": (-560, 340),
    "multi_kills": (0, 340),
    "team_context": (520, -180),
    "opponent_diff": (520, 220),
}
EDGE_STYLES = {
    "directly_increases": {"color": "#d73027", "width": 3, "dashes": False, "arrows": "to", "label": "directly increases"},
    "component_of": {"color": "#4575b4", "width": 2, "dashes": [4, 4], "arrows": "", "label": "component of"},
    "precondition": {"color": "#984ea3", "width": 2.5, "dashes": [8, 5], "arrows": "to", "label": "precondition"},
}


def group_for(node: str) -> str:
    if node in {"Result", "Duration"}:
        return "game_info"
    if node in TEAM_CONTEXT or node.startswith("team_"):
        return "team_context"
    if node in OPPONENT_DIFF or node.endswith("_at_15") or node.endswith("_diff_vs_role_opp"):
        return "opponent_diff"
    if node in MULTI_KILLS:
        return "multi_kills"
    if node in CORE_RAW_STATS:
        return "raw_stats"
    return "additional_stats"

raw_edges: list[tuple[str, str, str, str]] = []

def add(src: str, dst: str, kind: str, note: str = "") -> None:
    raw_edges.append((src, dst, kind, note))

for src in ["cs", "kills", "assists", "shutdown_bounty_collected"]:
    add(src, "golds", "directly_increases", f"{src} directly contributes to player gold")
for src in ["team_cs", "team_kills", "team_assists"]:
    add(src, "team_golds", "directly_increases", f"{src} directly contributes to team gold")
add("csd_at_15", "gd_at_15", "directly_increases", "CS difference contributes to gold difference at 15 minutes")
for src in ["cs_diff_vs_role_opp", "kills_diff_vs_role_opp", "assists_diff_vs_role_opp"]:
    add(src, "golds_diff_vs_role_opp", "directly_increases", f"{src} contributes to same-role gold difference")
for src, dst in [
    ("kills", "team_kills"), ("deaths", "team_deaths"), ("assists", "team_assists"),
    ("cs", "team_cs"), ("golds", "team_golds"), ("vision_score", "team_vision_score"),
    ("total_damage_to_champion", "team_total_damage_to_champion"), ("total_heals_on_teammates", "total_heal"),
]:
    add(src, dst, "component_of", f"{src} is one component of {dst}")
add("deaths", "total_time_spent_dead", "directly_increases", "More deaths directly create more death timers")
for dst in ["shutdown_bounty_collected", "solo_kills", "double_kills", "triple_kills", "quadra_kills", "penta_kills"]:
    add("kills", dst, "precondition", f"A kill is required for {dst}, but not every kill creates one")

edges = []
seen = set()
for src, dst, kind, note in raw_edges:
    if src in visible_set and dst in visible_set and src != dst:
        key = (src, dst, kind)
        if key not in seen:
            seen.add(key)
            edges.append({"source": src, "target": dst, "kind": kind, "note": note})

groups_to_nodes: dict[str, list[str]] = {g: [] for g in GROUP_STYLES}
for node in visible_columns:
    groups_to_nodes[group_for(node)].append(node)
positions: dict[str, tuple[float, float]] = {}
for group, group_nodes in groups_to_nodes.items():
    cx, cy = GROUP_CENTERS[group]
    n = max(1, len(group_nodes))
    radius = 60 if n <= 3 else 95 + 4 * min(n, 18)
    for i, node in enumerate(group_nodes):
        angle = 2 * math.pi * i / n
        positions[node] = (cx + radius * math.cos(angle), cy + 0.72 * radius * math.sin(angle))


def build_nodes(layout: str) -> list[dict]:
    out = []
    for node in visible_columns:
        group = group_for(node)
        degree = sum(1 for e in edges if e["source"] == node or e["target"] == node)
        item = {
            "id": node,
            "label": DISPLAY_RENAMES.get(node, node),
            "group": group,
            "title": f"{escape(node)}<br>label: {escape(DISPLAY_RENAMES.get(node, node))}<br>group: {escape(group.replace('_', ' '))}<br>degree: {degree}",
            "value": max(1, degree + 1),
        }
        x, y = positions[node]
        if layout == "grouped":
            item.update({"x": round(x, 1), "y": round(y, 1), "physics": False, "fixed": {"x": True, "y": True}})
        else:
            item.update({"x": round(x, 1), "y": round(y, 1), "physics": True})
        out.append(item)
    return out

vis_edges = []
for idx, e in enumerate(edges):
    style = EDGE_STYLES[e["kind"]]
    src_label = DISPLAY_RENAMES.get(e["source"], e["source"])
    dst_label = DISPLAY_RENAMES.get(e["target"], e["target"])
    vis_edges.append({
        "id": idx, "from": e["source"], "to": e["target"], "kind": e["kind"],
        "label": style["label"],
        "title": f"{escape(src_label)} -> {escape(dst_label)}<br>{escape(style['label'])}<br>{escape(e['note'])}",
        "arrows": style["arrows"], "color": {"color": style["color"]}, "width": style["width"],
        "dashes": style["dashes"],
        "font": {"align": "middle", "size": 10, "strokeWidth": 3, "strokeColor": "#ffffff"},
    })

with OUT_EDGES.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "target", "source_label", "target_label", "kind", "note"])
    writer.writeheader()
    for e in edges:
        writer.writerow({**e, "source_label": DISPLAY_RENAMES.get(e["source"], e["source"]), "target_label": DISPLAY_RENAMES.get(e["target"], e["target"])})
with OUT_NODES.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["node", "label", "group", "degree", "grouped_x", "grouped_y"])
    writer.writeheader()
    for node in visible_columns:
        x, y = positions[node]
        writer.writerow({
            "node": node, "label": DISPLAY_RENAMES.get(node, node), "group": group_for(node),
            "degree": sum(1 for e in edges if e["source"] == node or e["target"] == node),
            "grouped_x": round(x, 1), "grouped_y": round(y, 1),
        })

node_group_legend = "".join(
    f'''<span class="legend-item"><span class="node-dot" style="background:{cfg['background']}; border-color:{cfg['border']};"></span>{escape(group.replace('_', ' '))}</span>'''
    for group, cfg in GROUP_STYLES.items()
)
edge_legend = "".join(
    f'''<span class="legend-item"><span class="edge-swatch" style="border-top-color:{cfg['color']}; {'border-top-style:dashed;' if cfg['dashes'] else ''}"></span>{escape(cfg['label'])}</span>'''
    for cfg in EDGE_STYLES.values()
)
group_options = {
    group: {"shape": "dot", "size": cfg["size"], "color": {"background": cfg["background"], "border": cfg["border"]}, "font": {"color": cfg["font"]}}
    for group, cfg in GROUP_STYLES.items()
}
checkboxes = "".join(
    f'''<label class="check"><input type="checkbox" class="edge-toggle" data-kind="{kind}" checked onchange="applyEdgeFilters()"> {escape(cfg['label'])}</label>'''
    for kind, cfg in EDGE_STYLES.items()
)


def make_html(layout: str, out_path: Path) -> None:
    is_grouped = layout == "grouped"
    physics_options = "{ enabled: false }" if is_grouped else """{
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: { gravitationalConstant: -90, centralGravity: 0.008, springLength: 170, springConstant: 0.04, damping: 0.55, avoidOverlap: 0.65 },
        stabilization: { iterations: 220 }
      }"""
    smooth_options = "{ type: 'cubicBezier', forceDirection: 'horizontal', roundness: 0.35 }" if is_grouped else "{ type: 'dynamic' }"
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>LCK S15 Feature Relationship Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #fafafa; color: #222; }}
    header {{ padding: 10px 14px; border-bottom: 1px solid #ddd; background: white; max-height: 24vh; overflow:auto; }}
    #graph {{ width: 100vw; height: 76vh; background: #fff; }}
    .legend-block {{ margin-top: 4px; }}
    .legend-title {{ font-weight: 700; margin-right: 8px; }}
    .legend {{ font-size: 13px; display: flex; gap: 10px 14px; flex-wrap: wrap; align-items:center; }}
    .legend-item {{ white-space: nowrap; }}
    .edge-swatch {{ display:inline-block; width: 26px; border-top: 3px solid #777; vertical-align: middle; margin-right: 5px; }}
    .node-dot {{ display:inline-block; width: 13px; height: 13px; border: 2px solid #777; border-radius: 50%; vertical-align: middle; margin-right: 5px; }}
    .check {{ margin-right: 16px; white-space: nowrap; }}
    input[type="checkbox"] {{ transform: translateY(1px); }}
  </style>
</head>
<body>
<header>
  <div class="legend-block"><span class="legend-title">Node groups:</span><div class="legend">{node_group_legend}</div></div>
  <div class="legend-block"><span class="legend-title">Edge types:</span><div class="legend">{edge_legend}</div></div>
  <div class="legend-block"><span class="legend-title">Show edges:</span>{checkboxes}</div>
</header>
<div id="graph"></div>
<script>
const nodes = new vis.DataSet({json.dumps(build_nodes(layout), indent=2)});
const allEdges = {json.dumps(vis_edges, indent=2)};
const edges = new vis.DataSet(allEdges);
const container = document.getElementById('graph');
const data = {{ nodes, edges }};
const options = {{
  interaction: {{ hover: true, tooltipDelay: 80, navigationButtons: true, keyboard: true, dragNodes: true }},
  physics: {physics_options},
  layout: {{ improvedLayout: false }},
  nodes: {{ borderWidth: 1.5, font: {{ size: 14, face: 'system-ui' }}, scaling: {{ min: 15, max: 30 }} }},
  edges: {{ smooth: {smooth_options} }},
  groups: {json.dumps(group_options, indent=4)}
}};
const network = new vis.Network(container, data, options);
function applyEdgeFilters() {{
  const enabled = new Set(Array.from(document.querySelectorAll('.edge-toggle')).filter(cb => cb.checked).map(cb => cb.dataset.kind));
  edges.update(allEdges.map(edge => ({{ id: edge.id, hidden: !enabled.has(edge.kind) }})));
}}
</script>
</body>
</html>
'''
    out_path.write_text(html, encoding="utf-8")

make_html("grouped", OUT_GROUPED_HTML)
make_html("flowy", OUT_FLOWY_HTML)

OUT_DISCUSSION.write_text("""### Additional Relationship-Type Discussion Points

**1. Duration as an exposure variable**

Raw counting stats often increase in longer games because there is more time for events to occur. This applies naturally to team-level totals such as `team_kills`, `team_deaths`, `team_assists`, `team_cs`, `team_golds`, and total team damage. It can also apply to player-level stats, but the relationship is weaker because the additional events are distributed across five players and depend heavily on role, champion, game state, and team strategy.

Because of this, it is safer to interpret duration relationships as **exposure effects** rather than direct causal effects. A possible graph edge would be `Duration -> team_kills` or `Duration -> team_deaths` labeled as `scales_with`, while player-level edges such as `Duration -> kills` should be treated more cautiously.

**2. Player raw stats and same-role difference stats**

A same-role difference stat is partly derived from the player stat and the opposing same-role player's stat. For example, `kills_diff` is related to the player's `kills`, but it is not simply the same variable because it also depends on the opponent's kills.

This means raw stats and same-role difference stats should not be treated as independent evidence. However, the relationship is not a clean direct causal mechanism. It is better described as a **derived comparison** or **relative-performance relationship**.

A possible edge type for this would be `relative_to_opponent`, for example `kills -> kills_diff`, but this should be visually distinct from `directly increases` because the value of `kills_diff` also depends on the opponent.

**3. Fifteen-minute checkpoint stats and full-game difference stats**

The `*_at_15` variables are early-game checkpoints, while the role-opponent difference variables describe a broader same-role performance difference. For example, `csd_at_15` is an early checkpoint related to lane or role advantage, while `cs_diff` describes the final or full-game CS difference against the same-role opponent.

In League of Legends, early advantages can snowball into later advantages through gold, experience, map control, and item timing. Therefore, it is reasonable to discuss a soft relationship from `*_at_15` features to later `*_diff` features.

However, this is not deterministic. A player can have an early advantage and lose it later, or recover from an early deficit. These edges should therefore be labeled as **snowball tendency** or `snowballs_into`, not as direct causal effects.

**4. Recommended graph treatment**

For the conservative graph, keep only strong direct mechanisms, preconditions, and component relationships. Duration effects, raw-to-difference relationships, and early-to-late snowball relationships are useful discussion points, but they should be added later using softer edge types such as:

- `scales_with` for duration and exposure effects
- `relative_to_opponent` for raw stat to same-role difference relationships
- `snowballs_into` for early-game checkpoint advantages influencing later advantages

This keeps the graph useful without overstating causal certainty.
""", encoding="utf-8")

print(f"Wrote {OUT_GROUPED_HTML}")
print(f"Wrote {OUT_FLOWY_HTML}")
print(f"Wrote {OUT_EDGES}")
print(f"Wrote {OUT_NODES}")
print(f"Wrote {OUT_DISCUSSION}")
