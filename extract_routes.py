import sqlite3
import pandas as pd
import re

#############################
#        EXTRACT DATA       #
#############################

conn = sqlite3.connect("BoardLib/kilter_board.db")

dataset = pd.read_sql("""
    SELECT
        climbs.uuid,
        climbs.name,
        climb_stats.angle,
        ROUND(climb_stats.display_difficulty) AS display_difficulty,
        difficulty_grades.boulder_name,
        climb_stats.ascensionist_count,
        climbs.frames,
        climbs.layout_id
    FROM climbs
    JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid
    LEFT JOIN difficulty_grades ON ROUND(climb_stats.display_difficulty) = difficulty_grades.difficulty
    JOIN layouts ON climbs.layout_id = layouts.id
    JOIN products ON layouts.product_id = products.id
    JOIN product_sizes ON products.id = product_sizes.product_id
    WHERE climb_stats.ascensionist_count >= 50 AND climbs.layout_id = 1 AND product_sizes.id = 10
    """, conn)

#############################
#        CONVERT HOLDS      #
#############################

placements = pd.read_sql("SELECT id, hole_id FROM placements", conn)
holes = pd.read_sql("SELECT id, x, y FROM holes", conn)
roles = pd.read_sql("SELECT id, name FROM placement_roles", conn)

placement_to_hole = dict(zip(placements.id, placements.hole_id))
hole_to_coord = dict(zip(holes.id, zip(holes.x, holes.y)))
role_names = dict(zip(roles.id, roles.name))


def placements_to_coords(frame_string):
    pairs = re.findall(r"p(\d+)r(\d+)", frame_string)
    pairs = [(int(p), int(r)) for p, r in pairs]

    start = []
    middle = []
    finish = []

    for placement_id, role_id in pairs:
        hole_id = placement_to_hole[placement_id]
        coord = hole_to_coord[hole_id]
        role = role_names[role_id]

        if coord is None:
            continue

        if role == "start":
            start.append([role, coord])

        elif role == "finish":
            finish.append([role, coord])

        else:
            middle.append([role, coord])

    return start, middle, finish

dataset[["start", "middle", "finish"]] = dataset["frames"].apply(lambda f: pd.Series(placements_to_coords(f)))


#############################
#      DATASET CLEANUP      #
#############################

dataset = dataset.drop(columns=["frames", "ascensionist_count", "display_difficulty", "layout_id"])

dataset = dataset.rename(columns={"boulder_name": "v_grade"})

dataset["v_grade"] = dataset["v_grade"].str.extract(r"V(\d+)").astype(int)

#############################
#       CREATE DATASET      #
#############################

dataset.to_csv("DATA.csv", index=False)


