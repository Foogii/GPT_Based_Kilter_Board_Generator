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
        climbs.description,
        climb_stats.angle,
        ROUND(climb_stats.display_difficulty) AS display_difficulty,
        difficulty_grades.boulder_name,
        climb_stats.ascensionist_count,
        climbs.frames
    FROM climbs
    JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid
    LEFT JOIN difficulty_grades ON ROUND(climb_stats.display_difficulty) = difficulty_grades.difficulty
    WHERE climb_stats.ascensionist_count >= 100
    """, conn)

#############################
#        CONVERT HOLDS      #
#############################

placements = pd.read_sql("SELECT id, hole_id FROM placements", conn)
holes = pd.read_sql("SELECT id, x, y FROM holes", conn)

placement_to_hole = dict(zip(placements.id, placements.hole_id))
hole_to_coord = dict(zip(holes.id, zip(holes.x, holes.y)))


def placements_to_coords(frame_string):
    import re

    placement_ids = [int(x) for x in re.findall(r"p(\d+)", frame_string)]

    coords = []
    for p in placement_ids:
        hole_id = placement_to_hole.get(p)
        if hole_id:
            coord = hole_to_coord.get(hole_id)
            if coord:
                coords.append(coord)

    return coords

dataset["coords"] = dataset["frames"].apply(placements_to_coords)

dataset = dataset.drop(columns=["frames"])

#############################
#       CREATE DATASET      #
#############################

dataset.to_csv("DATA.csv", index=False)


