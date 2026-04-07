import sqlite3
import pandas as pd
import re
import gpt
import webbrowser

new_route = gpt.new_best_route()
# print(new_route)


def parse_tokens(tokens):
    coords = []

    pattern = r"<(START|MIDDLE|FINISH|FOOT)_X(-?\d+)_Y(-?\d+)>"

    for token in tokens:
        match = re.search(pattern, token)

        if match:
            role = match.group(1)
            x = int(match.group(2))
            y = int(match.group(3))

            coords.append((role, x, y))

    return coords


coords = parse_tokens(new_route)


conn = sqlite3.connect("kilter_board.db")
placements = pd.read_sql("SELECT id, hole_id FROM placements", conn)
holes = pd.read_sql("SELECT id, x, y FROM holes WHERE product_id = 1", conn)  # Product_id = 1 specifies to only use Kilter Boards (from the database)
roles = pd.read_sql("SELECT id, name FROM placement_roles", conn)

roles = roles.drop_duplicates("name").set_index("name")["id"].to_dict()


# Conversion

placement_to_hole = dict(zip(placements.id, placements.hole_id))
hole_to_placement = dict(zip(placements.hole_id, placements.id))

hole_to_coord = dict(zip(holes.id, zip(holes.x, holes.y)))
coord_to_hole = {v: k for k, v in hole_to_coord.items()}

role_map = {k.upper(): v for k, v in roles.items()}

def build_frame(coords):

    frame_parts = []

    for role, x, y in coords:

        hole_id = coord_to_hole.get((x, y))
        if hole_id is None:
            continue

        placement_id = hole_to_placement.get(hole_id)
        if placement_id is None:
            continue

        role_id = role_map.get(role)
        if role_id is None:
            continue

        frame_parts.append(f"p{placement_id}r{role_id}")

    return "".join(frame_parts)

frame_string = build_frame(coords)

url = f"https://grip-connect-kilter-board.vercel.app/?route=<{frame_string}>"

webbrowser.open(url)
