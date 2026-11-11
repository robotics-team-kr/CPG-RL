"""Seeded physical courses, with a surface for lowered as well as raised steps."""

import xml.etree.ElementTree as ET

import numpy as np


def add_course(world, cfg):
    settings = getattr(cfg, "coop_settings", None)
    if settings is None or settings.terrain == "flat":
        return
    # Lower the infinite backing plane: a floor at z=0 would fill every depression.
    floor = world.find("geom[@name='floor']")
    floor.set("pos", "0 0 -0.25")
    rng = np.random.default_rng(settings.terrain_seed)
    length, width = 0.4, 1.5
    obstacle_start = rng.uniform(1.4, 2.2)
    obstacle_end = obstacle_start + rng.uniform(0.4, 0.9)
    # Wide start platform plus tiled course, all laid out in world +X.
    tiles = [(-1.0, 2.0, 0.0)]
    x = 1.0
    while x < settings.course_length + 2.0:
        length = rng.uniform(0.25, 0.55)
        height = 0.0
        if settings.terrain == "raised":
            height = settings.terrain_height if obstacle_start <= x < obstacle_end else 0.0
        elif settings.terrain == "lowered":
            height = -settings.terrain_height if obstacle_start <= x < obstacle_end else 0.0
        elif settings.terrain == "irregular":
            height = rng.uniform(-settings.terrain_height, settings.terrain_height)
        tiles.append((x + length / 2, length, height))
        x += length
    # Start platform covers [-2, 1]; the center/length below intentionally meets the first tile.
    tiles[0] = (-0.5, 3.0, 0.0)
    for i, (x, span, height) in enumerate(tiles):
        friction = settings.terrain_friction
        if settings.terrain == "friction" and i % 3 == 1:
            friction *= 0.4
        ET.SubElement(
            world,
            "geom",
            name=f"course_{i}",
            type="box",
            pos=f"{x} 0 {(height - 0.20) / 2}",
            size=f"{span / 2} {width} {(height + 0.20) / 2}",
            friction=f"{friction} 0.005 0.0001",
            priority="2",
            condim="3",
            rgba="0.4 0.45 0.5 1",
        )
