"""Pymunk physics simulation for the falling icon deluge animation.

Generates a JSON file of icon positions/rotations over time, consumed by
``TitleAndMotivation`` in ``slides.py``.

Usage:
    python falling_icon_simulation.py          # regenerate (always)
    python falling_icon_simulation.py --only-missing   # skip if file exists
"""

import json
import random

import numpy as np
import pymunk

from data_manifest import ICON_DELUGE_SIMULATION, SIMULATION_DIR


def run_staggered_simulation(
    output_path=ICON_DELUGE_SIMULATION, num_icons=200, duration=20, fps=30
):
    """Run a pymunk falling-icon simulation and write results to *output_path*."""
    space = pymunk.Space()
    space.gravity = (0, -900)
    space.collision_bias = pow(1.0 - 0.1, 30.0)
    random.seed(42)

    static_body = space.static_body

    floor = pymunk.Segment(static_body, (-600, -300), (600, -300), 20)
    floor.elasticity = 0.6
    floor.friction = 1.0

    left_wall = pymunk.Segment(static_body, (-400, -300), (-400, 800), 5)
    right_wall = pymunk.Segment(static_body, (400, -300), (400, 800), 5)

    space.add(floor)

    # Exponential growth spawn curve
    k = 8.0
    x = np.linspace(0, 1, num_icons)
    exponential_curve = (np.exp(k * x) - 1) / (np.exp(k) - 1)
    spawn_times = duration * 0.5 * (1 - exponential_curve[::-1])
    spawn_times.sort()

    active_bodies = []
    pending_icons = []

    for i in range(num_icons):
        mass = 1
        size = (40, 40)
        moment = pymunk.moment_for_box(mass, size)
        body = pymunk.Body(mass, moment)

        if i < 4:
            body.position = (-200, 150 - (i * 75))
            body.velocity = (random.uniform(-250, 250), random.uniform(0, 100))
            body.angular_velocity = random.uniform(-10, 10)
        else:
            body.position = (random.uniform(-350, 350), 600)
            body.angle = random.uniform(-0.2, 0.2)
            body.angular_velocity = random.uniform(-2, 2)

        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.2
        shape.friction = 0.8

        pending_icons.append(
            {
                "id": i,
                "body": body,
                "shape": shape,
                "spawn_t": spawn_times[i],
                "path": [],
            }
        )

    dt = 1.0 / fps
    for frame in range(int(duration * fps)):
        current_t = frame * dt

        for icon in pending_icons[:]:
            if current_t >= icon["spawn_t"]:
                space.add(icon["body"], icon["shape"])
                active_bodies.append(icon)
                pending_icons.remove(icon)

        space.step(dt)

        for icon in active_bodies:
            icon["path"].append(
                (
                    round(icon["body"].position.x, 2),
                    round(icon["body"].position.y, 2),
                    round(icon["body"].angle, 3),
                    frame,
                )
            )

        total_ke_above_floor = sum(
            b.kinetic_energy for b in space.bodies if b.position.y >= -300
        )
        if total_ke_above_floor < 0.1 and frame < (num_icons * 2):
            break

    output_data = []
    for icon in active_bodies + pending_icons:
        output_data.append(
            {
                "id": icon["id"],
                "spawn_frame": int(icon["spawn_t"] * fps),
                "path": icon["path"],
            }
        )

    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f)


def ensure_data():
    """Generate simulation data only if the output file is missing."""
    if not ICON_DELUGE_SIMULATION.exists():
        print("Generating icon deluge simulation data …")
        run_staggered_simulation()
    else:
        print(f"Simulation data already exists: {ICON_DELUGE_SIMULATION}")


def regenerate_data():
    """Always regenerate simulation data, even if it exists."""
    print("Regenerating icon deluge simulation data …")
    run_staggered_simulation()


if __name__ == "__main__":
    import sys

    if "--only-missing" in sys.argv:
        ensure_data()
    else:
        regenerate_data()
