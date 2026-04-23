import pygame
import pygame.gfxdraw
import sys
import numpy as np
from entities.particle import Particle
from world.environment import Environment
from engine.physics import resolve_collision, attempt_reaction
# --- Setup ---
WIDTH, HEIGHT = 800, 600
FPS = 60
dt = 0.1

pygame.init()
font = pygame.font.SysFont("monospace", 13)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("World Zero — Life Simulation")
clock = pygame.time.Clock()
# --- Environment ---
env = Environment(WIDTH, HEIGHT)

# --- Particles ---

import random
particles = [
    Particle(
        x=random.randint(300, 500),
        y=random.randint(200, 400),
        vx=random.uniform(-100, 100),
        vy=random.uniform(-100, 100),
        mass=random.uniform(1.5, 4.0),
        internal_energy=60.0,
        p_type=random.choice(["alpha", "beta", "gamma"])
    )
    for _ in range(100)
]


# --- Main Loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 1. Draw temperature field (background)
    env.draw(screen)

    # 2. Update + draw particles
    for p in particles:
        t_local = env.get_temperature(p.pos[0], p.pos[1])
        p.update(dt, local_temp=t_local)
        p.apply_boundary(WIDTH, HEIGHT)


        x, y = int(p.pos[0]), int(p.pos[1])
        radius = max(8, int(p.mass * 6))

        # Glow layers (soft outer ring)
        for glow in range(3):
            gr = radius + (3 - glow) * 4
            gc = tuple(min(255, c // 3) for c in p.color)
            pygame.gfxdraw.filled_circle(screen, x, y, gr, (*gc, 40))

        # Smooth filled body
        pygame.gfxdraw.filled_circle(screen, x, y, radius, p.color)

        # Anti-aliased edge
        pygame.gfxdraw.aacircle(screen, x, y, radius, (255, 255, 255))
    # HUD — show each particle's energy
    for i, p in enumerate(particles):
        label = font.render(
            f"{p.p_type} | E={p.total_energy():.1f} | T={env.get_temperature(p.pos[0], p.pos[1]):.0f}K",
            True, (220, 220, 220)
        )
        screen.blit(label, (10, 10 + i * 18))
    # Resolve all pairwise collisions
    to_remove = set()  # indices of particles to remove after reactions
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            if i in to_remove or j in to_remove:
                continue

            p1, p2 = particles[i], particles[j]

            # Check collision first
            delta = p1.pos - p2.pos
            dist  = np.linalg.norm(delta)
            r1    = max(4, int(p1.mass * 6))
            r2    = max(4, int(p2.mass * 6))

            if dist < r1 + r2:
                # Try reaction first
                reacted = attempt_reaction(p1, p2, particles)
                if reacted:
                    to_remove.add(i)
                    to_remove.add(j)
                else:
                    resolve_collision(p1, p2)

    # Remove reacted particles
    particles = [p for idx, p in enumerate(particles) if idx not in to_remove]

    pygame.display.flip()
    clock.tick(FPS)