import pygame
import pygame.gfxdraw
import sys
import numpy as np
import random
from entities.particle import Particle
from world.environment import Environment
from engine.physics import resolve_collision, attempt_reaction

WIDTH, HEIGHT = 800, 600
FPS = 60
dt = 0.1

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reversible Reaction Test — A + B ⇌ C + D")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 14)

env = Environment(WIDTH, HEIGHT)

# --- Reversible Reaction Table ---
# Forward:  alpha + beta  → delta + gamma
# Reverse:  delta + gamma → alpha + beta
REVERSIBLE = {
    frozenset({"alpha", "beta"})  : ("delta",  "gamma"),   # forward
    frozenset({"delta", "gamma"}) : ("alpha",  "beta"),    # reverse
}

def attempt_reversible(p1, p2, particles, activation_energy=2.0):
    key = frozenset({p1.p_type, p2.p_type})
    if key not in REVERSIBLE:
        return False

    total_ke = p1.kinetic_energy() + p2.kinetic_energy()
    if total_ke < activation_energy:
        return False
        # Boost reverse reaction probability
    if key == frozenset({"delta", "gamma"}):
        prob = 0.8   # override — reverse reaction is highly favored
    prob = p1.reactivity * p2.reactivity
    if np.random.rand() > prob:
        return False

    prod_a, prod_b = REVERSIBLE[key]

    # Center of mass position and velocity
    total_mass = p1.mass + p2.mass
    mid_pos    = (p1.pos + p2.pos) / 2
    com_vel    = (p1.mass * p1.vel + p2.mass * p2.vel) / total_mass
    new_mass   = total_mass * 0.5       # split mass evenly
    new_ie     = (p1.internal_energy + p2.internal_energy) * 0.5

    # --- EXPLOSION — products fly apart in opposite directions ---
    # Random unit vector for separation direction
    angle      = np.random.uniform(0, 2 * np.pi)
    direction  = np.array([np.cos(angle), np.sin(angle)])
    speed = max(3.0, np.sqrt(total_ke / total_mass) * 2.0)  # conserve energy

    vel_a = com_vel + direction * speed   # fly one way
    vel_b = com_vel - direction * speed   # fly opposite

    # Spawn both products separated slightly
    offset = direction * (new_mass * 3)

    product_a = Particle(
        x=mid_pos[0] + offset[0],
        y=mid_pos[1] + offset[1],
        vx=vel_a[0], vy=vel_a[1],
        mass=new_mass,
        internal_energy=new_ie,
        p_type=prod_a
    )
    product_b = Particle(
        x=mid_pos[0] - offset[0],
        y=mid_pos[1] - offset[1],
        vx=vel_b[0], vy=vel_b[1],
        mass=new_mass,
        internal_energy=new_ie,
        p_type=prod_b
    )

    particles.append(product_a)
    particles.append(product_b)
    return True


# --- Balanced starting population ---
def make_particles():
    particles = []
    for t in ["alpha", "beta", "delta", "gamma"]:
        for _ in range(20):                          # 32 total, balanced
            particles.append(Particle(
                x=random.randint(100, 700),
                y=random.randint(100, 500),
                vx=random.uniform(-80, 80),
                vy=random.uniform(-80, 80),
                mass=2.0,                           # equal mass — fair test
                internal_energy=60.0,
                p_type=t
            ))
    return particles

particles = make_particles()

# --- Population counter ---
step = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    env.draw(screen)

    # Update particles
    for p in particles:
        t_local = env.get_temperature(p.pos[0], p.pos[1])
        p.update(dt, local_temp=t_local)
        p.apply_boundary(WIDTH, HEIGHT)
        # If particle nearly stopped — thermal rescue
        if np.linalg.norm(p.vel) < 0.5:
            p.vel += np.random.randn(2) * 0.5 * \
                    env.get_temperature(p.pos[0], p.pos[1]) / 1000

    # Reversible reactions
    to_remove = set()
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            if i in to_remove or j in to_remove:
                continue
            p1, p2 = particles[i], particles[j]
            delta   = p1.pos - p2.pos
            dist    = np.linalg.norm(delta)
            r1      = max(4, int(p1.mass * 6))
            r2      = max(4, int(p2.mass * 6))
            if dist < r1 + r2:
                reacted = attempt_reversible(p1, p2, particles)
                if reacted:
                    to_remove.add(i)
                    to_remove.add(j)
                else:
                    resolve_collision(p1, p2)

    particles = [p for idx, p in enumerate(particles) if idx not in to_remove]

    # Draw particles
    for p in particles:
        x, y   = int(p.pos[0]), int(p.pos[1])
        radius = max(8, int(p.mass * 6))
        pygame.gfxdraw.filled_circle(screen, x, y, radius, p.color)
        pygame.gfxdraw.aacircle(screen, x, y, radius, (255, 255, 255))

    # Population HUD
    counts = {"alpha":0, "beta":0, "gamma":0, "delta":0}
    for p in particles:
        if p.p_type in counts:
            counts[p.p_type] += 1

    hud_lines = [
        f"Step: {step}",
        f"Total particles: {len(particles)}",
        f"alpha (blue)  : {counts['alpha']}",
        f"beta  (orange): {counts['beta']}",
        f"delta (green) : {counts['delta']}",
        f"gamma (red)   : {counts['gamma']}",
        f"",
        f"Reaction: alpha+beta <-> delta+gamma",
    ]
    for i, line in enumerate(hud_lines):
        label = font.render(line, True, (220, 220, 220))
        screen.blit(label, (10, 10 + i * 18))

    pygame.display.flip()
    clock.tick(FPS)
    step += 1