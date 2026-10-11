import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# --- 1. Configuration ---
GRID_SIZE = 120
N_PARTICLES = 200
DIFFUSION = 1.2
COOLING = 0.94
HEAT_INJECTION = 4.0

# --- 2. Initialize State ---
# Temperature grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Particle positions (x, y)
x = np.random.rand(N_PARTICLES) * GRID_SIZE
y = np.random.rand(N_PARTICLES) * GRID_SIZE

# Particle velocities (vx, vy)
vx = np.zeros(N_PARTICLES)
vy = np.zeros(N_PARTICLES)

# --- 3. Figure Setup (Cinematic / No UI) ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')      # Black background
ax.set_facecolor('black')
ax.axis('off')                        # Hide axes for a clean look
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)

# The heat map image
im = ax.imshow(grid, cmap='magma', extent=[0, GRID_SIZE, 0, GRID_SIZE], 
               origin='lower', vmin=0, vmax=10, animated=True)

# The particles drawn on top (small white dots)
particles_plot, = ax.plot(x, y, 'w.', ms=1.5, alpha=0.5)

# Sci-Fi HUD overlay
hud_text = ax.text(
    2, GRID_SIZE - 4, '', color='cyan', fontsize=9, 
    family='monospace', ha='left', va='top', 
    bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='cyan', alpha=0.5)
)

# --- 4. Physics & Animation Logic ---
def update(frame):
    global grid, x, y, vx, vy
    
    # 1. Calculate Swirling Flow Field
    # Creates a dynamic, swirling wind vector based on position and time
    time_val = frame * 0.02
    angle = np.sin(x * 0.03 + time_val) + np.cos(y * 0.03 - time_val * 1.5)
    
    # 2. Update Particle Physics
    # Add wind force, apply friction, update positions
    vx = vx * 0.9 + np.cos(angle) * 0.6
    vy = vy * 0.9 + np.sin(angle) * 0.6
    
    x += vx
    y += vy
    
    # Wrap particles around the screen edges
    x %= GRID_SIZE
    y %= GRID_SIZE
    
    # 3. Inject Heat into Grid at Particle Locations
    ix = x.astype(int)
    iy = y.astype(int)
    
    # Modulate heat injection with a sine wave for a "pulsing" effect
    pulse = 0.5 + 0.5 * np.sin(frame * 0.1)
    grid[iy, ix] += HEAT_INJECTION * (0.5 + pulse)
    
    # 4. Grid Physics: Diffusion & Cooling
    # Spread heat to neighbors (gaussian blur)
    grid = gaussian_filter(grid, sigma=DIFFUSION)
    # Cool down slightly
    grid *= COOLING
    # Clean up near-zero values
    grid[grid < 0.01] = 0
    
    # 5. Update Visuals
    im.set_array(grid)
    particles_plot.set_data(x, y)
    
    # 6. Update HUD
    max_temp = np.max(grid)
    avg_temp = np.mean(grid)
    hud_text.set_text(
        f"SYSTEM: THERMAL FLUX\n"
        f"FRAME: {frame:04d}\n"
        f"PARTICLES: {N_PARTICLES}\n"
        f"MAX TEMP: {max_temp:6.2f} K\n"
        f"AVG TEMP: {avg_temp:6.2f} K"
    )
    
    return [im, particles_plot, hud_text]

# --- 5. Run Animation ---
ani = animation.FuncAnimation(
    fig, update, frames=2000, interval=30, blit=True
)

plt.show()