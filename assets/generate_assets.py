import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch

# Create Logo
def create_logo():
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background circle
    circle_bg = Circle((5, 5), 4.5, color='#1f77b4', alpha=0.1)
    ax.add_patch(circle_bg)
    
    # Main circle
    circle_main = Circle((5, 5), 4, color='#1f77b4', alpha=0.9)
    ax.add_patch(circle_main)
    
    # Road segments (white)
    # Horizontal road
    rect_h = Rectangle((1, 4.5), 8, 1, color='white', alpha=0.8)
    ax.add_patch(rect_h)
    
    # Vertical road
    rect_v = Rectangle((4.5, 1), 1, 8, color='white', alpha=0.8)
    ax.add_patch(rect_v)
    
    # Road markings (dashed)
    ax.plot([1, 9], [5, 5], 'y--', linewidth=2, alpha=0.6)
    ax.plot([5, 5], [1, 9], 'y--', linewidth=2, alpha=0.6)
    
    # Vehicles (simple circles)
    vehicles = [(2, 5), (8, 5), (5, 2), (5, 8)]
    for x, y in vehicles:
        vehicle = Circle((x, y), 0.3, color='#FF6B6B', alpha=0.9)
        ax.add_patch(vehicle)
    
    # AI/Network nodes
    nodes = [(3, 6.5), (7, 6.5), (3, 3.5), (7, 3.5)]
    for x, y in nodes:
        node = Circle((x, y), 0.2, color='#4ECDC4', alpha=0.9)
        ax.add_patch(node)
    
    # Connection lines
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            ax.plot([x1, x2], [y1, y2], color='#4ECDC4', alpha=0.4, linewidth=1)
    
    plt.tight_layout(pad=0)
    plt.savefig('/Volumes/MD/traffic_gnn_idd/assets/logo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Logo created successfully!")
    
    plt.close()

# Create Background Image
def create_background():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#f0f2f6')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Large semi-transparent circles
    circle1 = Circle((2, 8), 2, color='#1f77b4', alpha=0.05)
    circle2 = Circle((8.5, 2), 2.5, color='#FF6B6B', alpha=0.05)
    circle3 = Circle((5, 5), 1.5, color='#4ECDC4', alpha=0.05)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Grid pattern
    for i in range(0, 11):
        ax.plot([i, i], [0, 10], color='gray', alpha=0.05, linewidth=0.5)
        ax.plot([0, 10], [i, i], color='gray', alpha=0.05, linewidth=0.5)
    
    # Road network visualization
    ax.plot([1, 9], [7, 7], color='#1f77b4', alpha=0.15, linewidth=3)
    ax.plot([1, 9], [5, 5], color='#1f77b4', alpha=0.15, linewidth=3)
    ax.plot([1, 9], [3, 3], color='#1f77b4', alpha=0.15, linewidth=3)
    
    ax.plot([3, 3], [1, 9], color='#FF6B6B', alpha=0.15, linewidth=3)
    ax.plot([5, 5], [1, 9], color='#FF6B6B', alpha=0.15, linewidth=3)
    ax.plot([7, 7], [1, 9], color='#FF6B6B', alpha=0.15, linewidth=3)
    
    # Intersection points
    intersections = [(3, 3), (3, 5), (3, 7), (5, 3), (5, 5), (5, 7), (7, 3), (7, 5), (7, 7)]
    for x, y in intersections:
        circle = Circle((x, y), 0.15, color='#4ECDC4', alpha=0.2)
        ax.add_patch(circle)
    
    plt.tight_layout(pad=0)
    plt.savefig('/Volumes/MD/traffic_gnn_idd/assets/background.png', dpi=300, bbox_inches='tight', facecolor='#f0f2f6')
    print("Background created successfully!")
    
    plt.close()

# Create Traffic Network Icon
def create_network_icon():
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Network nodes
    nodes = [(2, 8), (5, 8), (8, 8), (2, 5), (5, 5), (8, 5), (2, 2), (5, 2), (8, 2)]
    
    # Connection lines
    connections = [
        (0, 1), (1, 2), (0, 3), (1, 4), (2, 5),
        (3, 4), (4, 5), (3, 6), (4, 7), (5, 8),
        (6, 7), (7, 8)
    ]
    
    for i, j in connections:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        ax.plot([x1, x2], [y1, y2], color='#1f77b4', alpha=0.4, linewidth=2)
    
    # Draw nodes
    for i, (x, y) in enumerate(nodes):
        size = 300 if i == 4 else 200
        ax.scatter(x, y, s=size, color='#4ECDC4', alpha=0.9, zorder=10)
    
    plt.tight_layout(pad=0)
    plt.savefig('/Volumes/MD/traffic_gnn_idd/assets/network_icon.png', dpi=200, bbox_inches='tight', facecolor='white')
    print("Network icon created successfully!")
    
    plt.close()

if __name__ == "__main__":
    create_logo()
    create_background()
    create_network_icon()
    print("All assets created successfully!")
