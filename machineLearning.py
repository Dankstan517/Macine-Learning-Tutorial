import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

OUT_DIR = os.getcwd()


# FIGURE 1 — MLP Architecture (MATCH REPORT)

def plot_mlp_architecture():
    fig, ax = plt.subplots(figsize=(12,7))
    ax.set_xlim(0,10)
    ax.set_ylim(0,7)
    ax.axis('off')

    layers = [
        (1, [2,3,4,5], '#4472C4', "Input Layer\n(Raw pixels/data)", ['x₁','x₂','x₃','x₄']),
        (3, [1.5,2.5,3.5,4.5,5.5], '#ED7D31', "Hidden Layer 1\n(Edges & patterns)", ['','','','','']),
        (6, [2,3,4,5], '#70AD47', "Hidden Layer 2\n(Shapes & textures)", ['','','','']),
        (9, [3,4,5], '#FF0000', "Output Layer\n(Predictions)", ['Bird','Dog','Cat'])
    ]

    radius = 0.3

 
    for x, ys, color, label, texts in layers:
        for i, y in enumerate(ys):
            circle = plt.Circle((x,y), radius, color=color, ec='white', lw=1.5)
            ax.add_patch(circle)

            if texts[i]:
                ax.text(x, y, texts[i], ha='center', va='center',
                        fontsize=8, color='white', weight='bold')

        
        ax.text(x, 1, label, ha='center', fontsize=9, color=color, weight='bold')

    
    for i in range(len(layers)-1):
        x1, ys1 = layers[i][0], layers[i][1]
        x2, ys2 = layers[i+1][0], layers[i+1][1]

        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1+radius, x2-radius], [y1,y2],
                        color='gray', alpha=0.3, lw=0.8)

   
    ax.text(5, 6.5, "Multi-Layer Perceptron Architecture",
            ha='center', fontsize=14, weight='bold')

   
    ax.text(5, 6.1, "Flow left → right through weighted connections",
            ha='center', fontsize=9, color='gray')

   
    ax.annotate('', xy=(9,0.5), xytext=(1,0.5),
                arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.text(5, 0.2, "Increasing Abstraction →",
            ha='center', fontsize=9, style='italic')

    path = os.path.join(OUT_DIR, "mlp_architecture.png")

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# FIGURE 2 — Hierarchical Learning (MATCH)

def plot_hierarchy():
    fig, axes = plt.subplots(1, 4, figsize=(14,5))
    fig.patch.set_facecolor('#1A1A2E') 

 
    data_list = [
        np.array([
            [0,0,1,1,0],
            [0,1,0,0,1],
            [1,1,1,1,1],
            [1,0,0,0,1],
            [1,0,0,0,1]
        ]),
        np.array([
            [0,1,1,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0]
        ]),
        np.array([
            [0,0,1,0,0],
            [0,1,0,1,0],
            [1,0,0,0,1],
            [0,1,0,1,0],
            [0,0,1,0,0]
        ]),
        np.array([
            [0,1,1,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0]
        ])
    ]

    titles = ["Input Layer", "Layer 1", "Layer 2", "Output Layer"]
    subtitles = ["Pixel Intensities", "Edges & Gradients", "Curves & Shapes", "\"Circle\" Concept"]

    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FF5757']
    cmaps = ['Blues', 'Oranges', 'Greens', 'Reds']

    for ax, data, title, sub, color, cmap in zip(axes, data_list, titles, subtitles, colors, cmaps):
        ax.set_facecolor('#1A1A2E')

        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

       
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

       
        ax.set_title(title, color=color, fontsize=11, weight='bold')

       
        ax.set_xlabel(sub, color='white', fontsize=9)

   
    fig.suptitle(
        "Hierarchical Representation Learning\nEach layer extracts more abstract features",
        color='white',
        fontsize=14,
        weight='bold'
    )

    path = os.path.join(OUT_DIR, "hierarchy.png")

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1A1A2E')
    plt.show()
    plt.close()


# FIGURE 3 — Activation Functions (MATCH)

def plot_activation_functions():
    x = np.linspace(-5, 5, 400)

    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    fig.patch.set_facecolor('#F8F9FB')

    # Data for each activation
    functions = [
        ("ReLU  f(x)=max(0,x)", np.maximum(0, x), 'green',
         ["+ No vanishing gradient (x>0)",
          "+ Fast convergence",
          "+ Sparse activation",
          "- Dead neurons possible"]),

        ("Sigmoid  f(x)=1/(1+e^-x)", 1/(1+np.exp(-x)), 'blue',
         ["+ Smooth output (0,1)",
          "+ Good for probabilities",
          "- Vanishing gradient",
          "- Not zero-centered"]),

        ("Tanh  f(x)=tanh(x)", np.tanh(x), 'orange',
         ["+ Zero-centered (-1,1)",
          "+ Stronger gradients than sigmoid",
          "- Vanishing gradient",
          "- Slower than ReLU"])
    ]

    for ax, (title, y, color, notes) in zip(axes, functions):
        ax.plot(x, y, color=color, linewidth=2)

        
        ax.axhline(0, linestyle='--', color='gray')
        ax.axvline(0, linestyle='--', color='gray')

        ax.set_title(title, fontsize=10, weight='bold', color=color)
        ax.set_xlabel("Input (x)")
        ax.set_ylabel("Output f(x)")
        ax.grid(True, alpha=0.3)

        
        for i, note in enumerate(notes):
            col = 'green' if note.startswith('+') else 'red'
            ax.text(0.02, 0.95 - i*0.12, note,
                    transform=ax.transAxes,
                    fontsize=8,
                    color=col,
                    verticalalignment='top')

    fig.suptitle("Activation Functions: Mathematical Shapes & Properties",
                 fontsize=14, weight='bold')

    path = os.path.join(OUT_DIR, "activation.png")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# FIGURE 4 — Training Accuracy (MATCH)

def plot_training_accuracy():
    epochs = np.arange(1, 51)

    
    relu = 0.95 * (1 - np.exp(-epochs / 8))
    tanh = 0.86 * (1 - np.exp(-epochs / 14))
    sigmoid = 0.72 * (1 - np.exp(-epochs / 22))

    plt.figure(figsize=(12, 6))

   
    plt.plot(epochs, relu, color='green', linewidth=2.5, label='ReLU')
    plt.plot(epochs, tanh, color='orange', linewidth=2.5, label='Tanh')
    plt.plot(epochs, sigmoid, color='blue', linewidth=2.5, label='Sigmoid')

   
    plt.fill_between(epochs, relu, color='green', alpha=0.1)
    plt.fill_between(epochs, tanh, color='orange', alpha=0.1)
    plt.fill_between(epochs, sigmoid, color='blue', alpha=0.1)

    
    plt.text(45, relu[-1], f"ReLU: {relu[-1]:.2f}", color='green', fontsize=10, weight='bold')
    plt.text(45, tanh[-1]-0.02, f"Tanh: {tanh[-1]:.2f}", color='orange', fontsize=10, weight='bold')
    plt.text(45, sigmoid[-1]-0.02, f"Sigmoid: {sigmoid[-1]:.2f}", color='blue', fontsize=10, weight='bold')

    
    plt.title(
        "Activation Function Performance Over Training\n"
        "ReLU converges fastest and achieves highest accuracy",
        fontsize=13,
        weight='bold'
    )

    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")

  
    plt.grid(alpha=0.3)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend()

    path = os.path.join(OUT_DIR, "training_accuracy.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



# FIGURE 5 — Vanishing Gradient (MATCH)

def plot_vanishing_gradient():
    layers = ['Output', 'Layer 4', 'Layer 3', 'Layer 2', 'Layer 1', 'Input']
    
    
    sigmoid = [1.00, 0.25, 0.062, 0.015, 0.004, 0.001]
    relu = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75]

    x = np.arange(len(layers))
    width = 0.32

    plt.figure(figsize=(12, 6))

   
    bars1 = plt.bar(x - width/2, sigmoid, width=width, color='#4C72B0', label='Sigmoid gradient')
    bars2 = plt.bar(x + width/2, relu, width=width, color='#6BA84F', label='ReLU gradient')

   
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                 f"{height:.3f}" if height < 0.1 else f"{height:.2f}",
                 ha='center', va='bottom', fontsize=8, color='#4C72B0', weight='bold')

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                 f"{height:.2f}",
                 ha='center', va='bottom', fontsize=8, color='#6BA84F', weight='bold')

    
    plt.title(
        "Vanishing Gradient: Sigmoid vs ReLU Across Layers",
        fontsize=13,
        weight='bold'
    )

    plt.text(0.5, 1.05,
             "Sigmoid gradients shrink to near-zero → slow/stalled learning",
             transform=plt.gca().transAxes,
             ha='center',
             fontsize=9,
             color='red',
             style='italic')

   
    plt.xticks(x, layers)
    plt.ylim(0, 1.1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend()

    path = os.path.join(OUT_DIR, "gradient.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# MAIN

if __name__ == "__main__":
    print("Generating Figures...")

    plot_mlp_architecture()
    plot_hierarchy()
    plot_activation_functions()
    plot_training_accuracy()
    plot_vanishing_gradient()

    print("All plots generated successfully!")