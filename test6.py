import networkx as nx
import matplotlib.pyplot as plt

def graficar_historia_informacion_v2():
    # Configuración de la figura: 3 columnas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Tiempos
    times = [0, 1, 2]
    time_labels = ["t-1 (Pasado)", "t (Presente)", "t+1 (Futuro)"]

    # --- CONFIGURACIÓN DE NODOS Y POSICIONES ---
    pos = {}
    nodes = []
    
    for t in times:
        s_node = f"S_{t}"
        o_node = f"O_{t}"
        a_node = f"A_{t}"
        nodes.extend([s_node, o_node, a_node])
        
        # Coordenadas (X=Tiempo, Y=Capa)
        pos[s_node] = (t, 2)   # Estado (Centro)
        pos[o_node] = (t, 3)   # Obs (Arriba)
        pos[a_node] = (t, 1)   # Acción (Abajo)

    # Colores base (Pasteles)
    colors = {
        'S': '#FFB6C1', # Rosa
        'O': '#98FB98', # Verde
        'A': '#87CEFA'  # Azul
    }

    # ==========================================
    # 1. GRAFO ORIGINAL (POMDP)
    # ==========================================
    ax1 = axes[0]
    G1 = nx.DiGraph()
    G1.add_nodes_from(nodes)
    
    edges1 = []
    for t in times:
        edges1.append((f"S_{t}", f"O_{t}"))
        if t < max(times):
            edges1.append((f"S_{t}", f"S_{t+1}"))
            edges1.append((f"A_{t}", f"S_{t+1}"))
            edges1.append((f"O_{t}", f"A_{t}")) # Política

    G1.add_edges_from(edges1)
    
    cols1 = [colors[n[0]] for n in G1.nodes()]
    nx.draw(G1, pos, ax=ax1, node_color=cols1, with_labels=True, node_size=1200, font_size=9, font_weight='bold')
    ax1.set_title("1. Grafo Original ($\mathcal{G}$)\n(Todo conectado)", fontsize=13)

    # ==========================================
    # 2. GRAFO MODIFICADO (G-DAGGER)
    # ==========================================
    ax2 = axes[1]
    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    
    # Filtramos flechas hacia acciones
    edges2 = [e for e in edges1 if not e[1].startswith("A")]
    G2.add_edges_from(edges2)
    
    nx.draw(G2, pos, ax=ax2, node_color=cols1, with_labels=True, node_size=1200, font_size=9, font_weight='bold')
    
    # Dibujar las eliminadas en punteado
    removed = [e for e in edges1 if e[1].startswith("A")]
    nx.draw_networkx_edges(G2, pos, ax=ax2, edgelist=removed, style='dotted', edge_color='#AAAAAA', width=1)
    
    ax2.set_title("2. Grafo Físico ($\mathcal{G}^\dagger$)\n(Sin Flechas hacia Acciones)", fontsize=13)

    # ==========================================
    # 3. ESTADO INFO-ESTRUCTURAL (EL FINAL)
    # ==========================================
    ax3 = axes[2]
    G3 = G2.copy()
    
    # Colores: Apagar pasado/futuro, ENCENDER S_1
    cols3 = []
    sizes3 = []
    edge_cols = []
    
    for node in G3.nodes():
        if node == "S_1": # EL CUELLO DE BOTELLA
            cols3.append('#FF4444') # Rojo Fuerte
            sizes3.append(2800)     # Muy grande
            edge_cols.append('black')
        elif "1" in node: # Presente (pero no el estado)
            cols3.append('#DDDDDD')
            sizes3.append(1000)
            edge_cols.append('gray')
        else: # Pasado y Futuro (Fade out)
            cols3.append('#F0F0F0') # Casi blanco
            sizes3.append(800)
            edge_cols.append('#DDDDDD')

    nx.draw(G3, pos, ax=ax3, node_color=cols3, with_labels=True, node_size=sizes3, 
            font_size=9, font_weight='bold', edgecolors=edge_cols, arrowsize=15, edge_color='#999999')

    # Etiquetas de tiempo eje X
    for i, label in enumerate(time_labels):
        ax3.text(i, 0.3, label, ha='center', fontsize=9, color='#555555')

    ax3.set_title(r"3. Resultado: Estado Info-Estructural ($\mathbb{I}_h^\dagger$)", fontsize=13, color='#CC0000', fontweight='bold')
    
    # --- LA CORRECCIÓN DE LA FLECHA ---
    ax3.annotate(
        "Este nodo d-separa\nel Pasado del Futuro", 
        xy=(1, 2.2),          # Punta de la flecha (Justo encima del nodo rojo)
        xytext=(1.8, 3.5),    # Texto (Esquina superior derecha, zona libre)
        # Flecha curva (arc3, rad) y fina
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, connectionstyle="arc3,rad=-0.2"),
        ha='center',
        fontsize=10,
        # Caja de fondo para que se lea bien
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFAA", ec="#FFA500", alpha=1.0)
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_historia_informacion_v2()