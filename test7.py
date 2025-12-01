import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def graficar_historia_keydoor():
    # --- CONFIGURACIÓN DE LA FIGURA ---
    # Creamos 3 subplots alineados horizontalmente
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Definimos los pasos de tiempo
    times = [0, 1, 2]
    time_labels = ["t-1 (Pasado)", "t (Presente)", "t+1 (Futuro)"]

    # --- DEFINICIÓN DE NODOS Y POSICIONES ---
    pos = {}
    nodes = []
    
    for t in times:
        # Nombres de los nodos
        p_node = f"P_{t}"  # Posición
        k_node = f"K_{t}"  # Llave
        o_node = f"O_{t}"  # Observación
        a_node = f"A_{t}"  # Acción
        nodes.extend([p_node, k_node, o_node, a_node])
        
        # Coordenadas (X=Tiempo, Y=Altura)
        # Separamos K y P verticalmente para ver la estructura interna
        pos[k_node] = (t, 2.5) # Llave (Arriba-Medio) -> Memoria Larga
        pos[p_node] = (t, 1.5) # Posición (Abajo-Medio) -> Dinámica Rápida
        pos[o_node] = (t, 3.5) # Obs (Arriba del todo)
        pos[a_node] = (t, 0.5) # Acción (Abajo del todo)

    # Colores base (Pasteles)
    colors = {
        'P': '#87CEFA', # Azul Cielo (Pos)
        'K': '#FFD700', # Dorado (Key)
        'O': '#98FB98', # Verde (Obs)
        'A': '#FA8072'  # Salmon (Act)
    }

    # ==========================================
    # PANEL 1: GRAFO ORIGINAL (POMDP)
    # ==========================================
    ax1 = axes[0]
    G1 = nx.DiGraph()
    G1.add_nodes_from(nodes)
    
    edges1 = []
    for t in times:
        # Emisión: Estado (P, K) -> Obs
        edges1.append((f"P_{t}", f"O_{t}"))
        edges1.append((f"K_{t}", f"O_{t}"))
        
        if t < max(times):
            # Transición Temporal
            # 1. Posición: Afectada por P anterior, Acción y Llave (Puertas)
            edges1.append((f"P_{t}", f"P_{t+1}"))
            edges1.append((f"A_{t}", f"P_{t+1}"))
            edges1.append((f"K_{t}", f"P_{t+1}")) # Interacción Key->Pos
            
            # 2. Llave: Afectada por K anterior (Persistencia) y Posición (Recoger)
            edges1.append((f"K_{t}", f"K_{t+1}")) 
            edges1.append((f"P_{t}", f"K_{t+1}")) # Interacción Pos->Key
            
            # Política (Obs -> Acción)
            edges1.append((f"O_{t}", f"A_{t}")) 

    G1.add_edges_from(edges1)
    
    # Obtener colores para los nodos
    cols1 = [colors[n[0]] for n in G1.nodes()]
    
    # Dibujar G1
    nx.draw(G1, pos, ax=ax1, node_color=cols1, with_labels=True, node_size=1000, 
            font_size=8, font_weight='bold', arrowsize=15, edgecolors='black')
    ax1.set_title("1. Grafo Original KeyDoor ($\mathcal{G}$)\n(Entrelazado y con Política)", fontsize=12)

    # ==========================================
    # PANEL 2: GRAFO FÍSICO (G-DAGGER)
    # ==========================================
    ax2 = axes[1]
    G2 = nx.DiGraph()
    G2.add_nodes_from(nodes)
    
    # Filtramos flechas que apuntan a A (La Política)
    edges2 = [e for e in edges1 if not e[1].startswith("A")]
    G2.add_edges_from(edges2)
    
    # Dibujar G2
    nx.draw(G2, pos, ax=ax2, node_color=cols1, with_labels=True, node_size=1000, 
            font_size=8, font_weight='bold', arrowsize=15, edgecolors='black')
    
    # Dibujar las eliminadas en punteado (fantasma)
    removed = [e for e in edges1 if e[1].startswith("A")]
    nx.draw_networkx_edges(G2, pos, ax=ax2, edgelist=removed, style='dotted', edge_color='#AAAAAA', width=1)
    
    ax2.set_title("2. Grafo Físico ($\mathcal{G}^\dagger$)\n(Sin Flechas hacia Acciones)", fontsize=12)

    # ==========================================
    # PANEL 3: ESTADO INFO-ESTRUCTURAL (RESULTADO)
    # ==========================================
    ax3 = axes[2]
    G3 = G2.copy()
    
    # Lógica de resaltado: Apagar pasado/futuro, ENCENDER P_1 y K_1
    cols3 = []
    sizes3 = []
    edge_cols = []
    linewidths = []
    
    for node in G3.nodes():
        # EL CUELLO DE BOTELLA ES EL PAR {P_1, K_1}
        if node in ["P_1", "K_1"]: 
            cols3.append('#FF4444') # Rojo Fuerte
            sizes3.append(2500)     # Muy grande
            edge_cols.append('black')
            linewidths.append(2.0)
        elif "1" in node: # Presente (Obs, Act) - Normal
            cols3.append('#DDDDDD')
            sizes3.append(800)
            edge_cols.append('gray')
            linewidths.append(1.0)
        else: # Pasado y Futuro (Fade out)
            cols3.append('#F0F0F0') # Casi blanco
            sizes3.append(600)
            edge_cols.append('#DDDDDD')
            linewidths.append(1.0)

    # Dibujar G3
    nx.draw(G3, pos, ax=ax3, node_color=cols3, with_labels=True, node_size=sizes3, 
            font_size=9, font_weight='bold', edgecolors=edge_cols, linewidths=linewidths, 
            arrowsize=12, edge_color='#999999')

    # Etiquetas de tiempo eje X
    for i, label in enumerate(time_labels):
        ax3.text(i, 0.0, label, ha='center', fontsize=9, color='#555555')

    ax3.set_title(r"3. Resultado: $\mathbb{I}_h^\dagger = \{Pos_t, Key_t\}$", fontsize=13, color='#CC0000', fontweight='bold')
    
    # --- DECORACIÓN: LA CAJA ROJA DEL CONJUNTO ---
    # Creamos un rectángulo redondeado alrededor de P_1 y K_1
    # Coordenadas manuales ajustadas para englobar (1, 1.5) y (1, 2.5)
    bbox = FancyBboxPatch((0.6, 1.1), 0.8, 1.9, boxstyle="round,pad=0.1", 
                          ec="red", fc="none", lw=2, linestyle='--')
    ax3.add_patch(bbox)

    # Anotación explicativa
    ax3.annotate(
        "Conjunto Mínimo\n(Info-State)", 
        xy=(1.4, 2.0),          # Punto medio del lado derecho de la caja
        xytext=(1.8, 3.0),      # Posición del texto
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, connectionstyle="arc3,rad=0.2"),
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFAA", ec="#FFA500", alpha=1.0)
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_historia_keydoor()