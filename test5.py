import networkx as nx
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ROBUSTA DE IMPORTACIÓN ---
try:
    d_separated = nx.d_separated
except AttributeError:
    try:
        from networkx.algorithms.d_separation import d_separated
    except ImportError:
        try:
            from networkx.algorithms.d_separation import is_d_separator as d_separated
        except ImportError:
             print("Error de versión networkx")
             exit()

def graficar_flujo_agente_tiger():
    G = nx.DiGraph()
    
    # Reducimos a 5 pasos para que se vea claro el detalle
    steps = [0, 1, 2, 3, 4] 

    # --- 1. CONSTRUCCIÓN DEL GRAFO ---
    for t in steps:
        # Nombres de Nodos
        s1_node = f"S1_{t}"   # Estado Real (Latente)
        o1_node = f"Obs_{t}"  # Observación (Lo que ve el agente)
        h_node  = f"LSTM_{t}" # MEMORIA DEL AGENTE (Hidden State) -> ¡Lo nuevo!
        a_node  = f"Act_{t}"  # Acción
        
        # Agregamos nodos con metadatos para el estilo
        # Layer 3: Estados Latentes (Arriba del todo - El Mundo Oculto)
        G.add_node(s1_node, layer=3, time=t, label="S_real", type="env")
        
        # Layer 2: Observaciones (Interfaz Mundo-Agente)
        G.add_node(o1_node, layer=2, time=t, label="Obs", type="obs")
        
        # Layer 1: LSTM / Memoria (Dentro del Agente - El "Entrelazado")
        G.add_node(h_node, layer=1, time=t, label="Hidden", type="mem")
        
        # Layer 0: Acciones (Abajo - Salida del Agente)
        G.add_node(a_node, layer=0, time=t, label="Act", type="act")

        # --- ARISTAS DEL ENTORNO (Física) ---
        # El estado causa la observación
        G.add_edge(s1_node, o1_node, style='solid')
        
        # --- ARISTAS DEL AGENTE (Cerebro) ---
        # El agente ve la observación y la mete en su LSTM
        G.add_edge(o1_node, h_node, style='solid')
        
        # El LSTM decide la acción
        G.add_edge(h_node, a_node, style='solid')

        # --- DINÁMICA TEMPORAL ---
        if t < steps[-1]:
            next_t = t + 1
            
            # 1. Dinámica del Mundo (S -> S')
            # El estado evoluciona (y la acción previa lo afecta)
            G.add_edge(s1_node, f"S1_{next_t}", style='solid')
            G.add_edge(a_node, f"S1_{next_t}", style='solid')
            
            # 2. Dinámica del Agente (LSTM -> LSTM') -> ¡AQUÍ ESTÁ EL ENTRELAZADO!
            # La memoria pasa del pasado al futuro
            G.add_edge(h_node, f"LSTM_{next_t}", style='dashed', color='gold')

    # --- 2. DIBUJO ---
    print("Generando gráfico de Flujo de Información (Mundo + LSTM)...")
    plt.figure(figsize=(14, 8))
    
    # Posiciones Manuales
    pos = {}
    for node, data in G.nodes(data=True):
        t = data['time']
        layer = data['layer']
        
        # Escalar coordenadas
        x = t * 2.0
        y = layer * 1.5
        pos[node] = (x, y)

    # Colores
    colors = {
        "env": "#FF9999", # Rojo suave (Mundo Oculto)
        "obs": "#99FF99", # Verde (Input)
        "mem": "#FFD700", # Dorado (LSTM / Memoria)
        "act": "#9999FF"  # Azul (Acción)
    }
    node_colors = [colors[G.nodes[n]['type']] for n in G.nodes()]
    
    # Tamaños: Hacemos el LSTM más grande para resaltar que es el protagonista
    node_sizes = [3000 if G.nodes[n]['type'] == 'mem' else 2000 for n in G.nodes()]

    # Etiquetas
    labels = {n: f"{G.nodes[n]['label']}\n(t={G.nodes[n]['time']})" for n in G.nodes()}

    # Dibujar Nodos
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')

    # Dibujar Aristas
    # Aristas normales
    normal_edges = [e for e in G.edges if G.edges[e].get('color') != 'gold']
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', arrows=True, arrowsize=20)
    
    # Aristas de Memoria (El Flujo LSTM)
    mem_edges = [e for e in G.edges if G.edges[e].get('color') == 'gold']
    nx.draw_networkx_edges(G, pos, edgelist=mem_edges, edge_color='#DAA520', width=3, style='dashed', 
                           connectionstyle="arc3,rad=-0.1", arrows=True, arrowsize=25)

    # Decoración
    plt.title("¿Cómo se entrelazan las observaciones?\nMediante el Estado Oculto del Agente (Línea Dorada)", fontsize=14)
    
    # Leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Estado Latente (Mundo)', markerfacecolor=colors['env'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Observación', markerfacecolor=colors['obs'], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Memoria Agente (LSTM)', markerfacecolor=colors['mem'], markersize=15),
        Line2D([0], [0], color='#DAA520', lw=3, linestyle='--', label='Flujo de Información (Memoria)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_flujo_agente_tiger()