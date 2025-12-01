import networkx as nx
import matplotlib.pyplot as plt

# Configuración de importación robusta para d_separated
try:
    d_separated = nx.d_separated
except AttributeError:
    try:
        from networkx.algorithms.d_separation import d_separated
    except ImportError:
        try:
            from networkx.algorithms.d_separation import is_d_separator as d_separated
        except ImportError:
             print("Error crítico: No se encuentra d_separated en networkx")
             exit()

def analizar_key_door_maze_styled():
    G = nx.DiGraph()
    
    # Definimos 3 momentos clave en el tiempo
    times = ["T_Key", "T_Mid", "T_Door"]
    
    # --- 1. CONSTRUCCIÓN DEL GRAFO ---
    for i, t in enumerate(times):
        # Nombres de nodos únicos
        p_node = f"Pos_{t}"
        k_node = f"Key_{t}"
        o_node = f"Obs_{t}"
        a_node = f"Act_{t}"
        
        # Agregamos nodos con metadatos para el estilo
        # 'type': define el color
        # 'label': texto corto para mostrar
        G.add_node(p_node, type="Pos", time=i, label="Pos")
        G.add_node(k_node, type="Key", time=i, label="Key")
        G.add_node(o_node, type="Obs", time=i, label="Obs")
        G.add_node(a_node, type="Act", time=i, label="Act")
        
        # Aristas de Observación (El estado genera la imagen)
        G.add_edge(p_node, o_node)
        G.add_edge(k_node, o_node) 
        
        # Dinámica Temporal (hacia el futuro)
        if i < len(times) - 1:
            next_t = times[i+1]
            next_p = f"Pos_{next_t}"
            next_k = f"Key_{next_t}"
            
            # Evolución de la Posición
            G.add_edge(p_node, next_p)
            G.add_edge(a_node, next_p)
            G.add_edge(k_node, next_p) # La llave afecta si puedes cruzar puertas
            
            # Evolución de la Llave (MEMORIA)
            # Marcamos esta arista como 'persistente' para dibujarla diferente
            G.add_edge(k_node, next_k, style='persistente') 
            G.add_edge(p_node, next_k)

    # --- 2. CÁLCULO DE D-SEPARACIÓN (La Ciencia) ---
    pasado_lejano = {f"Key_T_Key", f"Pos_T_Key"}
    futuro = {f"Pos_T_Door"}
    memoria_corta = {f"Obs_T_Mid", f"Act_T_Mid"} # FrameStacking
    
    es_markoviano = d_separated(G, pasado_lejano, futuro, memoria_corta)
    print(f"¿DQN FrameStacking es suficiente?: {es_markoviano}")
    
    # Prueba 2: ¿Funciona reconstruir el Estado Latente (S_t)?
    estado_latente = {f"Pos_T_Mid", f"Key_T_Mid"}
    es_suficiente_lstm = d_separated(G, pasado_lejano, futuro, estado_latente)

    print(f"2. ¿Funciona inferir el Estado Latente (Pos + Key)?: {es_suficiente_lstm}")

    # --- 3. ESTILO Y DIBUJO (El Arte) ---
    plt.figure(figsize=(14, 8))
    
    # A. Posiciones Manuales (Grid Layout)
    pos = {}
    for node, data in G.nodes(data=True):
        t = data['time']
        # Definimos alturas fijas (Y) para cada tipo de variable
        if data['type'] == "Obs": y = 3.5  # Arriba del todo
        elif data['type'] == "Key": y = 2.2 # Medio-Alto
        elif data['type'] == "Pos": y = 1.0 # Medio-Bajo
        elif data['type'] == "Act": y = 0.0 # Abajo del todo
        
        # Separación horizontal (X)
        x = t * 2.5
        pos[node] = (x, y)

    # B. Asignación de Colores y Tamaños
    color_map = []
    node_sizes = []
    labels = {}
    
    # Paleta de colores profesional
    colors = {
        "Pos": '#87CEFA', # Azul Cielo
        "Key": '#FFD700', # Oro
        "Obs": '#98FB98', # Verde Pálido
        "Act": '#FA8072'  # Salmon
    }

    for node, data in G.nodes(data=True):
        tipo = data['type']
        color_map.append(colors[tipo])
        
        # Hacemos los nodos de Estado (Key/Pos) más grandes para resaltar importancia
        size = 2800 if tipo in ["Key", "Pos"] else 2000
        node_sizes.append(size)
        
        # Etiqueta limpia: "Nombre\n(Tiempo)"
        labels[node] = f"{data['label']}\n({times[t]})"

    # C. Dibujar Nodos
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_sizes, 
                           edgecolors='black', linewidths=1.5)
    
    # D. Dibujar Etiquetas
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", font_family="sans-serif")
    
    # E. Dibujar Aristas (Separando estilos)
    
    # 1. Aristas Normales (Física inmediata)
    normal_edges = [e for e in G.edges if 'style' not in G.edges[e]]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', 
                           arrows=True, arrowsize=25, width=1.5, alpha=0.6)
    
    # 2. Aristas de Persistencia (La MEMORIA CRÍTICA)
    persist_edges = [e for e in G.edges if 'style' in G.edges[e]]
    nx.draw_networkx_edges(G, pos, edgelist=persist_edges, edge_color='#DAA520', 
                           arrows=True, arrowsize=30, width=3.0, style='dashed', 
                           connectionstyle="arc3,rad=-0.1") # Curvada ligeramente

    # F. Leyenda y Títulos
    plt.title(f"Grafo de Estructura de Información: KeyDoorMazeEnv\nResultado Teórico: FrameStacking Suficiente = {es_markoviano}", fontsize=14, fontweight='bold')
    
    # Crear leyenda personalizada
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Input: Observación (Imagen 3x3)', 
                   markerfacecolor=colors['Obs'], markersize=15, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', label='Latente 1: Llave (Dinámica Lenta/Global)', 
                   markerfacecolor=colors['Key'], markersize=15, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', label='Latente 2: Posición (Dinámica Rápida/Local)', 
                   markerfacecolor=colors['Pos'], markersize=15, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', label='Input: Acción', 
                   markerfacecolor=colors['Act'], markersize=15, markeredgecolor='black'),
        plt.Line2D([0], [0], color='#DAA520', lw=3, linestyle='--', label='Flujo de Memoria (Persistencia)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Tipos de Variables")

    plt.axis('off') # Ocultar ejes
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analizar_key_door_maze_styled()