import networkx as nx
import matplotlib.pyplot as plt


def analizar_y_graficar_tiger():
    G = nx.DiGraph()
    steps = [0, 1, 2,3,4,5,6,7,8,9]

    # --- 1. CONSTRUCCIÓN DEL GRAFO ---
    for t in steps:
        s1_node = f"S1_t{t}"
        s2_node = f"S2_t{t}"
        o1_node = f"O1_t{t}"
        o2_node = f"O2_t{t}"
        a_node = f"A_t{t}"

        # Añadimos nodos con atributos para el dibujo
        G.add_node(s1_node, layer=1, time=t, subset=1) # Layer 1: Estados Latentes
        G.add_node(s2_node, layer=1, time=t, subset=2)
        G.add_node(o1_node, layer=2, time=t, subset=1) # Layer 2: Observaciones (Arriba)
        G.add_node(o2_node, layer=2, time=t, subset=2)
        G.add_node(a_node, layer=0, time=t, subset=0)  # Layer 0: Acciones (Abajo)

        # Aristas (S -> O)
        G.add_edge(s1_node, o1_node)
        G.add_edge(s2_node, o2_node)

        # Dinámica temporal
        if t < steps[-1]:
            next_t = t + 1
            # El estado depende de S anterior y A anterior
            G.add_edge(s1_node, f"S1_t{next_t}")
            G.add_edge(a_node, f"S1_t{next_t}")
            
            G.add_edge(s2_node, f"S2_t{next_t}")
            G.add_edge(a_node, f"S2_t{next_t}")

    # --- 2. NUEVAS PRUEBAS CLAVE ---
    print("--- Pruebas de Independencia Condicional ---")

    # PRUEBA CRÍTICA: Independencia cruzada DADO que conocemos la Acción
    # ¿S1 me dice algo sobre O2 si YA SÉ qué acción tomé?
    cruzada_condicional = nx.is_d_separator(G, {"S1_t1"}, {"O2_t2"}, {"A_t0"})
    
    print(f"1. ¿Son S1 y O2 independientes DADO que conozco la Acción A_t0?: {cruzada_condicional}")
    # Si esto es TRUE, ¡BINGO! Puedes separar las redes totalmente.
    
    # PRUEBA DE MEMORIA: ¿Basta con la observación inmediata? (Markovian check)
    # Sin estados latentes, solo O_t1
    es_markoviano = nx.is_d_separator(G, {"O1_t0"}, {"O1_t2"}, {"O1_t1", "A_t1"})
    print(f"2. ¿Es suficiente la observación actual O_t1 para predecir O_t2?: {es_markoviano}")
    # Debería ser False -> Justifica usar LSTM/Memoria.


    # --- 3. VISUALIZACIÓN DEL GRAFO ---
    print("\nGenerando gráfico del modelo...")
    plt.figure(figsize=(12, 8))
    
    # Definir posiciones manualmente para que se vea ordenado (Time Unrolled)
    pos = {}
    for node in G.nodes(data=True):
        name = node[0]
        data = node[1]
        t = data['time']
        layer = data['layer']
        subset = data['subset']
        
        # Eje X: Tiempo (con separación)
        x = t * 2.0
        
        # Eje Y: Capas (Acción abajo, Estado medio, Obs arriba)
        # Hacemos un pequeño offset en Y para separar Tigre 1 de Tigre 2 visualmente
        offset = 0.3 if subset == 1 else -0.3
        if layer == 0: offset = 0 # Acción centrada
            
        y = layer + offset
        pos[name] = (x, y)

    # Colores por tipo de nodo
    color_map = []
    for node in G.nodes():
        if "S" in node: color_map.append('#ff9999') # Rojo suave (Estados)
        elif "O" in node: color_map.append('#99ff99') # Verde suave (Obs)
        elif "A" in node: color_map.append('#9999ff') # Azul suave (Acción)

    # Dibujar
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=color_map, 
            node_size=2000, 
            font_size=9, 
            font_weight='bold', 
            arrows=True,
            arrowsize=20,
            edge_color='gray')
    
    # Etiquetas de tiempo
    plt.text(0, -0.5, "t=0", fontsize=12, ha='center')
    plt.text(2, -0.5, "t=1", fontsize=12, ha='center')
    plt.text(4, -0.5, "t=2", fontsize=12, ha='center')
    
    plt.title("Grafo de Estructura de Información (Tiger Factorizado)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analizar_y_graficar_tiger()