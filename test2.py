import networkx as nx


def analizar_cuello_botella_tiger_completo():
    G = nx.DiGraph()

    # Horizonte de tiempo: t=0 (Pasado), t=1 (Presente/Candidato), t=2 (Futuro)
    steps = [0, 1, 2,3,4,5,6,7,8,9]

    # --- 1. CONSTRUCCIÓN DEL GRAFO (Dinámica Causal) ---
    for t in steps:
        # --- Nodos ---
        # Estados Latentes (S): Dónde están los tigres (Variable NO observable)
        s1_node = f"S1_t{t}"
        s2_node = f"S2_t{t}"
        
        # Observaciones (O): El rugido que escuchas (Variable Observable)
        o1_node = f"O1_t{t}"
        o2_node = f"O2_t{t}"
        
        # Acciones (A): Qué puerta abres (Variable Observable)
        # Nota: La acción ocurre AL FINAL del paso t y afecta al paso t+1
        a_node = f"A_t{t}"

        # Agregamos nodos
        G.add_nodes_from([s1_node, s2_node, o1_node, o2_node, a_node])

        # --- Aristas (Causalidad) ---
        
        # A. Emisión: El Estado causa la Observación (S -> O)
        G.add_edge(s1_node, o1_node)
        G.add_edge(s2_node, o2_node)

        # B. Transición: El Estado y la Acción actual causan el Estado siguiente
        if t < steps[-1]:
            next_t = t + 1
            
            # Dinámica Tigre 1: S1(t) + Acción(t) -> S1(t+1)
            G.add_edge(s1_node, f"S1_t{next_t}")
            G.add_edge(a_node, f"S1_t{next_t}")
            
            # Dinámica Tigre 2: S2(t) + Acción(t) -> S2(t+1)
            G.add_edge(s2_node, f"S2_t{next_t}")
            G.add_edge(a_node, f"S2_t{next_t}")

            # NOTA: En este grafo, asumimos que una sola acción afecta a ambos tigres
            # (ej. si reinicias el juego). Si tienes acciones separadas (A1, A2), 
            # deberías crear nodos A1_t y A2_t y conectarlos solo a su tigre respectivo.

    # --- 2. DEFINICIÓN DE CONJUNTOS DE VARIABLES ---

    # X: El Pasado Observable (Historia)
    # Incluye observaciones Y acciones pasadas. Esto es lo que tiene tu LSTM en t=0.
    pasado_observable = {"O1_t0", "O2_t0", "A_t0"}

    # Y: El Futuro Observable
    # Lo que queremos predecir (observaciones futuras).
    futuro_observable = {"O1_t2", "O2_t2"}

    # Z: Los Candidatos a Estado Info-Estructural (Cuello de Botella)
    # Estas son las variables en t=1 que "cortan" el flujo.
    
    # Candidato 1: Estado Latente Conjunto (S1 y S2)
    candidato_estado_full = {"S1_t1", "S2_t1"}
    
    # Candidato 2: Solo Estado del Tigre 1
    candidato_estado_parcial = {"S1_t1"}
    
    # Candidato 3: Solo la historia observable reciente (sin estado latente)
    candidato_memoria = {"O1_t1", "O2_t1", "A_t1"}


    # --- 3. ANÁLISIS DE D-SEPARACIÓN ---
    print("--- Análisis de Estructura de Información (Tiger Environment) ---\n")

    # Prueba 1: ¿El Estado Latente Completo es suficiente?
    # Pregunta: ¿Están X e Y d-separados dado Z?
    es_bottleneck = nx.is_d_separator(G, pasado_observable, futuro_observable, candidato_estado_full)
    print(f"¿Es {{S1, S2}} un cuello de botella válido?: {es_bottleneck}")
    # Debería ser TRUE. Si sabes dónde están los tigres, el pasado (A_t0, O_t0) es irrelevante.

    # Prueba 2: ¿Basta con saber solo dónde está el Tigre 1?
    es_parcial = nx.is_d_separator(G, pasado_observable, futuro_observable, candidato_estado_parcial)
    print(f"¿Es {{S1}} suficiente para predecir TODO el futuro?: {es_parcial}")
    # Debería ser FALSE. S1 corta la info del Tigre 1, pero la info del Tigre 2 se "fuga" por el otro lado.

    # --- 4. ANÁLISIS DE INDEPENDENCIA (FACTORIZACIÓN) ---
    print("\n--- Análisis para Arquitectura Factorizada ---")
    
    # Aquí verificamos si podemos separar el problema en dos redes neuronales.
    # Pregunta: ¿El Estado S1 bloquea la información del PASADO DEL TIGRE 1 hacia el FUTURO DEL TIGRE 1?
    
    # Definimos sub-conjuntos específicos para el Tigre 1
    pasado_t1 = {"O1_t0", "A_t0"} # Asumimos que A afecta a ambos, así que es parte del pasado de ambos
    futuro_t1 = {"O1_t2"}
    bottleneck_t1 = {"S1_t1"}
    
    separa_t1 = nx.is_d_separator(G, pasado_t1, futuro_t1, bottleneck_t1)
    print(f"¿S1 separa la historia del Tigre 1 de su futuro?: {separa_t1}")
    
    # Verificación cruzada: ¿El estado S1 nos dice algo sobre el futuro del Tigre 2?
    # (Si son independientes, S1 no debería d-separar nada del Tigre 2 porque no hay camino,
    # pero más importante, no debería haber conexión directa).
    # En d-separation, comprobamos si hay camino activo.
    
    hay_camino_cruzado = not nx.is_d_separator(G, {"S1_t1"}, {"O2_t2"}, set())
    print(f"¿Hay camino causal directo de S1 a O2?: {hay_camino_cruzado}")
    # Debería ser FALSE (o True si no condicionamos en nada y hay un ancestro común, 
    # pero en este diseño S1 y S2 son independientes dada la acción).

if __name__ == "__main__":
    analizar_cuello_botella_tiger_completo()