import networkx as nx


def analizar_cuello_botella_tiger():
    # 1. Crear un Grafo Dirigido (DAG)
    G = nx.DiGraph()

    # Definimos el horizonte de tiempo para el análisis (ej. 3 pasos: t=0, t=1, t=2)
    # Queremos ver si el estado en t=1 corta la información entre t=0 y t=2.
    steps = [0, 1, 2]

    # --- CONSTRUCCIÓN DEL GRAFO (LA FÍSICA DEL ENTORNO) ---
    for t in steps:
        # Nodos para el Tigre 1
        s1_node = f"S1_t{t}"  # Estado Latente Tigre 1 (Izq/Der)
        o1_node = f"O1_t{t}"  # Observación Rugido 1 (Izq/Der)
        
        # Nodos para el Tigre 2
        s2_node = f"S2_t{t}"  # Estado Latente Tigre 2 (Izq/Der)
        o2_node = f"O2_t{t}"  # Observación Rugido 2 (Izq/Der)

        # Agregamos los nodos al grafo
        G.add_node(s1_node)
        G.add_node(o1_node)
        G.add_node(s2_node)
        G.add_node(o2_node)

        # ARISTA: El Estado causa la Observación (S -> O)
        G.add_edge(s1_node, o1_node)
        G.add_edge(s2_node, o2_node)

        # ARISTA: Transición temporal (El estado actual causa el siguiente)
        if t < steps[-1]:
            # Tigre 1 evoluciona por su cuenta
            G.add_edge(f"S1_t{t}", f"S1_t{t+1}")
            # Tigre 2 evoluciona por su cuenta
            G.add_edge(f"S2_t{t}", f"S2_t{t+1}")
            
            # NOTA IMPORTANTE: Fíjate que NO he puesto aristas cruzadas.
            # S1 no afecta a S2. S1 no afecta a O2.
            # Esta es la "Estructura de Información" de tu entorno.

    # --- DEFINICIÓN DE CONJUNTOS (PASADO, FUTURO, CANDIDATO) ---
    
    # 1. El Pasado Observable (Todo lo que vi en t=0)
    pasado_observable = {"O1_t0", "O2_t0"}
    
    # 2. El Futuro Observable (Todo lo que veré en t=2)
    futuro_observable = {"O1_t2", "O2_t2"}
    
    # 3. Los Candidatos a Cuello de Botella (En t=1)
    # Prueba A: Solo mirar las observaciones anteriores (lo que hace un Frame Stacking)
    candidato_A = {"O1_t1", "O2_t1"} 
    
    # Prueba B: Conocer el Estado Latente Conjunto completo
    candidato_B = {"S1_t1", "S2_t1"}
    
    # Prueba C: Conocer SOLO el Estado del Tigre 1 (¿Es suficiente para todo?)
    candidato_C = {"S1_t1"}

    # --- EL CÁLCULO (D-SEPARACIÓN) ---
    print("--- Análisis de Cuello de Botella (Tiger Environment) ---")
    
    # Chequeo A: ¿Bastan las observaciones pasadas?
    es_separado_A = nx.is_d_separator(G, pasado_observable, futuro_observable, candidato_A)
    print(f"Candidato A (Solo Observaciones t=1): {es_separado_A}")
    # Resultado esperado: FALSE. Porque la obs es ruidosa, no contiene toda la info del estado.

    # Chequeo B: ¿Basta el Estado Latente Completo?
    es_separado_B = nx.is_d_separator(G, pasado_observable, futuro_observable, candidato_B)
    print(f"Candidato B (Estado Latente Completo t=1): {es_separado_B}")
    # Resultado esperado: TRUE. Si sé dónde están los tigres, no me importa el pasado.

    # Chequeo C: ¿Basta solo el Estado del Tigre 1?
    es_separado_C = nx.is_d_separator(G, pasado_observable, futuro_observable, candidato_C)
    print(f"Candidato C (Solo Estado Tigre 1): {es_separado_C}")
    # Resultado esperado: FALSE. Cortas el flujo de info del Tigre 1, pero la info del Tigre 2 sigue fluyendo por su lado.

    # --- ANÁLISIS FACTORIZADO (LA CLAVE PARA TU AGENTE) ---
    print("\n--- Análisis Factorizado ---")
    # Pregunta: ¿El Estado S1 separa el Pasado del Tigre 1 del Futuro del Tigre 1?
    sep_factorizada = nx.is_d_separator(G, {"O1_t0"}, {"O1_t2"}, {"S1_t1"})
    print(f"¿S1 separa la sub-tarea del Tigre 1?: {sep_factorizada}")
    # TRUE. Esto prueba que puedes resolver el problema 1 ignorando el problema 2.

if __name__ == "__main__":
    analizar_cuello_botella_tiger()