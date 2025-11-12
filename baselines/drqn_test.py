import argparse
import logging
import os
import pickle
import sys
import time
from collections import deque
import numpy as np




import tensorflow as tf
from tensorflow.keras import layers


import gymnasium as gym # 
from gymnasium import spaces 


# # Config GPU on home computer
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")

"""END SLIME ATARI IMPORTS"""

import os
import absl.logging 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Oculta logs de "AVX2 FMA" y GPU
tf.get_logger().setLevel('ERROR') # Oculta logs de Keras
absl.logging.set_verbosity(absl.logging.ERROR)

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger



# def modify_env(env, p=1, k=1):
#     def preprocess(S):
#         im = Image.fromarray(np.uint8(S))
#         im = im.convert('YCbCr')
#         Y, Cb, Cr = im.split()
#         Y = Y.resize(size=(84, 84))
#         Y = np.array(Y)
#         Y = Y[..., np.newaxis]
#         return Y
#
#     def new_reset():
#         S = env.orig_reset()
#         return preprocess(S)
#
#     def new_step(A):
#         for i in range(k):
#             S, R, done, info = env.orig_step(A)
#         if np.random.rand() <= p:
#             return preprocess(S), R, done, info
#         else:
#             return np.zeros(shape=(84, 84, 1)), R, done, info
#
#     env.orig_reset = env.reset
#     env.reset = new_reset
#     env.orig_step = env.step
#     env.step = new_step
#     env.p = p
#     return env


"""
The main class which runs the Q-learning
"""
class Q_Learn:
    def __init__(self, env, network, max_time_steps, jid, weights=None, clone_steps=10000, batch_size=32, gamma=0.95,
                 epsilon=1.0, epsilonStep=18e-5, eval_X=250, buff_len=1000, initT=0, model=None, buffer=None,
                 render=True):
        """
        Initialize the model
        """
        self.jid = jid
        if buffer is None:
            self.buffer = deque(maxlen=buff_len)
        else:
            self.buffer = buffer
        self.env = env
        self.network = network
        self.max_time_steps = max_time_steps
        self.clone_steps = clone_steps
        self.batch_size = batch_size
        self.nA = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonStep = epsilonStep
        self.loss_function = tf.keras.losses.MeanSquaredError()
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue=10.0)
        #
        self.trace_length = 8  # Longitud de la secuencia para el LSTM
        #
        self.S = None
        self.eval_X = eval_X
        self.X = []
        self.Y = []
        self.t = initT
        self.render = render
        if model is None:
            self.init_model()
        else:
            self.model = model
        if weights is not None:
            logger.info(f"Attempting to load weights from {weights}")
            self.model.load_weights(weights)
            logger.info(f"Successfully loaded weights from {weights}")
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def run(self, eval=True):
        """
        Run the algorithm
        """
        
        self.S, info = self.env.reset()
        if self.network == "DRQN":
            self.model.layers[1].reset_states()# Resetea la memoria del LSTM
        
        for _ in range(self.max_time_steps):
            self.t += 1
            if eval and self.t != 0 and self.t % self.eval_X == 0 and self.t > len(self.buffer):
                self.X.append(self.t)
                self.Y.append(self.evaluate())

            if self.render:
                self.env.render()

            if self.t != 0 and self.t % self.clone_steps == 0:
                self.target_model.set_weights(self.model.get_weights())

            self.epsilon = max(0.1, self.epsilon - self.epsilonStep)

            action = self.get_action(self.epsilon)
            logger.info(
                "Step: {}, Action: {}, bufflen: {}, epsilon: {}".format(self.t, action, len(self.buffer), self.epsilon))
            S, R, done, info = self.play_step(action)
            if done:
                self.S, info = self.env.reset()
                if self.network == "DRQN":
                    self.model.layers[1].reset_states()
            if self.t > len(self.buffer):
                self.training_step()
        return self.model

    def play_step(self, action):
        """
        env.step(action) and append result to experience replay buffer
        """
        #S_tag, R, done, info = self.env.step(action)
        #S_tag, R, term, trunc, info = self.env.step(action) 
        S_tag, R, term, trunc, info = self.env.step(action) # <--- API de Gymnasium
        done = term or trunc 
        R = self.clip_reward(R)
        self.buffer.append([self.S, action, R, S_tag, done])
        return S_tag, R, done, info

    def clip_reward(self, R):
        """
        Clip the reward to abs(R)<=1
        """
        if R > 1:
            return 1
        elif R < -1:
            return -1
        else:
            return R

    def get_action(self, epsilon):
        """
        Return an action predicted by the model using an epsilon-greedy policy
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.nA)
        else:
            #Q_values = self.model.predict(self.S[np.newaxis])
            # self.S tiene shape (84, 84, 1)
            # Creamos un batch de shape (1, 1, 84, 84, 1)
            # (batch_size=1, time_steps=1, H, W, C)
            Q_values = self.model.predict(self.S[np.newaxis, np.newaxis], verbose=0)
            return np.argmax(Q_values[0])

    # def training_step(self):
    #     """
    #     Take a training step
    #     """
    #     experiences = self.get_experiences()
    #     states, actions, rewards, next_states, dones = experiences
    #     states = states.astype(np.float32)
    #     next_Q_values = self.target_model.predict(next_states)
    #     max_next_Q_values = np.max(next_Q_values, axis=1)
    #     target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
    #     mask = tf.one_hot(actions, self.nA)
    #     with tf.GradientTape() as tape:
    #         all_Q_values = self.model(states)
    #         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
    #         loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
    #         grads = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    def training_step(self):
        """
        Toma un paso de entrenamiento con secuencias (para DRQN)
        """
        # 1. Muestrear secuencias
        experiences = self.get_experiences()
        states, actions, rewards, next_states, dones = experiences
        
        # 'states' tiene shape (batch_size, trace_length, 84, 84, 1)
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)

        # 2. Calcular Q-values objetivo
        # Predecimos el valor Q para la *siguiente* secuencia de estados
        # Como el modelo es stateful (batch_size=1), debemos iterar
        
        # Almacenamos los Q-values de la target_net
        next_Q_values_batch = []
        for i in range(self.batch_size):
            # Reseteamos el estado para cada nueva secuencia
            self.target_model.layers[1].reset_states()
            # Damos forma (1, trace_length, H, W, C)
            next_Q_sequence = self.target_model.predict(next_states[i:i+1], verbose=0)
            next_Q_values_batch.append(next_Q_sequence[0]) # [0] para tomar el Q-value del último paso
        
        next_Q_values = np.array(next_Q_values_batch)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        
        # 3. Calcular Q-targets (para el último paso de la traza)
        target_Q_values = (
            rewards[:, -1] + (1 - dones[:, -1]) * self.gamma * max_next_Q_values
        )

        # 4. Calcular la pérdida
        # Creamos una máscara para las acciones tomadas en el *último* fotograma
        mask = tf.one_hot(actions[:, -1], self.nA)

        with tf.GradientTape() as tape:
            # Iteramos el entrenamiento, 1 secuencia a la vez
            all_Q_values_batch = []
            for i in range(self.batch_size):
                # Reseteamos el estado para cada nueva secuencia
                self.model.layers[1].reset_states()
                # Damos forma (1, trace_length, H, W, C)
                Q_sequence = self.model(states[i:i+1])
                all_Q_values_batch.append(Q_sequence[0]) # [0] para tomar el Q-value del último paso

            all_Q_values = tf.stack(all_Q_values_batch) # Convertir lista de tensores a un tensor
            
            # Seleccionamos el Q-value de la acción que realmente tomamos
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            
            # Calculamos la pérdida (MSE)
            loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
            
        # 5. Aplicar gradientes
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    # def get_experiences(self):
    #     """
    #     Sample self.batch_size experiences from the experience replay buffer 
    #     """
    #     idxs = np.random.randint(len(self.buffer), size=self.batch_size)
    #     batch = [self.buffer[i] for i in idxs]
    #     states, actions, rewards, next_states, dones = [
    #         np.array([experience[field_index] for experience in batch])
    #         for field_index in range(5)]
    #     return states, actions, rewards, next_states, dones
    
    def get_experiences(self):
        """
        Muestrea 'batch_size' secuencias (trazas) de longitud 'trace_length' del buffer.
        """
        # 1. Crear listas para guardar las secuencias
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for _ in range(self.batch_size):
            # 2. Encontrar un punto de inicio aleatorio
            # Nos aseguramos de tener espacio para 'trace_length' pasos
            start_idx = np.random.randint(0, len(self.buffer) - self.trace_length)
            
            # 3. Extraer la secuencia (traza)
            trace = [self.buffer[i] for i in range(start_idx, start_idx + self.trace_length)]
            
            # 4. Descomprimir la secuencia en sus componentes
            # (ignora si la traza cruza un límite de episodio por ahora,
            # la lógica de 'dones' en el training_step lo maneja)
            s_trace, a_trace, r_trace, ns_trace, d_trace = [
                np.array([experience[field_index] for experience in trace])
                for field_index in range(5)
            ]
            
            states.append(s_trace)
            actions.append(a_trace)
            rewards.append(r_trace)
            next_states.append(ns_trace)
            dones.append(d_trace)

        # 5. Apilar las secuencias en un solo batch
        # El shape de 'states' será (batch_size, trace_length, 84, 84, 1)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def evaluate(self, num_episodes=10, max_episode_time=5):
        """
        Evaluate the learned policy for num_episodes, where each episode is terminated after max_episode_time minutes.
        """
        values = []
        self.save_run()
        for iter in range(num_episodes):
            self.S, info = self.env.reset() # <--- API de Gymnasium
            if self.network == "DRQN":
                self.model.layers[1].reset_states()
                
            value = 0
            start_time = time.time()
            done = False
            while (time.time() - start_time) < max_episode_time * 60 and not done:
                if self.render:
                    self.env.render()
                A = self.get_action(epsilon=0.0)
                # --- ESTA ES LA CORRECCIÓN ---
                S, R, term, trunc, info = self.env.step(A) # Desempaqueta 5 valores
                done = term or trunc # Define 'done'
                # --- FIN DE LA CORRECCIÓN ---
                value += R

            logger.info("Evaluating: iter {} of {}, Value: {}".format(iter, num_episodes, value))
            values.append(value)
        return np.mean(values)

    def save_run(self):
        """
        persist the algorithm's parameters to allow continued run from current state 
        """
        model_fpath = os.path.join("results", str(self.jid), f"model_{self.network}_{self.t}.h5")
        fname = os.path.join("results", str(self.jid), f"data_{self.network}_{self.t}.pickle")
        self.model.save(model_fpath)
        data = {"model_fpath": model_fpath,
                "network": self.network,
                "epsilon": self.epsilon,
                "t": self.t,
                "buffer": self.buffer,
                # "env": self.env,
                "maxsteps": self.max_time_steps,
                "epsilonStep": self.epsilonStep,
                "jid": self.jid,
                "evalx": self.eval_X}
        pickle.dump(data, open(fname, "wb"))

    def init_model(self):
        """
        Inicia el modelo Keras para un entorno tabular (Grid World).
        """
        # Obtenemos el shape de la observación del entorno.
        # Para un grid world tabular, esto debería ser (N,) 
        # donde N es el número de estados (ej. un vector one-hot).
        input_shape = self.env.observation_space.shape
        
        # --- RED DQN (Simple) ---
        if self.network == "DQN":
            self.model = tf.keras.models.Sequential([
                layers.Input(shape=input_shape),
                layers.Dense(units=128, activation="relu"),
                layers.Dense(units=128, activation="relu"),
                layers.Dense(units=self.nA)
            ])
            
        # --- RED DRQN (Recurrente) ---
        elif self.network == "DRQN":
            # La forma de entrada es (batch_size, time_steps, features)
            # Para nuestro LSTM stateful, usamos (1, None, N)
            drqn_input_shape = (1, None,) + input_shape # (1, None, N)

            self.model = tf.keras.models.Sequential([
                layers.Input(batch_shape=drqn_input_shape),
                
                # TimeDistributed aplica la capa Dense a cada paso de la secuencia
                layers.TimeDistributed(
                    layers.Dense(units=128, activation="relu")
                ),
                
                # El LSTM 'stateful'
                layers.LSTM(units=128, stateful=True),
                
                layers.Dense(units=self.nA)
            ])
        
        else:
            raise ValueError

"""
Main
"""
if __name__ == '__main__':
    np.random.seed(0)
#     get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-pickle", help="the path to the pickle file ", type=str, default=None)
    
    #parser.add_argument("-render", help="render display", type=bool, default=False)
    parser.add_argument("-render", help="render display", action="store_true")
    
    parser.add_argument("-env", help="the name of the env to run", type=str, default="SlimeVolleyNoFrameskip-v0")
    parser.add_argument("-network", help="the type of network to load: DQN or DRQN", type=str, default='DQN')
    parser.add_argument("-weights", help="the path to weights to load initially", type=str, default=None)
    parser.add_argument("-initEpsilon", help="the initial epsilon to use", type=float, default=1.0)
    parser.add_argument("-epsilonStep", help="the initial epsilon to use", type=float, default=18e-7)
    parser.add_argument("-initT", help="the initial timestep to use", type=int, default=0)
    parser.add_argument("-jid", help="the slurm job id", type=int)
    parser.add_argument("-bufflen", help="the length of the buffer", type=int, default=10000)
    parser.add_argument("-maxsteps", help="the maximum steps to run", type=int, default=3000000)
    parser.add_argument("-evalx", help="the interval in which to evaluate", type=int, default=2500)
    args = parser.parse_args()
    logger = get_logger(__name__)

#     if args.pickle is not None, continue running from previously saved state
    if args.pickle is not None:
        data = pickle.load(open(args.pickle, "rb"))
        model_fpath = data["model_fpath"]
        network = data["network"]
        epsilon = data["epsilon"]
        t = data["t"]
        buffer = data["buffer"]
        # env = data["env"]
        if args.network == "DQN":
            stack = 4
        elif args.network == "DRQN":
            stack = 1
        env = gym.make("CartPole-v1")  # Placeholder, replace with actual env if needed
        maxsteps = data["maxsteps"]
        epsilonStep = data["epsilonStep"]
        evalx = data["evalx"]
        jid = data["jid"]
        model = tf.keras.models.load_model(model_fpath)

        logger.info(
            f"Continue training jid={args.jid} with network={network}, env={args.env},  initialEpsilon={epsilon}, epsilonStep={epsilonStep}, initT={t}, max_time_steps={maxsteps}, eval_x={evalx}")

        q_learn = Q_Learn(network=network, env=env, model=model, max_time_steps=maxsteps, epsilon=epsilon,
                          epsilonStep=epsilonStep, eval_X=evalx, initT=t, buffer=buffer, render=args.render, jid=jid)
        q_learn.run(eval=True)
        
#         else, begin fresh run
    else:
        results_dir_path = f"results/{args.jid}"
        try:
            os.mkdir(results_dir_path)
        except OSError:
            logger.info("Creation of the directory %s failed" % results_dir_path)
        else:
            logger.info("Successfully created the directory %s" % results_dir_path)

        logger.info(
            f"Begin training jid={args.jid} with network={args.network}, env={args.env}, initialEpsilon={args.initEpsilon}, epsilonStep={args.epsilonStep}, initT={args.initT}, max_time_steps={args.maxsteps}, buff_len={args.bufflen}, eval_x={args.evalx}")

        # stack: how many frames to load per data point
        if args.network == "DQN":
            stack = 4
        elif args.network == "DRQN":
            stack = 1
        env = gym.make("CartPole-v1") 
        q_learn = Q_Learn(env, args.network, args.maxsteps, jid=args.jid, weights=args.weights,
                          epsilon=args.initEpsilon,
                          epsilonStep=args.epsilonStep, initT=args.initT, eval_X=args.evalx, buff_len=args.bufflen,
                          render=args.render)
        model = q_learn.run(eval=True)
        env.close()

        # logger.info("save results (model and scores)")
        # model.save(f'results/{args.jid}/model_{args.network}.h5')
        # np.savetxt(f"results/{args.jid}/GraphA_{args.network}_X.csv", q_learn.X)
        # np.savetxt(f"results/{args.jid}/GraphA_{args.network}_Y.csv", q_learn.Y)