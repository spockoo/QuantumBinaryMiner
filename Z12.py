import numpy as np
import cirq
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import keras_tuner as kt
import os
import time
import pickle
import gc
import random
import string
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from subprocess import Popen

# Configuration des chemins
MINER_PATH = "PATH/TOFOLDER/NBminer_Win"
MINER_EXECUTABLE = "nbminer.exe"
POOL_URL = "POOL"
USER = "USER"
PASSWORD = "x"

INTERCEPT_CONSTANT = 0.60
BATCH_SIZE = 10  # Taille des lots pour écrire sur disque

MODEL_FILE_PATH = "model.keras"
CIRCUIT_FILE_PATH = "quantum_circuit.pkl"

# Fréquence initiale pour la simulation quantique
quantum_circuit_repetitions = 10
initial_qubits = 2

def generate_unique_filename(prefix="file", extension=".log"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    filename = f"{prefix}_{timestamp}_{random_id}{extension}"
    return os.path.join(os.getcwd(), filename)

def log_data(data, log_file_path):
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(data + "\n")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des données: {e}")

def save_file(obj, file_path, log_file_path, mode='wb'):
    try:
        with open(file_path, mode) as file:
            pickle.dump(obj, file)
        log_data(f"Fichier sauvegardé à {file_path}", log_file_path)
    except Exception as e:
        log_data(f"Erreur lors de la sauvegarde du fichier: {e}", log_file_path)

def load_file(file_path, log_file_path, mode='rb'):
    try:
        if os.path.isfile(file_path):
            with open(file_path, mode) as file:
                obj = pickle.load(file)
            log_data(f"Fichier chargé depuis {file_path}", log_file_path)
            return obj
        else:
            log_data(f"Le fichier n'existe pas à {file_path}", log_file_path)
            return None
    except Exception as e:
        log_data(f"Erreur lors du chargement du fichier: {e}", log_file_path)
        return None

def build_model(hp, input_shape):
    model = Sequential([
        Dense(hp.Int('units', min_value=64, max_value=128, step=32), activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def hyperparameter_tuning(X, y):
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, X.shape[1]),
        objective='val_loss',
        max_epochs=5,
        directory='tuner',
        project_name='hyperparameter_tuning'
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    best_model = tuner.get_best_models(num_models=1)[0]
    best_params = tuner.get_best_hyperparameters(num_trials=1)[0].values

    return best_model, best_params

def retry_on_failure(func, max_attempts=3, *args, **kwargs):
    attempts = 0
    while attempts < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempts += 1
            log_data(f"Tentative {attempts}/{max_attempts} échouée pour {func.__name__}: {e}", kwargs.get('log_file_path', ''))
            if attempts == max_attempts:
                log_data(f"Échec permanent de {func.__name__} après {max_attempts} tentatives.", kwargs.get('log_file_path', ''))
                raise e
            time.sleep(1)  # Réduire l'attente avant de réessayer

def train_tf_model(X, y, log_file_path, existing_model=None):
    if existing_model:
        return fine_tune_model(existing_model, X, y, log_file_path)
    model, params = hyperparameter_tuning(X, y)
    return fine_tune_model(model, X, y, log_file_path)

def fine_tune_model(model, X_train, y_train, log_file_path):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    try:
        start_time = time.time()
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred.flatten()) ** 2))
        r2 = r2_score(y_val, y_pred)
        log_data(f"Temps de fine-tuning pour cette itération: {time.time() - start_time:.2f} secondes", log_file_path)
        clear_session()  # Libère la mémoire utilisée par les graphes de TensorFlow
        gc.collect()  # Appelle le garbage collector manuellement
        return model, rmse, r2
    except Exception as e:
        log_data(f"Erreur pendant le fine-tuning du modèle: {e}", log_file_path)
        return None, None, None

def start_mining(log_file_path):
    global process
    miner_executable = os.path.join(MINER_PATH, MINER_EXECUTABLE)
    if not os.path.isfile(miner_executable):
        log_data(f"Erreur: Exécutable '{miner_executable}' non trouvé.", log_file_path)
        return
    command = [
        miner_executable,
        "-a", "kawpow",
        "-o", POOL_URL,
        "-u", USER,
        "-p", PASSWORD
    ]
    log_data(f"Exécution de la commande: {command}", log_file_path)
    try:
        if os.name == 'nt':  # Windows
            process = Popen(
                ["cmd.exe", "/c"] + command,
                cwd=MINER_PATH
            )
        else:
            process = Popen(
                command,
                cwd=MINER_PATH
            )
    except Exception as e:
        log_data(f"Erreur lors du démarrage du processus: {e}", log_file_path)

def stop_mining(log_file_path):
    global process
    if process:
        try:
            process.terminate()
            process.wait()
        except Exception as e:
            log_data(f"Erreur lors de l'arrêt du processus: {e}", log_file_path)
        finally:
            process = None

def create_memmap_array(filename, shape, dtype):
    return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

def collect_data_from_miner(log_file_path):
    try:
        if os.path.isfile("X_data.dat") and os.path.isfile("y_data.dat"):
            X_data[:] = np.random.randn(*X_data.shape)
            y_data[:] = np.random.randn(*y_data.shape)
            log_data("Données collectées et mises à jour à partir du miner.", log_file_path)
    except Exception as e:
        log_data(f"Erreur lors de la collecte des données du miner: {e}", log_file_path)

def calculate_gradient_and_intercept(X, y):
    if X.size == 0 or y.size == 0:
        return None, None
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None, None
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, None
    model = LinearRegression().fit(X, y)
    intercept = model.intercept_
    gradient = model.coef_
    return intercept, gradient

def match_matrices(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Les matrices doivent avoir les mêmes dimensions pour le match.")
    # Effectuer des opérations sur les matrices pour les associer ou les comparer
    matched_matrix = matrix1 @ matrix2.T  # Exemple de produit matriciel
    return matched_matrix

def simulate_quantum_circuit(log_file_path, repetitions, qubits_count):
    try:
        qubits = cirq.LineQubit.range(qubits_count)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key='result')
        ])

        save_file(circuit, CIRCUIT_FILE_PATH, log_file_path, mode='wb')

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=repetitions)
        log_data(f"Résultats de la simulation quantique (répétitions={repetitions}, qubits={qubits_count}): {result}", log_file_path)

    except Exception as e:
        log_data(f"Erreur lors de la simulation quantique: {e}", log_file_path)

def simulate_quantum_circuit_for_plot(repetitions, qubits_count):
    qubits = cirq.LineQubit.range(qubits_count)
    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key='result')
    ])

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=repetitions)
    return result.measurements['result'].flatten()

def quantum_error_correction(log_file_path, qubits_count):
    try:
        qubits = cirq.LineQubit.range(qubits_count)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.measure(*qubits, key='result')
        ])

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=quantum_circuit_repetitions)
        log_data(f"Résultats de la correction d'erreurs quantiques (qubits={qubits_count}): {result}", log_file_path)

    except Exception as e:
        log_data(f"Erreur lors de la correction d'erreurs quantiques: {e}", log_file_path)

def update_graph(frame, log_file_path, fig, ax1, ax2, ax3, ax4, ax5, ax6):
    global X_data, y_data, rmse_data, intercept_data, r2_data, current_model, quantum_circuit_repetitions, initial_qubits

    collect_data_from_miner(log_file_path)
    
    # Augmenter le nombre de qubits dynamiquement
    qubits_count = initial_qubits + (frame // 10)  # Augmente les qubits tous les 10 frames
    simulate_quantum_circuit(log_file_path, quantum_circuit_repetitions, qubits_count)
    quantum_error_correction(log_file_path, qubits_count)
    
    # Augmenter la fréquence pour la prochaine simulation
    quantum_circuit_repetitions += 10

    if X_data.shape[0] < 2 or y_data.shape[0] < 2:
        return

    if X_data.size == 0 or y_data.size == 0 or X_data.shape[0] != y_data.shape[0] or X_data.shape[0] < 2:
        log_data("Erreur: Dimensions des données inconsistantes ou échantillons insuffisants.", log_file_path)
        return

    if len(X_data) >= 10:
        model, rmse, r2 = retry_on_failure(train_tf_model, max_attempts=3, X=X_data, y=y_data, log_file_path=log_file_path, existing_model=current_model)
        if model:
            current_model = model
            save_file(model, MODEL_FILE_PATH, log_file_path, mode='wb')
            rmse_data.append(rmse)
            r2_data.append(r2)
            intercept, gradient = calculate_gradient_and_intercept(X_data, y_data)
            if intercept is not None and gradient is not None:
                intercept_data.append(intercept)

            ax1.clear()
            ax1.plot(rmse_data, label='RMSE')
            ax1.set_title('Erreur Quadratique Moyenne (RMSE)')
            ax1.set_xlabel('Itération')
            ax1.set_ylabel('RMSE')
            ax1.legend()

            ax2.clear()
            ax2.plot(intercept_data, label='Intercept')
            ax2.set_title('Intercept du Modèle')
            ax2.set_xlabel('Itération')
            ax2.set_ylabel('Intercept')
            ax2.legend()

            ax3.clear()
            ax3.plot(np.array(intercept_data) + INTERCEPT_CONSTANT, label='Intercept Ajusté')
            ax3.set_title('Intercept Ajusté')
            ax3.set_xlabel('Itération')
            ax3.set_ylabel('Intercept Ajusté')
            ax3.legend()

            qubit_results = simulate_quantum_circuit_for_plot(quantum_circuit_repetitions, initial_qubits + (frame // 10))
            ax4.clear()
            ax4.hist(qubit_results, bins=2, range=(0, 1), label='Résultats de la Simulation Quantique')
            ax4.set_title('Histogramme des Résultats de la Simulation Quantique')
            ax4.set_xlabel('Résultat')
            ax4.set_ylabel('Fréquence')
            ax4.legend()

            if X_data.shape[0] > 0:
                samples_idx = np.arange(len(y_data))
                ax5.clear()
                ax5.plot(samples_idx, y_data, 'b.', label='Valeur Réelle')
                ax5.set_title('Échantillons a/b vs Valeur Réelle')
                ax5.set_xlabel('Index d\'échantillon')
                ax5.set_ylabel('Valeur Réelle')
                ax5.legend()

            ax6.clear()
            ax6.plot(r2_data, label='R^2')
            ax6.set_title('Coefficient de Détermination (R^2)')
            ax6.set_xlabel('Itération')
            ax6.set_ylabel('R^2')
            ax6.legend()

            plt.tight_layout()
            gc.collect()  # Appelle le garbage collector après la mise à jour du graphique

def main():
    global X_data, y_data, batch_X, batch_y, rmse_data, intercept_data, r2_data, current_model
    global process, quantum_circuit_repetitions, initial_qubits

    log_file_path = generate_unique_filename()
    log_data(f"Début de l'exécution du script à {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file_path)

    X_data = create_memmap_array("X_data.dat", (1000, 10), dtype='float32')
    y_data = create_memmap_array("y_data.dat", (1000,), dtype='float32')
    batch_X = np.zeros((BATCH_SIZE, 10), dtype='float32')
    batch_y = np.zeros(BATCH_SIZE, dtype='float32')
    rmse_data = []
    intercept_data = []
    r2_data = []
    current_model = load_file(MODEL_FILE_PATH, log_file_path, mode='r')

    # Configuration TensorFlow pour utiliser le GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            log_data("GPU configuré pour utiliser la croissance de mémoire dynamique.", log_file_path)
        except Exception as e:
            log_data(f"Erreur lors de la configuration du GPU: {e}", log_file_path)

    start_mining(log_file_path)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 18))
    ani = FuncAnimation(fig, update_graph, fargs=(log_file_path, fig, ax1, ax2, ax3, ax4, ax5, ax6), interval=500, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        log_data("Interruption de l'utilisateur détectée.", log_file_path)
    finally:
        stop_mining(log_file_path)
        log_data(f"Fin de l'exécution du script à {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file_path)

if __name__ == "__main__":
    main()
