import numpy as np
import matplotlib.pyplot as plt
import serial
import time

# ==========================================
# 1. DEFINICIÓN DE PARÁMETROS Y ARQUITECTURA
# ==========================================

INPUTS = 4
LAYER1 = 2
LAYER2 = 3
OUTPUTS = 1

# IMPORTANTE:
# Debe ser el mismo valor que LEARNING_RATE del MCU.
LEARNING_RATE = 0.5
#Cantidad de datos
DATASET_SIZE = 100
#Epocas a realizar
EPOCHS = 100


# ==========================================
# 2. FUNCIONES DE ACTIVACIÓN Y DERIVADAS
# ==========================================

def sigmoid(x, a, b, c, d):
    return (a / (1.0 + b * np.exp(-c * x))) + d


def sigmoid_derivative(x, a, b, c):
    e = np.exp(-c * x)
    denominator = 1.0 + b * e
    return (a * b * c * e) / (denominator ** 2)


# ==========================================
# 3. INICIALIZACIÓN DE PARÁMETROS
# ==========================================

def get_initial_parameters():

    W1 = np.array([
        [0.01, -0.02, 0.03, -0.01],
        [-0.03, 0.02, 0.01, 0.03]
    ], dtype=np.float32)

    b1 = np.array([0.0, 0.0], dtype=np.float32)

    W2 = np.array([
        [0.02, -0.01],
        [-0.03, 0.02],
        [0.01, 0.03]
    ], dtype=np.float32)

    b2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    W3 = np.array([
        [0.02, -0.02, 0.01]
    ], dtype=np.float32)

    b3 = np.array([0.0], dtype=np.float32)

    return W1, b1, W2, b2, W3, b3


# Parámetros diferentes de la sigmoide para cada neurona.
sig_a = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 1.0], dtype=np.float32)
sig_b = np.array([1.0, 0.8, 1.1, 0.9, 1.2, 1.0], dtype=np.float32)
sig_c = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0], dtype=np.float32)
sig_d = np.array([0.0, 0.0, 0.05, 0.0, 0.02, 0.0], dtype=np.float32)


# ==========================================
# 4. FORWARD PROPAGATION
# ==========================================

def forward_propagation(X, W1, b1, W2, b2, W3, b3):

    Z1 = np.zeros(LAYER1, dtype=np.float32)
    A1 = np.zeros(LAYER1, dtype=np.float32)

    for i in range(LAYER1):
        Z1[i] = b1[i] + np.dot(W1[i], X)
        A1[i] = sigmoid(
            Z1[i],
            sig_a[i],
            sig_b[i],
            sig_c[i],
            sig_d[i]
        )

    Z2 = np.zeros(LAYER2, dtype=np.float32)
    A2 = np.zeros(LAYER2, dtype=np.float32)

    for i in range(LAYER2):
        Z2[i] = b2[i] + np.dot(W2[i], A1)
        A2[i] = sigmoid(
            Z2[i],
            sig_a[i + 2],
            sig_b[i + 2],
            sig_c[i + 2],
            sig_d[i + 2]
        )

    Z3 = b3[0] + np.dot(W3[0], A2)

    A3 = sigmoid(
        Z3,
        sig_a[5],
        sig_b[5],
        sig_c[5],
        sig_d[5]
    )

    return Z1, A1, Z2, A2, Z3, A3


# ==========================================
# 5. BACKPROPAGATION
# ==========================================

def backward_propagation(
    X, Y, Z1, A1, Z2, A2, Z3, A3,
    W1, W2, W3
):

    dCost_dA3 = -Y / A3 + (1.0 - Y) / (1.0 - A3)

    # Capa 3
    dZ3 = dCost_dA3 * sigmoid_derivative(
        Z3, sig_a[5], sig_b[5], sig_c[5]
    )

    dW3 = np.zeros_like(W3)
    dW3[0] = dZ3 * A2

    db3 = np.array([dZ3], dtype=np.float32)

    # Capa 2
    dA2 = W3[0] * dZ3

    dZ2 = np.zeros(LAYER2, dtype=np.float32)

    for i in range(LAYER2):
        dZ2[i] = dA2[i] * sigmoid_derivative(
            Z2[i],
            sig_a[i + 2],
            sig_b[i + 2],
            sig_c[i + 2]
        )

    dW2 = np.zeros_like(W2)
    db2 = np.zeros(LAYER2, dtype=np.float32)

    for i in range(LAYER2):
        dW2[i] = dZ2[i] * A1
        db2[i] = dZ2[i]

    # Capa 1
    dA1 = np.zeros(LAYER1, dtype=np.float32)

    for j in range(LAYER1):
        dA1[j] = np.sum(W2[:, j] * dZ2)

    dZ1 = np.zeros(LAYER1, dtype=np.float32)

    for i in range(LAYER1):
        dZ1[i] = dA1[i] * sigmoid_derivative(
            Z1[i],
            sig_a[i],
            sig_b[i],
            sig_c[i]
        )

    dW1 = np.zeros_like(W1)
    db1 = np.zeros(LAYER1, dtype=np.float32)

    for i in range(LAYER1):
        dW1[i] = dZ1[i] * X
        db1[i] = dZ1[i]

    return dW1, db1, dW2, db2, dW3, db3


# ==========================================
# 6. FUNCIÓN DE COSTO
# ==========================================

def compute_cost(A3, Y):

    epsilon = 1.0e-7

    A3_clamped = np.clip(
        A3,
        epsilon,
        1.0 - epsilon
    )

    return -(
        Y * np.log(A3_clamped)
        + (1.0 - Y) * np.log(1.0 - A3_clamped)
    )


# ==========================================
# 7. DATASET
# ==========================================

def generate_dataset(size=DATASET_SIZE):

    np.random.seed(42)

    X = np.random.uniform(
        0.0,
        1.0,
        size=(size, INPUTS)
    ).astype(np.float32)

    Y = (
        (X[:, 0] + X[:, 1])
        > (X[:, 2] + X[:, 3])
    ).astype(np.float32)

    return X, Y


# ==========================================
# 8. ENTRENAMIENTO DE REFERENCIA EN PC
# ==========================================

def train_network(X, Y, epochs=EPOCHS, lr=LEARNING_RATE):

    W1, b1, W2, b2, W3, b3 = get_initial_parameters()

    cost_history = []
    accuracy_history = []

    for epoch in range(epochs):

        epoch_cost = 0.0
        correct_predictions = 0

        for k in range(len(X)):

            X_sample = X[k]
            Y_sample = Y[k]

            # Forward
            Z1, A1, Z2, A2, Z3, A3 = forward_propagation(
                X_sample,
                W1, b1,
                W2, b2,
                W3, b3
            )

            # Costo ANTES de actualizar los pesos.
            cost = compute_cost(A3, Y_sample)
            epoch_cost += float(cost)

            # Clasificación
            prediction = 1.0 if A3 >= 0.5 else 0.0

            if prediction == Y_sample:
                correct_predictions += 1

            # Backward
            dW1, db1, dW2, db2, dW3, db3 = backward_propagation(
                X_sample,
                Y_sample,
                Z1, A1,
                Z2, A2,
                Z3, A3,
                W1, W2, W3
            )

            # Update
            W1 -= lr * dW1
            b1 -= lr * db1

            W2 -= lr * dW2
            b2 -= lr * db2

            W3 -= lr * dW3
            b3 -= lr * db3

        avg_cost = epoch_cost / len(X)
        accuracy = (
            correct_predictions / len(X)
        ) * 100.0

        cost_history.append(avg_cost)
        accuracy_history.append(accuracy)

    return (
        W1, b1,
        W2, b2,
        W3, b3,
        cost_history,
        accuracy_history
    )


# ==========================================
# 9. ENTRENAMIENTO DEL MCU POR UART
# ==========================================

def train_mcu_uart(
    X_dataset,
    Y_dataset,
    port="COM3",
    baudrate=115200,
    epochs=EPOCHS
):
    """
    Entrena la red directamente en el MCU.

    Por cada muestra:
        PC -> X1,X2,X3,X4,Y
        MCU -> A3,cost

    El MCU actualiza sus pesos después de calcular
    A3 y cost.

    Por lo tanto, Python puede hacer exactamente
    la misma operación en paralelo para comparar.
    """

    # ------------------------------------------
    # Abrir UART
    # ------------------------------------------

    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=2.0
    )

    time.sleep(1.0)

    # Limpiar cualquier dato anterior.
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("UART conectada.")
    print("Comenzando entrenamiento en MCU...")
    print(f"Épocas: {epochs}")
    print(f"Muestras por época: {len(X_dataset)}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # ------------------------------------------
    # Pesos de referencia de Python.
    #
    # Estos pesos avanzan muestra por muestra
    # igual que los del MCU.
    # ------------------------------------------

    W1, b1, W2, b2, W3, b3 = get_initial_parameters()

    # ------------------------------------------
    # Historial
    # ------------------------------------------

    mcu_loss_history = []
    pc_loss_history = []

    mcu_accuracy_history = []
    pc_accuracy_history = []

    epoch_output_error = []
    sample_results = []

    epoch_time_mcu = []
    total_start = time.perf_counter()

    # ==========================================
    # LOOP DE ÉPOCAS
    # ==========================================

    for epoch in range(epochs):

        mcu_epoch_cost = 0.0
        pc_epoch_cost = 0.0

        mcu_correct = 0
        pc_correct = 0

        comparison_errors = []

        epoch_start = time.perf_counter()

        # --------------------------------------
        # LOOP DE MUESTRAS
        # --------------------------------------

        for k in range(len(X_dataset)):

            X_sample = X_dataset[k]
            Y_sample = Y_dataset[k]

            # ==================================
            # FORWARD PYTHON
            # ==================================

            (
                Z1, A1,
                Z2, A2,
                Z3, A3_python
            ) = forward_propagation(
                X_sample,
                W1, b1,
                W2, b2,
                W3, b3
            )

            cost_python = compute_cost(
                A3_python,
                Y_sample
            )

            # ==================================
            # PREPARAR MENSAJE
            # ==================================

            message = (
                f"{float(X_sample[0]):.9g},"
                f"{float(X_sample[1]):.9g},"
                f"{float(X_sample[2]):.9g},"
                f"{float(X_sample[3]):.9g},"
                f"{float(Y_sample):.0f}\n"
            )

            # ==================================
            # ENVIAR AL MCU
            # ==================================

            tx_start = time.perf_counter()

            ser.write(
                message.encode("ascii")
            )

            # ==================================
            # RECIBIR A3,COST
            # ==================================

            response = (
                ser.readline()
                .decode("ascii")
                .strip()
            )

            tx_end = time.perf_counter()

            if response == "":
                print(
                    f"ERROR: timeout en época "
                    f"{epoch + 1}, muestra {k}"
                )
                continue

            if response == "ERROR":
                print(
                    f"ERROR: parsing en MCU. "
                    f"Época {epoch + 1}, muestra {k}"
                )
                continue

            try:

                A3_str, cost_mcu_str = response.split(",")

                A3_mcu = np.float32(
                    float(A3_str)
                )

                cost_mcu = np.float32(
                    float(cost_mcu_str)
                )

            except ValueError:

                print(
                    f"Respuesta inválida del MCU: "
                    f"{response}"
                )

                continue

            # ==================================
            # COMPARACIÓN PYTHON vs MCU
            # ==================================

            output_error = abs(
                float(A3_python)
                - float(A3_mcu)
            )

            cost_error = abs(
                float(cost_python)
                - float(cost_mcu)
            )

            comparison_errors.append(
                output_error
            )

            # ==================================
            # ACCURACY
            # ==================================

            pc_prediction = (
                1.0
                if A3_python >= 0.5
                else 0.0
            )

            mcu_prediction = (
                1.0
                if A3_mcu >= 0.5
                else 0.0
            )

            if pc_prediction == Y_sample:
                pc_correct += 1

            if mcu_prediction == Y_sample:
                mcu_correct += 1

            # ==================================
            # ACUMULAR COSTOS
            # ==================================

            pc_epoch_cost += float(cost_python)
            mcu_epoch_cost += float(cost_mcu)

            # ==================================
            # GUARDAR DATOS
            # ==================================

            sample_results.append({
                "epoch": epoch + 1,
                "sample": k,
                "Y": float(Y_sample),
                "A3_python": float(A3_python),
                "A3_mcu": float(A3_mcu),
                "cost_python": float(cost_python),
                "cost_mcu": float(cost_mcu),
                "output_error": output_error,
                "cost_error": cost_error,
                "uart_time_ms": (
                    tx_end - tx_start
                ) * 1000.0
            })

            # ==================================
            # ACTUALIZAR PYTHON
            #
            # IMPORTANTE:
            # El MCU también actualiza sus pesos
            # después de enviar A3 y cost.
            # ==================================

            (
                dW1, db1,
                dW2, db2,
                dW3, db3
            ) = backward_propagation(
                X_sample,
                Y_sample,
                Z1, A1,
                Z2, A2,
                Z3, A3_python,
                W1, W2, W3
            )

            W1 -= LEARNING_RATE * dW1
            b1 -= LEARNING_RATE * db1

            W2 -= LEARNING_RATE * dW2
            b2 -= LEARNING_RATE * db2

            W3 -= LEARNING_RATE * dW3
            b3 -= LEARNING_RATE * db3

        # ======================================
        # RESULTADOS DE LA ÉPOCA
        # ======================================

        avg_mcu_cost = (
            mcu_epoch_cost / len(X_dataset)
        )

        avg_pc_cost = (
            pc_epoch_cost / len(X_dataset)
        )

        mcu_accuracy = (
            mcu_correct / len(X_dataset)
        ) * 100.0

        pc_accuracy = (
            pc_correct / len(X_dataset)
        ) * 100.0

        epoch_elapsed = (
            time.perf_counter()
            - epoch_start
        )

        mcu_loss_history.append(
            avg_mcu_cost
        )

        pc_loss_history.append(
            avg_pc_cost
        )

        mcu_accuracy_history.append(
            mcu_accuracy
        )

        pc_accuracy_history.append(
            pc_accuracy
        )

        epoch_output_error.append(
            np.mean(comparison_errors)
            if comparison_errors
            else np.nan
        )

        epoch_time_mcu.append(
            epoch_elapsed
        )

        # Mostrar cada 10 épocas.
        if (
            (epoch + 1) % 10 == 0
            or epoch == 0
        ):

            print(
                f"Época {epoch + 1:3d}/{epochs} | "
                f"Loss PC: {avg_pc_cost:.6f} | "
                f"Loss MCU: {avg_mcu_cost:.6f} | "
                f"Acc PC: {pc_accuracy:6.2f}% | "
                f"Acc MCU: {mcu_accuracy:6.2f}% | "
                f"Error A3: "
                f"{epoch_output_error[-1]:.3e}"
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    ser.close()

    print()
    print("==========================================")
    print("ENTRENAMIENTO TERMINADO")
    print("==========================================")
    print(
        f"Tiempo total PC + UART: "
        f"{total_elapsed:.3f} s"
    )

    return {
        "mcu_loss": mcu_loss_history,
        "pc_loss": pc_loss_history,
        "mcu_accuracy": mcu_accuracy_history,
        "pc_accuracy": pc_accuracy_history,
        "epoch_error": epoch_output_error,
        "epoch_time": epoch_time_mcu,
        "samples": sample_results,
        "total_time": total_elapsed
    }


# ==========================================
# 10. GRÁFICA: LOSS PC VS MCU
# ==========================================

def plot_loss(history):

    epochs = np.arange(
        1,
        len(history["pc_loss"]) + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["pc_loss"],
        label="PC"
    )

    plt.plot(
        epochs,
        history["mcu_loss"],
        "--",
        label="MCU"
    )

    plt.title(
        "Evolución del costo: PC vs MCU"
    )

    plt.xlabel("Época")
    plt.ylabel("Costo medio (Binary Cross-Entropy)")

    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 11. GRÁFICA: ACCURACY PC VS MCU
# ==========================================

def plot_accuracy(history):

    epochs = np.arange(
        1,
        len(history["pc_accuracy"]) + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["pc_accuracy"],
        label="PC"
    )

    plt.plot(
        epochs,
        history["mcu_accuracy"],
        "--",
        label="MCU"
    )

    plt.title(
        "Accuracy durante el entrenamiento"
    )

    plt.xlabel("Época")
    plt.ylabel("Accuracy (%)")

    plt.ylim(0, 100)

    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 12. GRÁFICA: ERROR PC-MCU
# ==========================================

def plot_epoch_error(history):

    epochs = np.arange(
        1,
        len(history["epoch_error"]) + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["epoch_error"]
    )

    plt.title(
        "Error medio entre PC y MCU"
    )

    plt.xlabel("Época")
    plt.ylabel("|A3 PC - A3 MCU|")

    plt.yscale("log")

    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()


# ==========================================
# 13. COMPARACIÓN DE SALIDA EN ÚLTIMA ÉPOCA
# ==========================================

def plot_final_outputs(history):

    samples = [
        r["sample"] + 1
        for r in history["samples"]
        if r["epoch"] == EPOCHS
    ]

    pc_output = [
        r["A3_python"]
        for r in history["samples"]
        if r["epoch"] == EPOCHS
    ]

    mcu_output = [
        r["A3_mcu"]
        for r in history["samples"]
        if r["epoch"] == EPOCHS
    ]

    y_real = [
        r["Y"]
        for r in history["samples"]
        if r["epoch"] == EPOCHS
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        samples,
        y_real,
        "o",
        label="Y real"
    )

    plt.plot(
        samples,
        pc_output,
        label="A3 PC"
    )

    plt.plot(
        samples,
        mcu_output,
        "--",
        label="A3 MCU"
    )

    plt.axhline(
        0.5,
        linestyle=":"
    )

    plt.title(
        "Salida final: PC vs MCU"
    )

    plt.xlabel("Muestra")
    plt.ylabel("Salida A3")

    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 14. ERROR POR MUESTRA EN ÚLTIMA ÉPOCA
# ==========================================

def plot_final_sample_error(history):

    final_results = [
        r
        for r in history["samples"]
        if r["epoch"] == EPOCHS
    ]

    samples = [
        r["sample"] + 1
        for r in final_results
    ]

    errors = [
        r["output_error"]
        for r in final_results
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        samples,
        errors
    )

    plt.title(
        "Diferencia de salida entre PC y MCU "
        "en la última época"
    )

    plt.xlabel("Muestra")
    plt.ylabel("|A3 PC - A3 MCU|")

    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()


# ==========================================
# 15. LOSS POR MUESTRA EN ÚLTIMA ÉPOCA
# ==========================================

def plot_final_sample_loss(history):

    final_results = [
        r
        for r in history["samples"]
        if r["epoch"] == EPOCHS
    ]

    samples = [
        r["sample"] + 1
        for r in final_results
    ]

    pc_loss = [
        r["cost_python"]
        for r in final_results
    ]

    mcu_loss = [
        r["cost_mcu"]
        for r in final_results
    ]

    plt.figure(figsize=(12, 5))

    plt.plot(
        samples,
        pc_loss,
        label="PC"
    )

    plt.plot(
        samples,
        mcu_loss,
        "--",
        label="MCU"
    )

    plt.title(
        "Costo por muestra en la última época"
    )

    plt.xlabel("Muestra")
    plt.ylabel("Costo")

    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# 16. TIEMPO POR ÉPOCA
# ==========================================

def plot_epoch_time(history):

    epochs = np.arange(
        1,
        len(history["epoch_time"]) + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["epoch_time"]
    )

    plt.title(
        "Tiempo total de comunicación por época"
    )

    plt.xlabel("Época")
    plt.ylabel("Tiempo (s)")

    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()


# ==========================================
# 17. RESUMEN FINAL
# ==========================================

def print_final_summary(history):

    pc_loss = history["pc_loss"][-1]
    mcu_loss = history["mcu_loss"][-1]

    pc_acc = history["pc_accuracy"][-1]
    mcu_acc = history["mcu_accuracy"][-1]

    errors = np.array(
        [
            r["output_error"]
            for r in history["samples"]
        ]
    )

    cost_errors = np.array(
        [
            r["cost_error"]
            for r in history["samples"]
        ]
    )

    uart_times = np.array(
        [
            r["uart_time_ms"]
            for r in history["samples"]
        ]
    )

    print()
    print("==========================================")
    print("RESUMEN FINAL")
    print("==========================================")

    print(f"Épocas                 : {EPOCHS}")
    print(f"Muestras por época     : {DATASET_SIZE}")
    print(f"Muestras totales       : {len(history['samples'])}")

    print()
    print(f"Loss final PC          : {pc_loss:.8f}")
    print(f"Loss final MCU         : {mcu_loss:.8f}")
    print(
        f"Diferencia de loss     : "
        f"{abs(pc_loss - mcu_loss):.8e}"
    )

    print()
    print(f"Accuracy final PC      : {pc_acc:.2f}%")
    print(f"Accuracy final MCU     : {mcu_acc:.2f}%")

    print()
    print(
        f"Error A3 máximo        : "
        f"{np.max(errors):.8e}"
    )

    print(
        f"Error A3 medio        : "
        f"{np.mean(errors):.8e}"
    )

    print(
        f"Error de costo máximo  : "
        f"{np.max(cost_errors):.8e}"
    )

    print(
        f"Error de costo medio   : "
        f"{np.mean(cost_errors):.8e}"
    )

    print()
    print(
        f"Tiempo UART medio/muestra : "
        f"{np.mean(uart_times):.3f} ms"
    )

    print(
        f"Tiempo UART máximo/muestra: "
        f"{np.max(uart_times):.3f} ms"
    )

    print(
        f"Tiempo total PC + UART   : "
        f"{history['total_time']:.3f} s"
    )


# ==========================================
# 18. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":

    # --------------------------------------
    # Generar dataset
    # --------------------------------------

    X_dataset, Y_dataset = generate_dataset(
        DATASET_SIZE
    )

    print("Dataset generado.")
    print(
        f"Muestras: {len(X_dataset)}"
    )

    print(
        f"Learning rate PC: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Épocas: {EPOCHS}"
    )

    print()

    # --------------------------------------
    # IMPORTANTE:
    #
    # Reinicia el MCU ANTES de ejecutar
    # este script para que sus pesos comiencen
    # desde los mismos valores iniciales.
    #
    # Durante las 100 épocas NO reiniciar.
    # --------------------------------------

    history = train_mcu_uart(
        X_dataset,
        Y_dataset,
        port="COM3",
        baudrate=115200,
        epochs=EPOCHS
    )

    # --------------------------------------
    # Resumen
    # --------------------------------------

    print_final_summary(history)

    # --------------------------------------
    # Gráficas
    # --------------------------------------

    plot_loss(history)

    plot_accuracy(history)

    plot_epoch_error(history)

    plot_final_outputs(history)

    plot_final_sample_error(history)

    plot_final_sample_loss(history)

    plot_epoch_time(history)
