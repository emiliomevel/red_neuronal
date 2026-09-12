
#include "Clock_Ip.h"
#include "Siul2_Port_Ip.h"
#include "Lpuart_Uart_Ip.h"
#include "IntCtrl_Ip.h"
#include "Lpuart_Uart_Ip_Irq.h"

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define SIUL2_CONFIG g_pin_mux_InitConfigArr_PortContainer_0_BOARD_InitPeripherals
#define SIUL2_PINS NUM_OF_CONFIGURED_PINS_PortContainer_0_BOARD_InitPeripherals
#define UART_INSTANCE 6 /*Instancia de LPUART*/
#define BUFFER_SIZE 256U /* Tamaño del buffer*/
#define INPUTS 4 /*Entradas*/
#define LAYER1 2 /*Capa 1, 2 neuronas*/
#define LAYER2 3 /*Capa 2, 3 neuronas*/
#define OUTPUTS 1 /*Capa 3, 1 neurona*/

/*Tasa de apredizaje*/
#define LEARNING_RATE 0.5f


/*Pesos */
float W1[LAYER1][INPUTS];
float W2[LAYER2][LAYER1];
float W3[OUTPUTS][LAYER2];

/*Bias*/
float b1[LAYER1];
float b2[LAYER2];
float b3[OUTPUTS];

/* Gradientes */
float dW1[LAYER1][INPUTS];
float dW2[LAYER2][LAYER1];
float dW3[OUTPUTS][LAYER2];

float db1[LAYER1];
float db2[LAYER2];
float db3[OUTPUTS];

/* Parametros de la funcion sigmoid */
float sig_a[6];
float sig_b[6];
float sig_c[6];
float sig_d[6];

/*Entrada de la capa 3 y salida*/
float Z3;
float A3;
/*Valor experimental de los datos*/
float Y;
float cost;

/*Buffer para comunicacion serial*/
volatile uint8 BufferIdx = 0U;
uint8 au8Buffer[BUFFER_SIZE];
uint8 au8TxBuffer[BUFFER_SIZE];
volatile uint8 bRxFlag = 0;


/*Funciones */

float sigmoid(
		float x,
		float a,
		float b,
		float c,
		float d);
float sigmoid_derivative(
		float x,
		float a,
		float b,
		float c);

void initialize_parameters(void);

void initialize_sigmoid_parameters(void);

void forward_propagation(
		float X[INPUTS],
		float Z1[LAYER1],
		float A1[LAYER1],
		float Z2[LAYER2],
		float A2[LAYER2],
		float *Z3,
		float *A3);

void backward_propagation(
		float X[INPUTS],
        float Y,
        float Z1[LAYER1],
        float A1[LAYER1],
        float Z2[LAYER2],
        float A2[LAYER2],
        float Z3,
        float A3);

void update_parameters(void);

float compute_cost(float A3, float Y);

void Uart_Callback(
		const uint8 HwInstance,
		const Lpuart_Uart_Ip_EventType Event,
		const void *UserData);

static bool parse_uart_sample(uint8 *buffer, float X[INPUTS], float *Y);

static void process_uart_sample(void);

int main(void)
{


	/* Init clock configuration and awake the clock source */
	Clock_Ip_Init(Clock_Ip_aClockConfig);
	Clock_Ip_InitClock(Clock_Ip_aClockConfig);
	while(CLOCK_IP_PLL_LOCKED != Clock_Ip_GetPllStatus()){
		__asm volatile ("nop");
	}
	Clock_Ip_DistributePll();

	/* Init interrupt hardware */
    IntCtrl_Ip_Init(&IntCtrlConfig_0);
    IntCtrl_Ip_EnableIrq(LPUART6_IRQn);

    /* Init Siul2 ports */
    Siul2_Port_Ip_Init(SIUL2_PINS, SIUL2_CONFIG);

    /* Init Uart peripherical */
    Lpuart_Uart_Ip_Init(UART_INSTANCE, &Lpuart_Uart_Ip_xHwConfigPB_6);
    /*Inicia recibiendo el primer byte*/
    Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, au8Buffer, 1U);
    /*inicializacion de parametros*/
    initialize_parameters();
    initialize_sigmoid_parameters();


	while(1){
		if (bRxFlag == 1U)
		    {
		        /* Procesar la muestra */
		        process_uart_sample();

		        /* Limpiar variables de control RX */
		        BufferIdx = 0U;
		        bRxFlag = 0U;
		        memset(au8Buffer, 0, sizeof(au8Buffer));

		        /* Reiniciar recepción de 1 byte */
		        Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, au8Buffer, 1U);
		    }
	}
	return 0;
}

float sigmoid(float x, float a, float b, float c, float d)
{
    return (a / (1.0f + b * expf(-c * x))) + d;
}

float sigmoid_derivative(
    float x,
    float a,
    float b,
    float c)
{
    float e;
    float denominator;

    e = expf(-c * x);
    denominator = 1.0f + b * e;

    return (a * b * c * e) /
           (denominator * denominator);
}
void initialize_parameters(void)
{
    /* W1: 2 x 4 */
    W1[0][0] =  0.01f;
    W1[0][1] = -0.02f;
    W1[0][2] =  0.03f;
    W1[0][3] = -0.01f;

    W1[1][0] = -0.03f;
    W1[1][1] =  0.02f;
    W1[1][2] =  0.01f;
    W1[1][3] =  0.03f;

    /* Bias primera capa */
    b1[0] = 0.0f;
    b1[1] = 0.0f;


    /* W2: 3 x 2 */
    W2[0][0] =  0.02f;
    W2[0][1] = -0.01f;

    W2[1][0] = -0.03f;
    W2[1][1] =  0.02f;

    W2[2][0] =  0.01f;
    W2[2][1] =  0.03f;

    /* Bias segunda capa */
    b2[0] = 0.0f;
    b2[1] = 0.0f;
    b2[2] = 0.0f;


    /* W3: 1 x 3 */
    W3[0][0] =  0.02f;
    W3[0][1] = -0.02f;
    W3[0][2] =  0.01f;

    /* Bias salida */
    b3[0] = 0.0f;
}

void initialize_sigmoid_parameters(void)
{
    /* Capa 1 */
    sig_a[0] = 1.0f;
    sig_b[0] = 1.0f;
    sig_c[0] = 1.0f;
    sig_d[0] = 0.0f;

    sig_a[1] = 1.2f;
    sig_b[1] = 0.8f;
    sig_c[1] = 1.1f;
    sig_d[1] = 0.0f;

    /* Capa 2 */
    sig_a[2] = 0.9f;
    sig_b[2] = 1.1f;
    sig_c[2] = 0.9f;
    sig_d[2] = 0.05f;

    /*Capa 3 */
    sig_a[3] = 1.1f;
    sig_b[3] = 0.9f;
    sig_c[3] = 1.2f;
    sig_d[3] = 0.0f;

    /*Capa 4 */
    sig_a[4] = 1.0f;
    sig_b[4] = 1.2f;
    sig_c[4] = 0.8f;
    sig_d[4] = 0.02f;

    /* Salida */
    sig_a[5] = 1.0f;
    sig_b[5] = 1.0f;
    sig_c[5] = 1.0f;
    sig_d[5] = 0.0f;
}

void forward_propagation(
        float X[INPUTS],
        float Z1[LAYER1],
        float A1[LAYER1],
        float Z2[LAYER2],
        float A2[LAYER2],
        float *Z3,
        float *A3)
{
    int i;
    int j;

    /* -------- Capa 1 -------- */

    for (i = 0; i < LAYER1; i++)
    {
        Z1[i] = b1[i];

        for (j = 0; j < INPUTS; j++)
        {
            Z1[i] += W1[i][j] * X[j];
        }

        A1[i] = sigmoid(
        		Z1[i],
				sig_a[i],
				sig_b[i],
				sig_c[i],
				sig_d[i]);
    }


    /* -------- Capa 2 -------- */

    for (i = 0; i < LAYER2; i++)
    {
        Z2[i] = b2[i];

        for (j = 0; j < LAYER1; j++)
        {
            Z2[i] += W2[i][j] * A1[j];
        }

        A2[i] = sigmoid(
        		Z2[i],
				sig_a[i + 2],
				sig_b[i + 2],
				sig_c[i + 2],
				sig_d[i + 2]);
    }


    /* -------- Capa 3 / salida -------- */

    *Z3 = b3[0];

    for (j = 0; j < LAYER2; j++)
    {
        *Z3 += W3[0][j] * A2[j];
    }

    *A3 = sigmoid(
    		*Z3,
			sig_a[5],
			sig_b[5],
			sig_c[5],
			sig_d[5]);
}

void backward_propagation(
        float X[INPUTS],
        float Y,
        float Z1[LAYER1],
        float A1[LAYER1],
        float Z2[LAYER2],
        float A2[LAYER2],
        float Z3,
        float A3)
{
    int i;
    int j;

    float dZ3;
    float dA2[LAYER2];
    float dZ2[LAYER2];

    float dA1[LAYER1];
    float dZ1[LAYER1];

    float dCost_dA3;


    /* =====================================================
     * CAPA DE SALIDA
     * ===================================================== */

    /*
     * Derivada de Binary Cross Entropy respecto a A3:
     *
     * dJ/dA3 = -Y/A3 + (1-Y)/(1-A3)
     */

    dCost_dA3 = -Y / A3 + (1.0f - Y) / (1.0f - A3);

    /*
     * dZ3 = dJ/dA3 * dA3/dZ3
     *
     * Como la Sigmoid de salida tiene parametros:
     * a = sig_a[5]
     * b = sig_b[5]
     * c = sig_c[5]
     */

    dZ3 = dCost_dA3 * sigmoid_derivative(
    		Z3,
            sig_a[5],
            sig_b[5],
            sig_c[5]
        );


    /* Gradientes de W3 y b3 */

    for (j = 0; j < LAYER2; j++)
    {
        dW3[0][j] = dZ3 * A2[j];
    }

    db3[0] = dZ3;


    /* =====================================================
     * CAPA 2
     * ===================================================== */

    /*
     * Propagamos el error hacia A2
     */

    for (j = 0; j < LAYER2; j++)
    {
        dA2[j] = W3[0][j] * dZ3;
    }


    /*
     * Aplicamos la derivada de la Sigmoid
     * correspondiente a cada neurona.
     */

    for (i = 0; i < LAYER2; i++)
    {
        dZ2[i] = dA2[i] * sigmoid_derivative(
        		Z2[i],
                sig_a[i + 2],
                sig_b[i + 2],
                sig_c[i + 2]);
    }


    /* Gradientes W2 y b2 */

    for (i = 0; i < LAYER2; i++)
    {
        for (j = 0; j < LAYER1; j++)
        {
            dW2[i][j] = dZ2[i] * A1[j];
        }

        db2[i] = dZ2[i];
    }


    /* =====================================================
     * CAPA 1
     * ===================================================== */

    /*
     * Propagamos el error hacia A1
     */

    for (j = 0; j < LAYER1; j++)
    {
        dA1[j] = 0.0f;

        for (i = 0; i < LAYER2; i++)
        {
            dA1[j] += W2[i][j] * dZ2[i];
        }
    }


    /*
     * Aplicamos la derivada de la Sigmoid
     * de cada neurona de la capa 1.
     */

    for (i = 0; i < LAYER1; i++)
    {
        dZ1[i] = dA1[i] * sigmoid_derivative(
                Z1[i],
                sig_a[i],
                sig_b[i],
                sig_c[i]
            );
    }


    /* Gradientes W1 y b1 */

    for (i = 0; i < LAYER1; i++)
    {
        for (j = 0; j < INPUTS; j++)
        {
            dW1[i][j] = dZ1[i] * X[j];
        }

        db1[i] = dZ1[i];
    }
}

void update_parameters(void)
{
    int i;
    int j;

    /* Actualizar W1 y b1 */

    for (i = 0; i < LAYER1; i++)
    {
        for (j = 0; j < INPUTS; j++)
        {
            W1[i][j] = W1[i][j] - LEARNING_RATE * dW1[i][j];
        }

        b1[i] = b1[i] - LEARNING_RATE * db1[i];
    }


    /* Actualizar W2 y b2 */

    for (i = 0; i < LAYER2; i++)
    {
        for (j = 0; j < LAYER1; j++)
        {
            W2[i][j] = W2[i][j] - LEARNING_RATE * dW2[i][j];
        }

        b2[i] = b2[i] - LEARNING_RATE * db2[i];
    }


    /* Actualizar W3 y b3 */

    for (i = 0; i < OUTPUTS; i++)
    {
        for (j = 0; j < LAYER2; j++)
        {
            W3[i][j] = W3[i][j] - LEARNING_RATE * dW3[i][j];
        }

        b3[i] = b3[i] - LEARNING_RATE * db3[i];
    }
}

float compute_cost(float A3, float Y)
{
    float epsilon = 1.0e-7f;

    /* Evitar log(0) */
    if (A3 < epsilon)
        A3 = epsilon;

    if (A3 > (1.0f - epsilon))
        A3 = 1.0f - epsilon;

    return -(Y * logf(A3) + (1.0f - Y) * logf(1.0f - A3));
}

/* =====================================================
   FUNCIONES CORREGIDAS DE COMUNICACIÓN UART
   ===================================================== */

void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData)
{
    (void)UserData;
    (void)HwInstance;

    switch (Event)
    {
        case LPUART_UART_IP_EVENT_RX_FULL:
            /* Detectar fin de linea \n o \r */
            if ((au8Buffer[BufferIdx] == '\n') || (au8Buffer[BufferIdx] == '\r'))
            {
                au8Buffer[BufferIdx] = '\0'; /* Terminar la cadena correctamente */
                bRxFlag = 1U;
            }
            else if (BufferIdx < (BUFFER_SIZE - 2U))
            {
                BufferIdx++;
                /* Solicitar el siguiente byte */
                Lpuart_Uart_Ip_SetRxBuffer(UART_INSTANCE, &au8Buffer[BufferIdx], 1U);
            }
            else
            {
                /* Buffer lleno, forzar fin */
                au8Buffer[BufferIdx] = '\0';
                bRxFlag = 1U;
            }
            break;

        case LPUART_UART_IP_EVENT_TX_EMPTY:
        case LPUART_UART_IP_EVENT_END_TRANSFER:
        case LPUART_UART_IP_EVENT_ERROR:
        default:
            __asm volatile ("nop");
            break;
    }
}

static bool parse_uart_sample(uint8 *buffer, float X[INPUTS], float *Y)
{
    /* Parseo seguro usando sscanf en lugar de strtok para evitar corromper la memoria */
    int parsed = sscanf((char *)buffer, "%f,%f,%f,%f,%f", &X[0], &X[1], &X[2], &X[3], Y);

    return (parsed == 5);
}

static void process_uart_sample(void)
{
    float X[INPUTS];
    float Z1[LAYER1], A1[LAYER1];
    float Z2[LAYER2], A2[LAYER2];
    float Y_rx;
    int txLength;

    /* Parsear datos recibidos */
    if (parse_uart_sample(au8Buffer, X, &Y_rx) == false)
    {
        txLength = snprintf((char *)au8TxBuffer, BUFFER_SIZE, "ERROR\n");
        Lpuart_Uart_Ip_AsyncSend(UART_INSTANCE, au8TxBuffer, (uint32)txLength);
        return;
    }

    Y = Y_rx;

    /* Propagación hacia adelante */
    forward_propagation(X, Z1, A1, Z2, A2, &Z3, &A3);
    cost = compute_cost(A3, Y);

    backward_propagation(
        X,
        Y,
        Z1,
        A1,
        Z2,
        A2,
        Z3,
        A3
    );

    update_parameters();
    /* Enviar A3 de vuelta a Python con formato preciso */

    txLength = snprintf(
        (char *)au8TxBuffer,
        BUFFER_SIZE,
        "%.7f,%.7f\n",
        (double)A3,
        (double)cost
    );
    Lpuart_Uart_Ip_AsyncSend(UART_INSTANCE, au8TxBuffer, (uint32)txLength);
}

