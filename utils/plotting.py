import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

#Definizione della figura e il nome del file da leggere
recording_name= "sensor_data_FD-CD-F2-2A-C9-6A.csv"
skip_rows=0
fig, (accelerometer_fig, gyroscope_fig) = plt.subplots(2, 1, figsize=(16,10)) #subplots definisce quante rige e quante colonne ci sono in una schermata, 2 righe e 1 colonna

def animate(i):
    global skip_rows, recording_name
    if not os.path.exists(recording_name):
        #print(f'Non trovato: {recording_name}')
        return

    df=pd.read_csv( #pandas legge file csv cioè una tabella con ogni riga a un istante di tempo diverso
        recording_name,
        header=None,
        names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"], #qua ho dato delle etichette alle colonne, quando accedo alla variabile acc_x mi porto dietro anche il tempo, mi da sia la colonna acc_x che quella del tempo
        index_col=0,
        parse_dates=True,
        skiprows=skip_rows
    )

    if len(df) >= 600: #serve per aggiornare l'indice da skippare, ogni volta taglio gli ultimi 600 elementi perchè li avevo già mostrati prima
        skip_rows += len(df) - 600 #vedo sempre la coda finale del file appresentata dagli ultimi 600 dati

    accelerometer_fig.clear()
    gyroscope_fig.clear()

    accelerometer_fig.plot(df["acc_x"], color="red", label="X") #passa a plot solo la colonna con etichetta acc_x
    accelerometer_fig.plot(df["acc_y"], color="green", label="Y")
    accelerometer_fig.plot(df["acc_z"], color="blue", label="Z")
    accelerometer_fig.set_yticks(np.arange(-2, 2.5, 0.5))
    accelerometer_fig.set_ylabel("Magnitude")
    accelerometer_fig.set_xlabel("Time")
    accelerometer_fig.set_title("Accelerometer Data")
    accelerometer_fig.legend()

    gyroscope_fig.plot(df["gyro_x"], color="cyan", label="X")
    gyroscope_fig.plot(df["gyro_y"], color="magenta", label="Y")
    gyroscope_fig.plot(df["gyro_z"], color="yellow", label="Z")
    gyroscope_fig.set_yticks(np.arange(-1200, 1800, 600))
    gyroscope_fig.set_ylabel("Magnitude")
    gyroscope_fig.set_xlabel("Time")
    gyroscope_fig.set_title("Gyroscope Data")
    gyroscope_fig.legend()

    fig.suptitle(f"Wearable Data", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, top=0.88)

def live_plotting():
    ani = FuncAnimation(fig, animate, interval=60, cache_frame_data=False) #intervallo -> ogni 60 millisecondi il plot viene aggiornato
    plt.show()

