import tkinter as tk
from tkinter import filedialog
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

theme = plt.get_cmap('viridis')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

#tkinter theme 


model  = pickle.load(open('./models/medium-model/model_trained.pkl', 'rb'))
n_components = 1
time_steps = 100
n_future = 40


def main():
    root = tk.Tk()
    root.title("Crypto Price Prediction")
    root.geometry("1080x720")
    root.configure(background='black')
    #color of the text
    root.option_add('*Label*foreground', 'white')
    root.option_add('*Label*background', 'black')
    root.option_add('*Button*foreground', 'white')
    root.option_add('*Button*background', 'blue')
    
    #load file
    def buscar_archivo():
        # Abrir el cuadro de diálogo para seleccionar un archivo
        ruta_archivo = filedialog.askopenfilename()
        ruta_archivo_limpia = ruta_archivo.strip() 
        # Mostrar la ruta del archivo seleccionado en la etiqueta
        if ruta_archivo_limpia:
            etiqueta.config(text=f"Archivo seleccionado: {ruta_archivo_limpia}")

    boton = tk.Button(root, text="Buscar archivo", command=buscar_archivo)
    boton.pack(pady=20)

    etiqueta = tk.Label(root, text="Ningún archivo seleccionado")
    etiqueta.pack(pady=20)

    #predict
    def predecir():
        ruta_archivo = etiqueta.cget("text")
        ruta_archivo_limpia = ruta_archivo.strip()

        if ruta_archivo_limpia == "Ningún archivo seleccionado":
            return
        if ruta_archivo_limpia == "Archivo seleccionado: ":
            return
        #if not is csv file
        if ruta_archivo_limpia[-4:] != '.csv':
            return

        data = pd.read_csv(ruta_archivo_limpia[22:], sep=';')
        data = preprocess_data(data)
        result = use_model(data)
        ax.clear()
        #print(data['close'])
        
        result=result[len(data):]
        max_close_id = result['close'].idxmax()

        resultLabel.config(text=f"El precio máximo se alcanzará el {max_close_id.strftime('%Y-%m-%d')} y será de {round(result['close'].max(),2)} teniendo su precio actual en {round(data['close'].iloc[-1],2)} su precio se multiplicará por {round(result['close'].max()/data['close'].iloc[-1],2)}")

        ax.axvline(x=pd.to_datetime("2024-04-20"), color='g', linestyle='--', label='Halving 2024')
        ax.axvline(x=pd.to_datetime("2024-04-20")+pd.Timedelta(days=250), color='m', linestyle='--', label='Halving 2024 + 250 days')
        ax.axvline(x=max_close_id, color='r', linestyle='--', label='Max close')
        ax.plot(result.index, result['close'], label='Predicted')
        ax.plot(data.index, data['close'], label='Original')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        ax.xaxis.set_tick_params(rotation=45)
        ax.yaxis.set_tick_params(labelcolor='white')
        ax.xaxis.set_tick_params(labelcolor='white')
        ax.legend()
        ax.set_title('Predicted vs Original')
        ax.set_facecolor('black')
        canvas.draw()


    
    boton_predecir = tk.Button(root, text="Predecir", command=predecir)
    boton_predecir.pack(pady=20)

    resultLabel = tk.Label(root, text="El resultado de la predicción se mostrará aquí")
    resultLabel.pack(pady=20)




    fig = Figure(figsize=(15, 7))
    #fig background color
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()



    root.mainloop()


def preprocess_data(data):
    data['date'] = data['timeOpen'].str.split('T').str[0]
    data.drop(columns=['timeOpen','timeClose','timeHigh','timeLow','timestamp','name'], inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data=data.set_index('date')
    data.sort_index(inplace=True)
    return data

def predict_next_n_days(tokenForModel_scaled):
    n = time_steps + 300
    initial_data = tokenForModel_scaled.values
    for i in range(0, n, n_future):
        if i == 0:
            start = initial_data[-time_steps:]
        else:
            start = initial_data[i:i+time_steps+n_future][-time_steps:]
        if start.shape[0] != time_steps:
            continue  
        predictions = model.predict(start.reshape(1, time_steps, n_components))
        initial_data = np.append(initial_data, predictions.T)
    return initial_data

def use_model(data):
    tokenForModel=data.copy()
    tokenForModel.index = pd.to_datetime(tokenForModel.index)
    tokenForModel.sort_index(inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    pca = PCA(n_components=n_components)
    tokenForModel_scaled = scaler.fit_transform(tokenForModel)
    tokenForModel_scaled = pca.fit_transform(tokenForModel_scaled)
    tokenForModel_scaled = pd.DataFrame(tokenForModel_scaled)
    tokenForModel_scaled.index = tokenForModel.index
    tokenForModel_scaled.columns = ['pca_%d' % i for i in range(n_components)]

    predict = predict_next_n_days(tokenForModel_scaled)
    result = inverse_pca(predict,scaler,pca,tokenForModel)
    return result

def inverse_pca(predict,scaler,pca,orignal_data):
    predicted_data_rebuild = predict.reshape(-1, n_components)
    predicted_data_rebuild = pca.inverse_transform(predicted_data_rebuild)
    predicted_data_rebuild = scaler.inverse_transform(predicted_data_rebuild)
    predicted_data_rebuild = pd.DataFrame(predicted_data_rebuild)
    predicted_data_rebuild.columns = orignal_data.columns
    predicted_data_rebuild.index =pd.date_range(start=orignal_data.index[0], periods=predicted_data_rebuild.shape[0])
    return predicted_data_rebuild



if __name__ == '__main__':
    main()