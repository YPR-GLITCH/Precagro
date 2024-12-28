import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Function to predict plant type
def predict_plant_type(model, values, scaler):
    new_data = pd.DataFrame(values, index=[0])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction[0]

# Load the trained model, scaler, and label encoder
loaded_model = load_model('crop_recommendation_model.keras')

# Recompile the loaded model to ensure metrics are set up correctly
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class PrecisionAgricultureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision Agriculture App")
        self.root.geometry("1000x600")
        self.root.configure(background='#aee6aa')
        
        # Create a menu bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Create Tabs
        self.tab_control = ttk.Notebook(self.root)
        
        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.npk_calculations_tab = ttk.Frame(self.tab_control)
        self.blogs_tab = ttk.Frame(self.tab_control)
        self.settings_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.dashboard_tab, text="Dashboard")
        self.tab_control.add(self.npk_calculations_tab, text="NPK Calculations")
        self.tab_control.add(self.blogs_tab, text="Blogs")
        self.tab_control.add(self.settings_tab, text="Settings")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Setup Dashboard Tab
        self.setup_dashboard_tab()
        
        # Setup NPK Calculations Tab
        self.setup_npk_calculations_tab()
        
        # Setup Blogs Tab
        self.setup_blogs_tab()
        
        # Setup Settings Tab
        self.setup_settings_tab()
        
    def show_about(self):
        messagebox.showinfo("About", "Precision Agriculture App v1.0\nDeveloped by OpenAI")
        
    def setup_dashboard_tab(self):
        header_frame = ttk.Frame(self.dashboard_tab, style='Dashboard.TFrame')
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="Precision Agriculture App", font=("Helvetica", 16)).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(header_frame, text="Profile").pack(side=tk.RIGHT, padx=10, pady=10)
        ttk.Button(header_frame, text="Settings").pack(side=tk.RIGHT, padx=10, pady=10)
        
        dashboard_frame = ttk.Frame(self.dashboard_tab, style='Dashboard.TFrame')
        dashboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # Simulated NPK values (replace with actual hardware data)
        npk_values = {
            'N': '20',
            'P': '15',
            'K': '10'
        }
        
        npk_frame = ttk.LabelFrame(dashboard_frame, text="NPK Values", style='Dashboard.TLabelframe')
        npk_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ttk.Label(npk_frame, text=f"N: {npk_values['N']}").pack(pady=5)
        ttk.Label(npk_frame, text=f"P: {npk_values['P']}").pack(pady=5)
        ttk.Label(npk_frame, text=f"K: {npk_values['K']}").pack(pady=5)
        
        # User inputs for pH and weather
        ph_frame = ttk.LabelFrame(dashboard_frame, text="pH Levels", style='Dashboard.TLabelframe')
        ph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.ph_entry = ttk.Entry(ph_frame, width=10)
        self.ph_entry.pack(pady=5)
        
        weather_frame = ttk.LabelFrame(dashboard_frame, text="Weather", style='Dashboard.TLabelframe')
        weather_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(weather_frame, text="Humidity:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.humidity_entry = ttk.Entry(weather_frame, width=20)
        self.humidity_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(weather_frame, text="Temperature:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.temperature_entry = ttk.Entry(weather_frame, width=20)
        self.temperature_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(weather_frame, text="Rainfall:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.rainfall_entry = ttk.Entry(weather_frame, width=20)
        self.rainfall_entry.grid(row=2, column=1, padx=5, pady=5)
        
        dashboard_frame.rowconfigure(0, weight=1)
        dashboard_frame.rowconfigure(1, weight=1)
        dashboard_frame.columnconfigure(0, weight=1)
        dashboard_frame.columnconfigure(1, weight=1)
        
    def setup_npk_calculations_tab(self):
        npk_calculations_frame = ttk.Frame(self.npk_calculations_tab, style='Dashboard.TFrame')
        npk_calculations_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(npk_calculations_frame, text="NPK Calculations Section").pack(pady=20)
        
        npk_values_frame = ttk.LabelFrame(npk_calculations_frame, text="Enter NPK Values", style='Dashboard.TLabelframe')
        npk_values_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        ttk.Label(npk_values_frame, text="N Value:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.n_entry = ttk.Entry(npk_values_frame, width=10)
        self.n_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(npk_values_frame, text="P Value:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.p_entry = ttk.Entry(npk_values_frame, width=10)
        self.p_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(npk_values_frame, text="K Value:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.k_entry = ttk.Entry(npk_values_frame, width=10)
        self.k_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(npk_values_frame, text="Temperature:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.temperature_entry_npk = ttk.Entry(npk_values_frame, width=10)
        self.temperature_entry_npk.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(npk_values_frame, text="Humidity:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.humidity_entry_npk = ttk.Entry(npk_values_frame, width=10)
        self.humidity_entry_npk.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(npk_values_frame, text="pH:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.ph_entry_npk = ttk.Entry(npk_values_frame, width=10)
        self.ph_entry_npk.grid(row=5, column=1, padx=5, pady=5)
        
        ttk.Label(npk_values_frame, text="Rainfall:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.rainfall_entry_npk = ttk.Entry(npk_values_frame, width=10)
        self.rainfall_entry_npk.grid(row=6, column=1, padx=5, pady=5)
        
        self.predict_button = ttk.Button(npk_values_frame, text="Predict Plant Type", command=self.predict_plant_type_gui)
        self.predict_button.grid(row=7, column=0, columnspan=2, pady=10)
        
        self.result_label = ttk.Label(npk_values_frame, text="")
        self.result_label.grid(row=8, column=0, columnspan=2, pady=10)

    def setup_blogs_tab(self):
        blogs_frame = ttk.Frame(self.blogs_tab, style='Dashboard.TFrame')
        blogs_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(blogs_frame, text="Crops and Soil Health Requirements").pack(pady=20)
        
        crops = [
            ("Wheat", "pH 6.0-7.0, N 20-30 kg/ha, P 20-30 kg/ha, K 10-20 kg/ha"),
            ("Rice", "pH 5.5-6.5, N 80-120 kg/ha, P 40-60 kg/ha, K 40-60 kg/ha"),
            ("Maize", "pH 5.5-7.0, N 100-150 kg/ha, P 60-90 kg/ha, K 40-60 kg/ha"),
            ("Soybean", "pH 6.0-7.0, N 20-30 kg/ha, P 50-80 kg/ha, K 40-80 kg/ha"),
            ("Barley", "pH 6.0-7.0, N 20-30 kg/ha, P 20-30 kg/ha, K 20-30 kg/ha"),
            ("Sorghum", "pH 5.8-6.5, N 80-120 kg/ha, P 40-60 kg/ha, K 40-60 kg/ha"),
            ("Cotton", "pH 5.8-6.5, N 80-100 kg/ha, P 40-60 kg/ha, K 40-60 kg/ha"),
            ("Sugarcane", "pH 6.0-7.5, N 100-150 kg/ha, P 60-80 kg/ha, K 120-150 kg/ha"),
            ("Potato", "pH 5.0-5.5, N 100-150 kg/ha, P 60-90 kg/ha, K 150-200 kg/ha"),
            ("Tomato", "pH 6.0-6.8, N 50-80 kg/ha, P 40-60 kg/ha, K 80-120 kg/ha")
        ]
        
        for crop, requirements in crops:
            ttk.Label(blogs_frame, text=f"{crop}: {requirements}").pack(pady=5)
        
    def setup_settings_tab(self):
        settings_frame = ttk.Frame(self.settings_tab, style='Dashboard.TFrame')
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(settings_frame, text="Settings Section").pack(pady=20)
        
    def predict_plant_type_gui(self):
        try:
            input_values = {
                'N': float(self.n_entry.get()),
                'P': float(self.p_entry.get()),
                'K': float(self.k_entry.get()),
                'temperature': float(self.temperature_entry_npk.get()),
                'humidity': float(self.humidity_entry_npk.get()),
                'ph': float(self.ph_entry_npk.get()),
                'rainfall': float(self.rainfall_entry_npk.get())
            }
            print("Input values for prediction:", input_values)
            predicted_probabilities = predict_plant_type(loaded_model, input_values, scaler)
            predicted_label_encoded = np.argmax(predicted_probabilities)
            predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
            self.result_label.config(text=f"Predicted Plant Type: {predicted_label}")
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PrecisionAgricultureApp(root)
    root.mainloop()
