import tkinter as tk
import tensorflow as tf
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modi_hist_extract import modi_hist_extract
import numpy as np
from dataprocess import * 
import sys

sys.path.append("./")


class myApp:
    def __init__(self, root):
        row = [0, 6, 10]
        col = [0, 3, 6, 11, 15, 20, 25]
        
        # 창 제어
        self.root = root
        self.root.title("Quick View")
        
        
        # Main 글씨
        self.label = tk.Label(root, text="1.Signal CSV 선택 (optional - cut 할 시간 선택) \n 2. model 선택 \n 3.predict")
        self.label.grid(row=row[0], column=col[0], padx=10, pady=10)
        
        
        # 1. csv 선택 버튼
        self.csv_button = tk.Button(root, text="CSV 파일", command=self.csv_open_file)
        self.csv_button.grid(row=row[1], column=col[0], padx=5, pady=5)
        self.csv_label = tk.Label(root, text="Signal csv file name :  ")
        self.csv_label.grid(row=row[1]+1, column=col[0], padx=5, pady=5)
        
        self.csvshow_button = tk.Button(root, text="Check and Apply", command=self.show_graph, state="disabled")
        self.csvshow_button.grid(row=row[1]+2, column=col[0], padx=5, pady=5)
        
        
        
        #. start time 설정 버튼
        self.starttime = 0
        
        self.starttime_entry = tk.Entry(root, width=3)
        self.starttime_entry.grid(row=row[1], column=col[1], padx=10, pady=10)
        self.starttime_label = tk.Label(root, text="start time 설정된 값: ")
        self.starttime_label.grid(row=row[1]+2, column=col[1], padx=10, pady=10)
        self.starttime_button = tk.Button(root, text="start time 값 설정", command=self.starttime_set_value)  # command 에 빈 괄호 넣지 않도록 주의!
        self.starttime_button.grid(row=row[1]+1, column=col[1], padx=10, pady=10)

        # end time 설정 버튼
        self.endtime = 999
        
        self.endtime_entry = tk.Entry(root, width=3)
        self.endtime_entry.grid(row=row[1], column=col[2], padx=10, pady=10)
        self.endtime_label = tk.Label(root, text="end time 설정된 값: ")
        self.endtime_label.grid(row=row[1]+2, column=col[2], padx=10, pady=10)
        self.endtime_button = tk.Button(root, text="end time 값 설정", command=self.endtime_set_value)  # command 에 빈 괄호 넣지 않도록 주의!
        self.endtime_button.grid(row=row[1]+1, column=col[2], padx=10, pady=10)
        
        
        # 2. model 선택 버튼
        self.model_button = tk.Button(root, text="model 파일", command=self.model_file)
        self.model_button.grid(row=row[1], column=col[3], padx=5, pady=5)
        self.model_label = tk.Label(root, text="model file name :  ")
        self.model_label.grid(row=row[1]+1, column=col[3], padx=5, pady=5)
        
        
        # 2-2. model input version
        # 1 : 1,1000,2  | 2 : 1,1000,3
        self.input_ver = 1 # initial
        self.selected_option = tk.StringVar()
        self.radio1 = tk.Radiobutton(root, text="input version 1", variable=self.selected_option, value=1, command=self.show_selected)
        self.radio1.grid(row=row[0], column=col[3], padx=5, pady=5)
        self.radio2 = tk.Radiobutton(root, text="input version 2", variable=self.selected_option, value=2, command=self.show_selected)
        self.radio2.grid(row=row[0]+1, column=col[3], padx=5, pady=5)
        self.input_ver_label = tk.Label(root, text="input version : ")
        self.input_ver_label.grid(row=row[2], column=col[3], padx=5, pady=5)
        
        # 2-3 background file (for poisson)
        self.back_button = tk.Button(root, text="background 파일", command=self.back_file)
        self.back_button.grid(row=row[1], column=col[4], padx=5, pady=5)
        self.back_label = tk.Label(root, text="background file name :  ")
        self.back_label.grid(row=row[1]+1, column=col[4], padx=5, pady=5)
        
        
        # 2-4. check 중간 result
        
        # 3. predict 버튼
        self.predict_button = tk.Button(root, text="예측 실행", command=self.model_predict, state="disabled")
        self.predict_button.grid(row=row[1], column=col[6], padx=5, pady=5)
        self.predict_label = tk.Label(root, text="predict result : ")
        self.predict_label.grid(row=row[1]+1, column=col[6], padx=5, pady=5)
        self.predict_info_label = tk.Label(root, text="Ba133, Cs137, Na22, Bacgkround, ...")
        self.predict_info_label.grid(row=row[1]+2, column=col[6], padx=5, pady=5)
        

        
        
        # 배경 설정인듯?
        self.canvas = None
    
    

    # start time
    def starttime_set_value(self):
        entered_value = self.starttime_entry.get()
        if entered_value == '':
            entered_value = 0
        self.starttime_label.config(text=f"설정된 값: {entered_value}")
        self.starttime = float(entered_value)
        
        
    # end time
    def endtime_set_value(self):
        entered_value = self.endtime_entry.get()
        if entered_value == '':
            entered_value = 999
        self.endtime_label.config(text=f"설정된 값: {entered_value}")
        self.endtime = float(entered_value)


    
    # 1. csv_open_file
    def csv_open_file(self):
        csvfile_path = filedialog.askopenfilename(filetypes=[("CSV 파일", "*.csv")])
        if csvfile_path:
            self.create_histogram(csvfile_path)
            self.csv_label.config(text=f"{csvfile_path}")
            self.csvshow_button.config(state="normal")
    # 1-2. make histogram        
    def create_histogram(self, csvfile_path):
        # CSV 파일 로드
        print(csvfile_path)
        df = modi_hist_extract(csvfile_path)
        self.df = df
        
            
    # 1-3. show histogram and apply time filter
    def show_graph(self):
        self.df.filtered(self.starttime, self.endtime)
        
        # print(self.df.filtered_hist.shape)   (1000,)
        
        # 새 윈도우에 히스토그램 출력
        new_window = tk.Toplevel(self.root)
        new_window.title("히스토그램")
        
        # 히스토그램 생성
        fig, ax = plt.subplots()
        ax.plot(self.df.filtered_hist)
        ax.set_title("Current Signal")
        ax.set_xlabel("Energy bin")
        ax.set_ylabel("Counts")
        
        # Tkinter에 matplotlib Figure 추가
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.canvas = FigureCanvasTkAgg(fig, master=new_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()            
        
    
    
    # 3-1 find model file
    def model_file(self):
        model_path = filedialog.askopenfilename(filetypes=[("model 파일", "*.h5")])
        if model_path:
            self.predict_button.config(state="normal")
            self.model_label.config(text=f"{model_path}")
            self.modelpath = model_path
            self.model = tf.keras.models.load_model(self.modelpath)
    
    # 3-2. select input version
    def show_selected(self):
        selected_value = self.selected_option.get()
        self.input_ver_label.config(text=f"선택된 옵션: {selected_value}")
        self.input_ver = int(selected_value)
        
        
        
    # 3-3. background file
    def back_file(self):
        back_path = filedialog.askopenfilename(filetypes=[("back 파일", "*.csv")])
        if back_path:
            self.back_label.config(text=f"{back_path}")
            self.backpath = back_path
            ut_dt = modi_hist_extract(back_path)
            self.back_ut = ut_dt.hist / sum(abs(ut_dt.hist))
            
            
    # 4. predict
    def model_predict(self):
        mixed_spectrum = self.df.filtered_hist / (np.max(self.df.filtered_hist))
        mixed_spectrum = mixed_spectrum[np.newaxis, :]
        filtered1 = noisefiltering(mixed_spectrum, 0, 0)
        derivatives1 = derivative_signal(filtered1)
        filtered2 = noisefiltering2(derivatives1, 0, 0)   
        
        if self.input_ver == 1:
            temp = np.zeros((1, 1, 1000, 2))
            temp[0, :, :, 0] = mixed_spectrum
            temp[0, :, :, 1] = filtered2
        
        
        else:
            temp = np.zeros((1, 1, 1000, 3))
            pos_dev = my_pos_dev(mixed_spectrum[0,:], self.back_ut)
            
            temp[0, :, :, 0] = mixed_spectrum
            temp[0, :, :, 1] = filtered2
            temp[0, :, :, 2] = pos_dev
        
        pred = self.model.predict(temp)
        self.predict_label.config(text=f"{pred}")
    
    

        
        

                
        

# Tkinter 애플리케이션 실행
root = tk.Tk()
app = myApp(root)
root.mainloop()