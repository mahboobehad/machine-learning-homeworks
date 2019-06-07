# In the name of God

import tkinter as tk

import numpy as np

from kmeans import KMeans


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.cluster_input = None
        self.cluster = None
        self.create_widgets()

    def create_widgets(self):
        L1 = tk.Label(text="Number of clusters")
        L1.pack(side=tk.LEFT)
        E1 = tk.Entry()
        E1.pack(side=tk.RIGHT)
        self.cluster_input = E1

        self.cluster = tk.Button(self)
        self.cluster["text"] = "cluster"
        self.cluster["command"] = self.process_data
        self.cluster.pack(side="bottom")

    def process_data(self):
        print("processing data")
        number_of_clusters = int(self.cluster_input.get())
        data = list()

        for _ in range(50):
            p_x = np.random.random() * 100
            p_y = np.random.random() * 100
            data.append(np.array([p_x, p_y]))
        kmeans = KMeans(number_of_clusters=number_of_clusters, max_iterations=100)
        kmeans.cluster(data)
        kmeans.plot_clusters()


root = tk.Tk()
app = Application(master=root)
app.mainloop()
