import os
import os.path
import pickle
import sys

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import PIL
import PIL.Image
import PIL.ImageDraw
import cv2 as cv
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


IMG_SIZE = 50
N_FEATURES = IMG_SIZE * IMG_SIZE  # 2500


def wrap(estimator):
    return make_pipeline(StandardScaler(), estimator)


def unwrap(clf):
    if hasattr(clf, "steps"):
        return clf.steps[-1][1]
    return clf


class DrawingClassifier:
    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None

        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None

        self.clf = None
        self.proj_name = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring(
            "Project Name", "Please enter your project name down below!", parent=msg
        )

        if not self.proj_name:
            msg.destroy()
            sys.exit()
        self.proj_name = self.proj_name.strip()

        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']
            self.clf = data['clf']
            self.proj_name = data['pname']
        else:
            self.class1 = self._ask_class(msg, "class 1", "What is the first class called?")
            self.class2 = self._ask_class(msg, "class 2", "What is the second class called?")
            self.class3 = self._ask_class(msg, "class 3", "What is the third class called?")

            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1

            self.clf = wrap(LinearSVC(max_iter=5000))

            os.makedirs(os.path.join(self.proj_name, self.class1), exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class2), exist_ok=True)
            os.makedirs(os.path.join(self.proj_name, self.class3), exist_ok=True)

        msg.destroy()

    @staticmethod
    def _ask_class(parent, title, prompt):
        name = simpledialog.askstring(title, prompt, parent=parent)
        if not name or not name.strip():
            parent.destroy()
            sys.exit()
        return name.strip()

    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"AI Drawing Classifier - {self.proj_name}")

        self.canvas = Canvas(self.root, width=WIDTH - 10, height=HEIGHT - 10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W + E)

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W + E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W + E)

        brush_minus_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        brush_minus_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        brush_plus_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        brush_plus_btn.grid(row=1, column=2, sticky=W + E)

        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)


        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W + E)

        save_everything_btn = Button(btn_frame, text="Save Everything", command=self.save_everything)
        save_everything_btn.grid(row=3, column=2, sticky=W + E)

        self.status_label = Label(
            btn_frame, text=f"Current Model: {type(unwrap(self.clf)).__name__}"
        )
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = event.x, event.y
        x2, y2 = event.x + self.brush_width, event.y + self.brush_width
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.rectangle([x1, y1, x2, y2], fill="black", outline="black")

    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img = img.resize((IMG_SIZE, IMG_SIZE), PIL.Image.LANCZOS)

        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.proj_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1

        self.clear()

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def _load_class_images(self, class_name, counter, label, img_list, class_list):
        for x in range(1, counter):
            path = f"{self.proj_name}/{class_name}/{x}.png"
            img = cv.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not read training image: {path}")
            img = img[:, :, 0].reshape(N_FEATURES)
            img_list.append(img)
            class_list.append(label)

    def train_model(self):
        img_list = []
        class_list = []

        self._load_class_images(self.class1, self.class1_counter, 1, img_list, class_list)
        self._load_class_images(self.class2, self.class2_counter, 2, img_list, class_list)
        self._load_class_images(self.class3, self.class3_counter, 3, img_list, class_list)

        if len(set(class_list)) < 2:
            tkinter.messagebox.showerror(
                "AI Drawing Classifier",
                "You need saved drawings in at least two classes before training.",
                parent=self.root,
            )
            return

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.clf.fit(img_list, class_list)
        self.status_label.config(text=f"Current Model: {type(unwrap(self.clf)).__name__}")
        tkinter.messagebox.showinfo(
            "AI Drawing Classifier", "Model successfully trained!", parent=self.root
        )

    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img = img.resize((IMG_SIZE, IMG_SIZE), PIL.Image.LANCZOS)
        img.save("prediction.png", "PNG")

        img = cv.imread("prediction.png")[:, :, 0].reshape(N_FEATURES)

        try:
            prediction = self.clf.predict([img])
        except Exception:
            tkinter.messagebox.showerror(
                "AI Drawing Classifier",
                "Train or load a model before predicting.",
                parent=self.root,
            )
            return

        names = {1: self.class1, 2: self.class2, 3: self.class3}
        name = names.get(int(prediction[0]))
        if name:
            tkinter.messagebox.showinfo(
                "AI Drawing Classifier",
                f"The drawing is probably a {name}",
                parent=self.root,
            )

    def rotate_model(self):
        current = unwrap(self.clf)

        if isinstance(current, LinearSVC):
            new = KNeighborsClassifier(n_neighbors=3)
        elif isinstance(current, KNeighborsClassifier):
            new = LogisticRegression(max_iter=2000)
        elif isinstance(current, LogisticRegression):
            new = DecisionTreeClassifier()
        elif isinstance(current, DecisionTreeClassifier):
            new = RandomForestClassifier()
        elif isinstance(current, RandomForestClassifier):
            new = GaussianNB()
        else:
            new = LinearSVC(max_iter=5000)

        self.clf = wrap(new)
        self.status_label.config(
            text=f"Current Model: {type(new).__name__} (needs retraining)"
        )

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pickle")
        if not file_path:
            return
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo(
            "AI Drawing Classifier", "Model successfully saved!", parent=self.root
        )

    def load_model(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        self.status_label.config(text=f"Current Model: {type(unwrap(self.clf)).__name__}")
        tkinter.messagebox.showinfo(
            "AI Drawing Classifier", "Model successfully loaded!", parent=self.root
        )

    def save_everything(self):
        data = {
            "c1": self.class1, "c2": self.class2, "c3": self.class3,
            "c1c": self.class1_counter, "c2c": self.class2_counter,
            "c3c": self.class3_counter, "clf": self.clf, "pname": self.proj_name,
        }
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo(
            "AI Drawing Classifier", "Project successfully saved!", parent=self.root
        )

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel(
            "Quit?", "Do you want to save your work?", parent=self.root
        )
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            sys.exit()


if __name__ == "__main__":
    DrawingClassifier()