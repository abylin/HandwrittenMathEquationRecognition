from tkinter import *
from tkinter import _flatten
from tensorflow.keras.models import load_model
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image, ImageOps
from sklearn import preprocessing
import numpy as np
import time
import threading
import cv2
import calcuate_expression
from sliding import formula_recognizer

model = load_model('mix2.h5') 

def stringToSymbol(pred): 
    if pred == "zero":
        return '0'    
    elif pred == "one":
        return '1'
    elif pred == "two":
        return '2'
    elif pred == "three":
        return '3'
    elif pred == "four":
        return '4'
    elif pred == "five":
        return '5'
    elif pred == "six":
        return '6'
    elif pred == "seven":
        return '7'
    elif pred == "eight":
        return '8'
    elif pred == "nine":
        return '9'
    elif pred == "plus":
        return '+'
    elif pred == "minus":
        return '-'
    elif pred == "div":
        return '/'
    elif pred == "equal":
        return '='
    elif pred == "decimal":
        return '.'
    elif pred == "times":
        return '*'
    elif pred == "left":
        return '('
    elif pred == "right":
        return ')'

def predict_digit(img):
    img = img.resize((100, 100))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(100, 100, 1)       
    img = np.expand_dims(img, axis=0)     
    pred = model.predict(img)  
    result = np.argsort(pred)   
    result = result[0][::-1] 
    final_label_array = label_encoder.inverse_transform(np.array(result))
    final_label = "".join(final_label_array[0])
    return stringToSymbol(final_label)

img_path = 'temp.jpg'

def findPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # remove some noise
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints

# Find the edges and return to the top left and bottom right corners of the border
def findBorder(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Edge expansion
    padding = np.ones((50, 50), np.uint8) 
    img = cv2.dilate(img, padding, iterations=1)
    _, img = cv2.threshold(img, 128, 0, cv2.THRESH_TOZERO)

    # row
    hori_vals = np.sum(img, axis=1)
    hori_points = findPeek(hori_vals)
    # column
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = findPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders

def show_split_results(origin_image, borders):
    clone = np.copy(origin_image)
    clone = np.expand_dims(clone, axis=2)
    for border in borders:
        x1, y1, x2, y2, label, score = border
        cv2.rectangle(clone, (x1, y1), (x2, y2), (150, 150, 150), thickness=1)
    cv2.imshow('The image with borders', clone)

def get_express_string(symbols, expression_idx):
    expression = ""
    final_ret = []
    i, exp_idx = 0, 0
    for symbol in symbols:
        label = symbol[4]
        end_idx = expression_idx[exp_idx]
        expression += stringToSymbol(label)
        if end_idx == i:
            # current expression ends
            exp_idx += 1
            if expression[-1] == '=':
                expression = expression[0:-1]
            ret_val = calcuate_expression.remove_parentheses(expression)
            ret = expression + "=" + ret_val
            final_ret.append(ret)
            expression = ""
        i += 1
    final_ret = "\n".join(final_ret)
    return final_ret

class CalThread(threading.Thread):
    terminated = False

class App(tk.Tk):
    calculate_thread = None

    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=1350, height=310, bg="white", cursor="cross")
        self.label = tk.Label(self, width=50, text="Waiting...", font=("Helvetica", 26))
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.button_Recognize = tk.Button(self, text="Recognize", command=self.recognize_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.button_clear.grid(row=0, column=1, pady=2)
        # self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.label.grid(row=1, column=0, pady=2, sticky=W)
        self.button_Recognize.grid(row=1, column=1, pady=2)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.make_widgets()
        self.recognizer = formula_recognizer.Recognizer(".//mix2.h5")

    def make_widgets(self):
        self.winfo_toplevel().title("hand written mathematical equation recognition")

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text='Cleared!')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 6
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')       

    def recognize_all(self):
        hwind = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(hwind)  # get the coordinate of the canvas
        x1, y1, x2, y2 = rect
        clip_boarder = (x1+2, y1+2, x2-2, y2-2)
        img = ImageGrab.grab(clip_boarder)
        np_image = self.recognizer.pil_image_to_numpy(img)
        symbols, expression_idx = self.recognizer.recognize(np_image)
        show_split_results(np_image, symbols)  # It can be removed when published
        result = get_express_string(symbols, expression_idx)
        app.label.configure(text=str(result))

app = App()
mainloop()
