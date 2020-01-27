import tkinter as tk
from tkinter import filedialog
import os

def run_jump():
	file_path = filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))

	# Check filepath exists (user did not click cancel)
	check = os.path.exists(file_path)

	global label2_flag

	if check == False:
		label2 = tk.Label(root, text="Please choose a file")
		label2.grid(column=0, row=2)
		label2_flag = 1

	else: # File has been selected, run code and display
		button1.destroy()
		if label2_flag == 1:
			label2.destroy()
		

root = tk.Tk()
root.title("Maximum Jump Height")

root.geometry("400x400")
#root.resizable(False, False)

label1 = tk.Label(root, text="Maximum Jump Height", font=18)
label1.grid(column=0, row=0)

label2_flag = 0
button1 = tk.Button(root, command=run_jump, text="Select jump file")
button1.grid(column=0,row=1)

root.mainloop()
