import tkinter as tk
from tkinter import filedialog
import os
import csv
import numpy as np
from scipy import signal
from scipy import integrate
import matplotlib
import sys

def restart():
	label1.destroy()
	button1.destroy()

	button4.destroy()

	canvas.destroy()

	for label in label3:
		label.destroy()
	
	for label in label5:
		label.destroy()

	initial_screen()

	return

def test_input(user_input):
	n_jumps = user_input.get()

	try:
		global n_jumps_est
		n_jumps_est = int(n_jumps)

		label4.destroy()
		button5.destroy()
		entry1.destroy()

		calculate_times()

	except ValueError:
		label4.destroy()
		button5.destroy()
		entry1.destroy()

		enter_jumps()

	return

def enter_jumps():
	label2.destroy()
	button2.destroy()
	button3.destroy()

	global label4
	label4 = tk.Label(root, text='How many times did the person jump? (numerical value)', bg='grey20', fg='OrangeRed2', font=('Arial',20))
	label4.place(relx=0.5, rely=0.35, anchor="center")

	global entry1
	entry1 = tk.Entry(root, bg='OrangeRed2', fg='grey20', font=('Arial',20), borderwidth=0)
	entry1.place(relx=0.5, rely=0.4, anchor="center")

	global button5
	button5 = tk.Button(root, text='Select', command=(lambda e = entry1: test_input(e)), font=('Arial', 20), bg='OrangeRed2', fg='grey20', activebackground='grey20',
						activeforeground='OrangeRed2', borderwidth=0)
	button5.place(relx=0.5, rely=0.5, anchor="center")

	return

def calculate_times():
	label2.destroy()
	button2.destroy()
	button3.destroy()

	n_jumps = n_jumps_est
	n_peaks = 2 * n_jumps
	# First n_peaks are the important peaks
	minima_jumps = minima_acc[:n_peaks]

	ind = []
	for i in range(n_peaks):
		ind.append(np.where(filt_acc == minima_jumps[i])[0][0])

	# Two peaks per jump, so get the first peak of each jump.
	ind.sort()

	# First peak is every even indice (0, 2, 4... etc)
	first_peaks = ind[::2]
	second_peaks = ind[1::2]

	global label3
	label3 = []

	# Use flight time
	take_off_list = []
	touch_down_list = []

	for i in range(n_jumps):
		''' Get take off and landing time '''

		# Take-off occurs when acceleration crosses 0 line after the first peak (first positive value)
		acc_temp = filt_acc[first_peaks[i]:]
		first_pos = (np.where(acc_temp > 0)[0])[0]

		# Evaluate the value of this to see if the next number is closer to 0
		if abs(acc_temp[first_pos]) < abs(acc_temp[first_pos - 1]):
			take_off_ind = first_pos + first_peaks[i]
		else:
			take_off_ind = first_pos - 1 + first_peaks[i]
		
		# Touch-down occurs when acceleration crosses 0 line before the last peak (last positive value)
		acc_temp = filt_acc[:second_peaks[i]].tolist()
		first_pos = next(j for j in reversed(range(len(acc_temp))) if acc_temp[j] > 0)

		# Evaluate the value of this to see if the next number is closer to 0
		if abs(acc_temp[first_pos]) < abs(acc_temp[first_pos + 1]):
			touch_down_ind = first_pos
		else:
			touch_down_ind = first_pos + 1

		flight_time = time[touch_down_ind] - time[take_off_ind]

		take_off_list.append(take_off_ind)
		touch_down_list.append(touch_down_ind)

		g = 9.81
		max_height = (g * flight_time * flight_time) / 8

		label3.append(tk.Label(root, text=('Jump Height = {:.2f} cm'.format(max_height * 100)), bg='grey20', fg='OrangeRed2', font=('Arial',20)))
		label3[-1].place(relx=0.65, rely=0.35 + (i*0.05), anchor="center")

	# Use integration
	# Re-zeroed acceleration data/and filtered
	a_filt = -(filt_acc.copy())

	# Filter data to remove drift
	frequency = 500
	cut_off = 0.5 # Play around with this.
	d, c = signal.butter(N=4, Wn=cut_off/(frequency/2), btype='high')

	# Integrate to get velocity
	v = integrate.cumtrapz(y=a_filt, x=time, initial=0)
	v_filt = signal.filtfilt(b=d, a=c, x=v)

	# Integrate to get displacement
	s = integrate.cumtrapz(y=v_filt, x=time, initial=0)

	s_filt = signal.filtfilt(b=d, a=c, x=s)

	jump_height = []

	global label5
	label5 = []

	# Get maximum height. Since centre of mass is zeroed, the maximum height will be the jump height.
	for i in range(n_jumps):
		jump_height.append(max(s_filt[take_off_list[i]:touch_down_list[i]]))

		label5.append(tk.Label(root, text=('Jump Height = {:.2f} cm'.format(jump_height[i] * 100)), bg='grey20', fg='OrangeRed2', font=('Arial',20)))
		label5[-1].place(relx=0.35, rely=0.35 + (i*0.05), anchor="center")

	global button4
	button4 = tk.Button(root, text='Restart', command=restart, font=('Arial', 20), bg='OrangeRed2', fg='grey20', activebackground='grey20',
						activeforeground='OrangeRed2', borderwidth=0)

	# Set y placement depending on how many jumps were recorded (to avoid overlap)
	if i >= 2:
		y = 0.35 + (i*0.05) + 0.1

	else:
		y = 0.5

	button4.place(relx=0.5, rely=y, anchor="center")

	# Draw line between both outputs (time of flight and integration)

	global canvas
	# 0.6c is roughly the height of 1 label.
	canvas = tk.Canvas(root, width='2c', height= str(0.6*(2*n_jumps-1)) + 'c', bg='grey20', bd=0, highlightthickness=0)

	if int(n_jumps / 2) == n_jumps / 2: # Even number of jumps
		middle = 0.35 + int(n_jumps / 2) * 0.05 - 0.05/2
	else:
		middle = 0.35 + int(n_jumps / 2) * 0.05

	canvas.place(relx=0.5, rely=middle, anchor="center")

	canvas.create_line('1c', 0, '1c', 10000, fill='OrangeRed2', dash=50, width=4)

	return

def estimate_jumps():
	file_path = filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))

	# Check filepath exists (user did not click cancel)
	check = os.path.exists(file_path)

	if check == False:
		pass

	else: # File has been selected, run code and display
		button1.destroy()

		acc = []
		with open(file_path, newline='') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				acc.append(row)
		
		acc = np.array(acc)

		# Take the y component (vertical direction) and time (convert to float), assume movement only in the y direction
		y_acc = acc[1:,2].astype(np.float)

		global time
		time = acc[1:,0].astype(np.float)

		y_acc = y_acc[500:]
		time = time[500:]

		# Filter data
		frequency = 500 # Arbitrary for now
		cut_off = 5
		b, a = signal.butter(4, cut_off/(frequency/2), 'low')
		global filt_acc
		filt_acc = signal.filtfilt(b, a, y_acc)

		# Rotate data and remove effect of gravity

		# Subject will be still for first part of trial, first 250 data points should do (0.5 s)
		g = np.mean(filt_acc[:251]) # This value should be near 9.81 (assuming movement only in y direction)

		# Depending on calibration of sensor/rotation, positive acceleration may be positive or negative
		if g > 0:
			filt_acc = -(filt_acc - g) # Remove effect of gravity and flip
		else:
			filt_acc = filt_acc - g
		
		# Get minima points, for each jump there should be 2 siginficant peaks per jump
		minima_ind = np.where(np.r_[True, filt_acc[1:] < filt_acc[:-1]] & np.r_[filt_acc[:-1] < filt_acc[1:], True] == True)
		
		global minima_acc
		minima_acc = filt_acc[minima_ind].tolist()

		# Sort in decending order (largest first)
		minima_acc.sort()

		temp = minima_acc.copy()

		min_peak = min(temp)

		sig_mins = []
		sig_mins.append(min_peak)
		temp.remove(min_peak)

		for i in range(len(temp)):
			delta = min_peak / temp[i]
			if delta <= 4 and temp[i] < 0:
				sig_mins.append(temp[i])

		# Each min point should be separated by a positive value (a jump or a rise at the end of a jump)
		final_min_points_ind = []

		# Get indices of sig_mins
		sig_mins_ind = []
		for i in range(len(sig_mins)):
			sig_mins_ind.append(np.where(filt_acc == sig_mins[i])[0][0])

		sig_mins_ind.sort()

		tol = 0.2

		for i in range(len(sig_mins)):
			if i == len(sig_mins) - 2:
				if np.size(np.where(filt_acc[sig_mins_ind[i]:] > -tol)) > 0:
					final_min_points_ind.append(sig_mins_ind[i])

			elif i != len(sig_mins) - 1: # Not on the last minimum
				# Test if there is a positive number between the two mins
				if np.size(np.where(filt_acc[sig_mins_ind[i]:sig_mins_ind[i+1] + 1] > -tol)) > 0:
					final_min_points_ind.append(sig_mins_ind[i])
			else:
				if np.size(np.where(filt_acc[sig_mins_ind[i-1]:sig_mins_ind[i] + 1] > -tol)) > 0:
					final_min_points_ind.append(sig_mins_ind[i])
		global n_jumps_est
		n_jumps_est = int(np.floor(len(final_min_points_ind) / 2))

		# Check if number of jumps estimate is correct with user
		global label2
		if n_jumps_est == 1:
			label2 = tk.Label(root, text=('Did the person jump ' + str(n_jumps_est) + ' time?'), bg='grey20', fg='OrangeRed2', font=('Arial',20))
		else:
			label2 = tk.Label(root, text=('Did the person jump ' + str(n_jumps_est) + ' times?'), bg='grey20', fg='OrangeRed2', font=('Arial',20))
		label2.place(relx=0.5, rely=0.35, anchor="center")

		global button2
		global button3

		button2 = tk.Button(root, text='Yes', command= calculate_times, font=('Arial', 20), bg='OrangeRed2', fg='grey20', activebackground='grey20',
							activeforeground='OrangeRed2', borderwidth=0, width=5)
		button2.place(relx=0.46, rely=0.5, anchor="center")

		button3 = tk.Button(root, text='No', command=enter_jumps, font=('Arial', 20), bg='OrangeRed2', fg='grey20', activebackground='grey20',
							activeforeground='OrangeRed2', borderwidth=0, width=5)
		button3.place(relx=0.54, rely=0.5, anchor="center")

	return

def initial_screen ():
	global label1
	global button1

	label1 = tk.Label(root, text="Maximum Jump Height", font=('Arial',50), bg='grey20', fg='OrangeRed2', anchor="center")
	label1.place(relx=0.5, rely=0.2, anchor="center")

	button1 = tk.Button(root, command=estimate_jumps, font=('Arial', 20), text="Select jump file", bg='OrangeRed2', fg='grey20', 
						activebackground='grey20', activeforeground='OrangeRed2', borderwidth=0)
	button1.place(relx=0.5, rely=0.5, anchor="center")

	return

if __name__ == "__main__":
	root = tk.Tk()
	root.title("Maximum Jump Height")

	# Set fullscreen
	root.attributes('-fullscreen',True)
	root.bind("<F11>", lambda event: root.attributes('-fullscreen', not root.attributes('-fullscreen'))) # F11 toogles
	root.bind("<Escape>", lambda event: root.attributes('-fullscreen',False)) # Esc leaves fullscreen

	# Set background colour
	root.configure(bg='grey20')

	initial_screen()

	root.mainloop()
