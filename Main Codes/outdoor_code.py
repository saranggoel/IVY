import serial
import googlemaps
from datetime import datetime
from bs4 import BeautifulSoup
import subprocess
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play
from threading import Thread,Lock
import pdb
import board
import adafruit_icm20x
from math import atan2, pi, sqrt, sin, cos
import signal
import csv
import numpy as np
import RPi.GPIO as GPIO

# Board pin-numbering scheme

# Open serial port
ser = serial.Serial("/dev/ttyTHS1", 9600, timeout=1)

def vib_thread():
	global rose
	
	# Dictionary corresponding angle to each direction
	values = {
	"north":0,
	"northeast":45,
	"east":90,
	"southeast":135,
	"south":180,
	"southwest":225,
	"west":270,
	"northwest":315
	}
	
	GPIO.cleanup() # Clears up any pre-assigned settings
	
	GPIO.setmode(GPIO.BOARD)
    
	GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(15, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(19, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(21, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(23, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(29, GPIO.OUT, initial=GPIO.LOW)
	# Set pin as an output pin with optional initial state of LOW
	pins = [13, 15, 19, 21, 23, 29]
	while True:
		if rose != "none":
			
			degree = values[rose]
			while True: # Set vibration motor signal based on degree
				if yaw_deg > degree:
					angle_turn = yaw_deg - degree
					if (angle_turn >= 45):
						output_pin = 29						
					elif (20 <= angle_turn < 45):
						output_pin = 29	
					elif (10 <= angle_turn < 20):
						output_pin = 23
					else:
						output_pin = 21
					 
				else:
					angle_turn = degree - yaw_deg
					if (angle_turn >= 45):
						output_pin = 13
					elif (20 <= angle_turn < 45):
						output_pin = 15
					elif (10 <= angle_turn < 20):
						output_pin = 19
					else:
						output_pin = 21
				
				for pin in pins:
					if output_pin == pin:
						GPIO.output(pin, GPIO.HIGH)
					else:
						GPIO.output(pin, GPIO.LOW)
			
	

def imu_thread():
	global yaw_deg
	yaw_deg = 0
	i2c = board.I2C()  # Uses board.SCL and board.SDA
	icm = adafruit_icm20x.ICM20948(i2c) # Initializes IMU Sensor

	mpu_labels = ['a_x','a_y','a_z','w_x','w_y','w_z','m_x','m_y','m_z']
	cal_labels = [['a_x','m','b'],['a_y','m','b'],['a_z','m','b'],'w_x','w_y','w_z',
		          ['m_x','m_x0'],['m_y','m_y0'],['m_z','m_z0']]
	mag_cal_axes = ['z','y','x'] # axis order being rotated for mag cal
	cal_filename = 'mpu9250_cal_params.csv' # filename for saving calib coeffs
	cal_size = 200 # how many points to use for calibration averages
	cal_offsets = np.array([[],[],[],0.0,0.0,0.0,[],[],[]]) # cal vector
	
	with open(cal_filename,'r',newline='') as csvfile: # Open calibration offset values from CSV file and loads them
		reader = csv.reader(csvfile,delimiter=',')
		iter_ii = 0
		for row in reader:
			if len(row)>2:
				row_vals = [float(ii) for ii in row[int((len(row)/2)+1):]]
				cal_offsets[iter_ii] = row_vals
			else:
				cal_offsets[iter_ii] = float(row[1])
			iter_ii+=1
	
	cal_rot_indicies = [[6,7],[7,8],[6,8]] # heading indices
	plt_pts = 100 # points to plot
	ii_iter = 0 # plot update iteration counter 
	mpu_array = np.zeros((plt_pts,9)) # pre-allocate the 9-DoF vector
	mpu_array[mpu_array==0] = np.nan
	while True:
		try:
			ax,ay,az = icm.acceleration
			wx,wy,wz = icm.gyro # read and convert mpu6050 data
			mx,my,mz = icm.magnetic # read and convert AK8963 magnetometer data
		except:
			continue

		mpu_array[0:-1] = mpu_array[1:] # get rid of last point
		mpu_array[-1] = [ax,ay,az,wx,wy,wz,mx,my,mz] # update last point w/new data
		angle=[]
		if ii_iter==50:
			for ii in range(0,9):
				if ii in range(6,9):
					jj = ii-6 # index offsetted to 0-2
					x = np.array(mpu_array[:,cal_rot_indicies[jj][0]]) # raw x-variable
					y = np.array(mpu_array[:,cal_rot_indicies[jj][1]]) # raw y-variable
					x_prime = x - cal_offsets[cal_rot_indicies[jj][0]] # x-var for heading
					y_prime = y - cal_offsets[cal_rot_indicies[jj][1]] # y-var for heading
					x_prime[np.isnan(x)] = np.nan
					y_prime[np.isnan(y)] = np.nan
					r_var = np.sqrt(np.power(x_prime,2.0)+np.power(y_prime,2.0)) # radius vector
					theta = np.arctan2(-y_prime,x_prime) # angle vector for heading
					theta1 = 180 *theta/pi
                    
                    
					if (ii==6):	
						angle.append(theta1)                 
			yaw_deg = np.nanmean(angle)
			if (yaw_deg < 0):
				yaw_deg = 360 + yaw_deg
			ii_iter = 0 # reset plot counter
		ii_iter+=1 # update plot counter

def dms_to_decimal(coords):
	# Fuction to convert GPA format of GPS coordinate to decimal format 
	degrees, minutes, seconds, ms = coords
	print(degrees, minutes, seconds, ms)
	return float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600) + (float(ms)/216000)

def parse_input(input_string):
	# Split the input string at the decimal point
	parts = input_string.split(".")

	# The first two characters will be in the first part, before the decimal point
	a = parts[0][:2]

	# The second two characters will be in the first part, after the first two characters
	b = parts[0][2:4]

	# All characters after the decimal point will be in the second part
	c = parts[1][:2]

	d = parts[1][2:4]

	return (a, b, c, d)
	
def get_currgps():
	while True:
		# Read and print GPS data
		try:
			line = ser.readline().decode().strip()
			# Split the line into fields
			fields = line.split(",")
			# Check if this is a GPGGA line
			if fields[0] == "$GPGGA":
			# Extract the longitude and latitude fields
				lon = fields[4]
				lat = fields[2]
				#print(parse_input(str(float(lon))))
				try:
					new_lon = dms_to_decimal(parse_input(str(float(lon))))
					new_lat = dms_to_decimal(parse_input(str(float(lat))))
					print(f"Longitude: {str(new_lon)} Latitude: {str(new_lat)}")
					return (new_lat, -new_lon)
				except:
					print("1")
		except:
			print("2")
		

 
# Initial request of IGS device (will be updated so it listens and works on request of user)
tts = gTTS("Where would you like to go?")
tts.save("texttospeech.mp3")
song = AudioSegment.from_mp3("texttospeech.mp3")
play(song)

# Record user's response (4 seconds allocated)
record = "arecord -D hw:tegrasndt210ref,0 -r 48000 -f S32_LE -c 1 -d 4 recording.wav"
p = subprocess.Popen(record, shell = True)
sleep(5)

# Recognizes user's response and converts to text
r = sr.Recognizer()

hellow=sr.AudioFile('recording.wav')
with hellow as source:
    audio = r.record(source)
try:
    s = r.recognize_google(audio)
    print("Text: "+s)
except Exception as e:
    print("Exception: "+str(e))

# Get current coordinates
coords = get_currgps()

# API initialized with key
gmaps = googlemaps.Client(key="AIzaSyBYTAEiKD0SzXxJ1qCtjjYA00cFOGPZSLQ")

# Set the destination location
destination = s
print("")
print(destination)
print("")

# Find the address of the user requested spot
response = gmaps.places(destination, location="{coords[0]}, {coords[1]}", radius=2500, type="restaurant")

lat_start = 32.947070
lon_start = -96.962196

print(response["results"][0]['formatted_address'])

# Get the first result
result = response["results"][0]['formatted_address']

# Set the API key and the interval for updates (in seconds)
#api_key = "AIzaSyBYTAEiKD0SzXxJ1qCtjjYA00cFOGPZSLQ"
interval = 5

if __name__ == "__main__":
	thread = Thread(target=imu_thread)
	thread.daemon = True
	thread.start()
	
	rose = "none"
	#thread1 = Thread(target=vib_thread)
	#thread1.daemon = True
	#thread1.start()

	while True:
		# Get the current GPS coordinates
		#current_location = (new_lat, -new_lon)
		origin=get_currgps()
		
		gps_filename = "gps.csv"

		with open(gps_filename,'a+',newline='') as csvfile:
				writer = csv.writer(csvfile,delimiter=',')
				writer.writerow(origin)
                
		# Make the request to the Directions API
		directions = gmaps.directions(origin, result, mode="walking", departure_time=datetime.now())
		route = directions[0]

		# Extract the list of directions from the route
		directions = route['legs'][0]['steps'][0]
		print(directions)
		soup = BeautifulSoup(directions['html_instructions'], 'html.parser')
		text = soup.get_text()
		soup2 = BeautifulSoup(directions['distance']['text'], 'html.parser')
		text2 = soup2.get_text()
		print(text + " for " + text2)
		
		words = text.split()
		
		for i in words:
			if i == "east" or i == "west" or i == "north" or i == "south" or i == "southeast" or i == "northeast" or i == "southwest" or i == "southeast":
				rose = i
				print(rose)
				break
		instruction = text + " for " + text2
		
		# Say the direction and distance
		tts = gTTS(instruction)
		tts.save("texttospeech.mp3")
		song = AudioSegment.from_mp3("texttospeech.mp3")
		play(song)

		# Wait for the specified interval before getting the next update
		sleep(interval)
		
		
	ser.close()

