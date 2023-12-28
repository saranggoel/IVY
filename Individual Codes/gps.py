import serial
import csv
import time

# Open serial port
ser = serial.Serial("/dev/ttyTHS1", 9600, timeout=1)

def dms_to_decimal(coords):
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

time.sleep(60)

while True:
		go = True
		# Read and print GPS data
		start = time.time()
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
					origin = (new_lat, -new_lon)
					while go:
						if (time.time() - start >= 1): 
							with open("gps_loop.csv",'a+',newline='') as csvfile:
								writer = csv.writer(csvfile,delimiter=',')
								writer.writerow(origin)
							go = False
							print(f"Longitude: {str(new_lon)} Latitude: {str(new_lat)} Time: " + str(time.time()))
					
					
				except:
					print("1")
		except:
			print("2")
		
		
		
# Close serial port
ser.close()

