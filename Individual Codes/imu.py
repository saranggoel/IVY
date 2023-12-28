# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_icm20x
from math import atan2, pi, sqrt, sin, cos
from collections import deque 

class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.queue = deque()
        self.size = size



    def next(self, val: float):   #def next(self, val: int) -> float:
        if len(self.queue) == self.size:
            self.queue.popleft()
            self.queue.append(val)
        else:
            self.queue.append(val)
        return sum(self.queue)/len(self.queue)

i2c = board.I2C()  # uses board.SCL and board.SDA
icm = adafruit_icm20x.ICM20948(i2c)
num=5
x=0
while True:
    #print("Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (icm.acceleration))
    #print("Gyro X:%.2f, Y: %.2f, Z: %.2f rads/s" % (icm.gyro))
    
    #print("")
	accelX = (icm.acceleration)[0]
	accelY = (icm.acceleration)[1]
	accelZ = (icm.acceleration)[2]
	magReadX = (icm.magnetic)[0]
	magReadY = (icm.magnetic)[1]
	magReadZ = (icm.magnetic)[2]
	ax = MovingAverage(num)
	ay = MovingAverage(num)
	az = MovingAverage(num)
	accelX=ax.next(accelX)
	accelY=ay.next(accelY)
	accelZ=az.next(accelZ)
	mx = MovingAverage(num)
	my = MovingAverage(num)
	mz = MovingAverage(num)
	magReadX=mx.next(magReadX)
	magReadY=my.next(magReadY)
	magReadZ=mz.next(magReadZ)
	magNorm = sqrt((magReadX*magReadX)+(magReadY*magReadY)+(magReadZ*magReadZ))
	magReadX = magReadX/magNorm
	magReadY = magReadY/magNorm
	magReadZ = magReadZ/magNorm    
	pitch = atan2(accelX, sqrt(accelY*accelY + accelZ*accelZ))
	roll = atan2(accelY, sqrt(accelX*accelX + accelZ*accelZ))
	pitch_deg = 180 * atan2(accelX, sqrt(accelY*accelY + accelZ*accelZ)) / pi
	roll_deg = 180 * atan2(accelY, sqrt(accelX*accelX + accelZ*accelZ)) / pi
	mag_x = magReadX*cos(pitch) + magReadY*sin(roll)*sin(pitch)+magReadZ*cos(roll)*sin(pitch)
	mag_y = magReadY*cos(roll) - magReadZ * sin(roll)
	yaw_deg = 360 * atan2(-mag_y,mag_x)/pi

	print("Pitch: " + str(pitch_deg) + " Roll: " + str(roll_deg) + " Yaw: " + str(yaw_deg))
	x+=1
	if x%30 == 0:
		print("papapadooo")
		#print("Magnetometer X:%.2f, Y: %.2f, Z: %.2f uT" % (icm.magnetic))
		#print("Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (icm.acceleration))
	#time.sleep(0.05)
