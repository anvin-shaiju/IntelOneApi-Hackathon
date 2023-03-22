import RPi.GPIO as GPIO
import time
import serial

# Set up GPIO pins for ADC communication
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(24, GPIO.IN)

# Set up serial communication with Android device
ser = serial.Serial('/dev/rfcomm0', 9600)

while True:
    # Read analog value from ADC
    GPIO.output(18, GPIO.HIGH)
    time.sleep(0.000002)
    GPIO.output(18, GPIO.LOW)

    while GPIO.input(24) == GPIO.LOW:
        pass
    start = time.time()

    while GPIO.input(24) == GPIO.HIGH:
        pass
    end = time.time()

    duration = end - start
    analog= duration * 1000000 / 58.0

    # Send analog value to Android device
    ser.write(str(analog).encode('utf-8'))