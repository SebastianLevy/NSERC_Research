import serial
import time

# Open the serial port. Replace 'COM3' with the appropriate port for your system.
# On many Unix systems, it might be something like '/dev/ttyUSB0' or '/dev/ttyACM0'.
ser = serial.Serial('COM3', 9600)

# Open a file to save the data. 'data.txt' can be replaced with your desired file name.
with open('data.txt', 'w') as file:
    try:
        while True:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            
            # Write the line to the file
            file.write(line + '\n')

            # Optional: print the line to the console
            print(line)
            
    except KeyboardInterrupt:
        # Exit the loop when Ctrl+C is pressed
        pass

# Close the serial port
ser.close()