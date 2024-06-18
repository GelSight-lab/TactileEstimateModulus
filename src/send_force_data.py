import time
import sys
import socket

import RPi.GPIO as GPIO
from hx711py.hx711 import HX711

referenceUnit = 1

def cleanAndExit():
    GPIO.cleanup() 
    print("Cleaned!")
    sys.exit()

# Define the IP address and port of the receiving device
server_ip = "172.16.0.69"  # Replace with the IP address of the receiving device
server_port = 8888   # Replace with the port number you want to use

def send_data():

    hx = HX711(5, 6)
    hx.set_reading_format("MSB", "MSB")
    hx.set_reference_unit(referenceUnit)
    hx.reset()
    hx.tare()

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(10000)
    
    # Connect to the server
    # while True:
    #     try:
    client_socket.connect((server_ip, server_port))
        #     break

        # except (TimeoutError, ConnectionRefusedError, ConnectionAbortedError, ConnectionError, ConnectionResetError):
        #     pass

        # except (KeyboardInterrupt, SystemExit):
        #     print('ERROR')
        #     cleanAndExit()
        #     return False

    while True:
        try:
            val = hx.get_weight(5)
            val_str = "{:.2f}".format(val)
            print(val_str)
            client_socket.send(val_str.encode())

        except (ConnectionResetError, ConnectionRefusedError, ConnectionError, ConnectionAbortedError):
            client_socket.close()
            return True

        except (KeyboardInterrupt, SystemExit):
            print('ERROR')
            cleanAndExit()
            client_socket.close()
            return False

keep_sending = True
while keep_sending:
    keep_sending = send_data()

cleanAndExit()