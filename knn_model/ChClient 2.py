import socket
import threading

HOST = '192.168.1.12'  
PORT = 12345

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

def receive():
    while True:
        try:
            msg = client.recv(1024).decode()
            print(f"\n상대: {msg}")
        except:
            print("연결 종료")
            break

def send():
    while True:
        msg = input()
        client.send(msg.encode())

recv_thread = threading.Thread(target=receive)
recv_thread.start()

send_thread = threading.Thread(target=send)
send_thread.start()