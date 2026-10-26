import socket
import threading

HOST = '0.0.0.0'  # 모든 IP에서 접속 허용
PORT = 12345

clients = []

def handle_client(conn, addr):
    print(f"[접속] {addr}")
    clients.append(conn)
    try:
        while True:
            msg = conn.recv(1024)
            if not msg:
                break
            for client in clients:
                if client != conn:
                    client.send(msg)
    except:
        pass
    finally:
        conn.close()
        clients.remove(conn)
        print(f"[종료] {addr}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print(f"[서버 대기 중] 포트 {PORT}")

while True:
    conn, addr = server.accept()
    thread = threading.Thread(target=handle_client, args=(conn, addr))
    thread.start()
