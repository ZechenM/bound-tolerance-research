import pickle
import struct


def send_signal_tcp(sock, signal, counter):    
    """
    Send a signal over a TCP socket.
    
    :param sock: The TCP socket to send the signal through.
    :param signal: The signal to send, which should be a single byte.
    """
    if not isinstance(signal, bytes) or len(signal) != 1:
        raise ValueError("Signal must be a single byte.")
    try:
        signal_with_counter = struct.pack("!cI", signal, counter)
        sock.sendall(signal_with_counter)
    except Exception as e:
        print(f"Error sending signal: {e}")

def recv_signal_tcp(sock):
    """
    Receive a signal over a TCP socket.
    
    :param sock: The TCP socket to receive the signal from.
    :return: The received signal as a single byte.
    """
    try:
        signal = sock.recv(5)  # 1 byte for signal + 4 bytes for counter
        if len(signal) < 5:
            raise ValueError("Received signal is incomplete.")
        signal, counter = struct.unpack("!cI", signal)
        if not isinstance(signal, bytes) or len(signal) != 1:
            raise ValueError("Received signal is not a single byte.")

        return signal, int(counter)
    except Exception as e:
        raise ValueError(f"Error receiving signal: {e}")

def send_data_tcp(sock, data):
    """
    Send data over a TCP socket.
    
    :param sock: The TCP socket to send data through.
    :param data: The data to send, which should be serializable.
    """
    try:
        serialized_data = pickle.dumps(data)
        data_length = len(serialized_data)
        # Send the length of the data first
        sock.sendall(struct.pack('!I', data_length))
        sock.sendall(serialized_data)
    except Exception as e:
        print(f"Error sending data: {e}")
        
def recv_all(sock, size):
    """
    Receive all data from a socket until the specified size is reached.
    
    :param sock: The socket to receive data from.
    :param size: The number of bytes to receive.
    :return: The received data as bytes.
    """
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ValueError("Socket connection closed before receiving all data.")
        data += chunk
    return data
        
def receive_data_tcp(sock):
    """
    Receive data over a TCP socket.
    
    :param sock: The TCP socket to receive data from.
    :return: The deserialized data received from the socket.
    """
    try:
        # First, receive the length of the incoming data
        length_bytes = recv_all(sock, 4) 
        if not length_bytes:
            raise ValueError("Failed to receive data length, connection may be closed.")

        data_length = struct.unpack('!I', length_bytes)[0]
        serialized_data = recv_all(sock, data_length)
        if not serialized_data:
            raise ValueError("Failed to receive serialized data, connection may be closed.")
        
        return pickle.loads(serialized_data)
    except Exception as e:
        raise ValueError(f"Error receiving data: {e}")