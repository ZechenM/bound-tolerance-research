import argparse
import pickle
import random
import socket
import struct
from enum import Enum

import torch

import config
import mlt


class TrainingPhase(Enum):
    BEGIN = 0
    MID = 1
    FINAL = 2


class Server:
    def __init__(self, host, port, num_workers=3):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.training_phase = TrainingPhase.BEGIN
        self.server_socket = None
        self.prev_avg_acc = 0.0
        self.worker_eval_acc = []
        self.worker_epochs = []

        self.drop_rate = 0.0  # X% probability to zero out gradients

        # v-threshold configurations and trackers
        self.enable_v_threshold = False
        self.v = 0  # just like TCP, start with 0, update additively
        self.iter_count = 0  # increment every iter processed. If reached self.update_v_per, clear, and update v accordingly
        self.update_v_per = 100  # update v per 100 iter
        # self.AIDM_add = 0.1       # under-dropped, v = v + self.AIDM_add
        self.AIMD_decrease = 0.5  # over-dropped, v = v * self.AIMD_decrease
        # self.overall_max_abs_value = 0
        self.median_tracker = 0
        self.counter = 0
        self.total_params = 0
        self.zeroed_params = 0

        # gradient communication protocal configurations
        self.protocol = config.protocol
        self.UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.UDP_socket.bind(("0.0.0.0", self.port + 1))  # TODO: each worker should allocate one UDP socket
        self.loss_tolerance = config.loss_tolerance

        self.write_to_server_port()
        self.start_server()
        self.run_server()
        # Thread(target=self.handle_user_input, daemon=True).start()

    def write_to_server_port(self):
        print("Writing server port to .server_port file...")
        with open(".server_port", "w") as f:
            f.write(str(self.port))
            f.flush()

    def start_server(self) -> None:
        # Create a socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_workers)
        self.is_listening = True
        print(f"Server listening on {self.host}:{self.port}...")

        print(f"V-threshold enabled: {self.enable_v_threshold}")

    def recv_all(self, conn, size):
        """helper function to receive all data"""
        data = b""
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                return None
            data += packet

        return data

    def close_worker(self):
        """Close all worker connections"""
        for conn in self.connections:
            conn.close()
        self.connections = []
        self.conn_addr_map = {}
        # print("Closed all worker connections.")

    def worker_connection_handler(self) -> None:
        """Wait for all workers to connect"""
        self.connections = []
        self.conn_addr_map = {}
        for _ in range(self.num_workers):
            conn, addr = self.server_socket.accept()
            self.connections.append(conn)
            self.conn_addr_map[conn] = addr
            print(f"Connected to worker at {addr}")

        print("All workers connected.")

    def _training_phase_update(self):
        if len(self.worker_eval_acc) < 3:
            self.has_eval_acc = False
            return

        curr_avg_acc = sum(self.worker_eval_acc) / len(self.worker_eval_acc)
        acc_diff = curr_avg_acc - self.prev_avg_acc

        proposed_phase = self.training_phase  # Initially set as current phase

        if any(self.worker_epochs) == 0:
            proposed_phase = TrainingPhase.BEGIN
        elif acc_diff > 0.07:
            proposed_phase = TrainingPhase.BEGIN
        elif curr_avg_acc > 0.5:
            if acc_diff < 0.03:
                proposed_phase = TrainingPhase.FINAL
            elif acc_diff <= 0.07:
                proposed_phase = TrainingPhase.MID
            else:
                proposed_phase = TrainingPhase.BEGIN
        # Ensure no phase transitions if the accuracy is not above 50%, even if acc_diff is low
        else:
            proposed_phase = TrainingPhase.BEGIN

        # Ensure one-way state transitions:
        if proposed_phase.value >= self.training_phase.value:
            self.training_phase = proposed_phase

        self.prev_avg_acc = curr_avg_acc

        print(f"All worker eval acc: {self.worker_eval_acc}")
        print(f"All worker epochs: {self.worker_epochs}")
        print(f"Current averaged accuracy: {self.prev_avg_acc}")
        print(f"Current training phase: {self.training_phase}")

        self.worker_eval_acc.clear()
        self.worker_epochs.clear()

    def recv_data_TCP(self, TCP_sock):
        # Receive the size of the incoming data
        size_data = self.recv_all(TCP_sock, 4)
        if not size_data:
            raise ValueError("Failed to receive data size.")

        size = struct.unpack("!I", size_data)[0]

        # Receive the actual data
        data = self.recv_all(TCP_sock, size)
        if not data:
            raise ValueError("Failed to receive data.")

        # response with ACK
        # TCP_sock.sendall(b"A")

        grad = pickle.loads(data)

        return grad

    def send_data_TCP(self, TCP_sock, gradient):
        # Send the size of the data first
        TCP_sock.sendall(struct.pack("!I", len(gradient)))
        # Sendall the actual data
        TCP_sock.sendall(gradient)

    def recv_send(self):
        """Receive gradients from all workers and send back averaged gradients"""
        gradients = []
        self.has_eval_acc = False

        for conn in self.connections:
            if config.protocol == "MLT":
                # UDP socket needs to be changed to a 1-to-1 correspondence with each worker
                # for the toy example, we can only have one worker and one server
                socks = {"tcp": conn, "udp": self.UDP_socket}
                grad = mlt.recv_data_MLT(socks)
            elif config.protocol == "TCP":
                grad = self.recv_data_TCP(conn)
            else:
                raise TypeError(f"protocol {self.protocol} is not supported (TCP | MLT)")

            if grad is None:
                print(f"Failed to receive data from worker {self.conn_addr_map[conn]}.")
                continue

            if "eval_acc" in grad and "epoch" in grad:
                self.has_eval_acc = True
                self.worker_eval_acc.append(grad["eval_acc"])
                self.worker_epochs.append(grad["epoch"])
                del grad["epoch"]
                del grad["eval_acc"]

            gradients.append(grad)
            if config.DEBUG:
                print(f"Received gradients from worker {self.conn_addr_map[conn]}: {grad.keys()}")

        if config.DEBUG:
            print(f"Received {len(gradients)} gradients from workers.")

        # check accuracy and update training phase accrodingly
        if self.has_eval_acc:
            self._training_phase_update()

        # based on training phase, update the drop rate
        if self.training_phase == TrainingPhase.BEGIN:
            self.drop_rate = 0.0
        elif self.training_phase == TrainingPhase.MID:
            self.drop_rate = 0.0
        elif self.training_phase == TrainingPhase.FINAL:
            self.drop_rate = 0.0

        if self.has_eval_acc:
            print(f"Current drop rate: {self.drop_rate}\n")

        # print(f"1. type of gradients: {type(gradients)}, should be list")
        # print(f"2. len of gradients: {len(gradients)}, should be 3")
        # print(f"3. type of gradients[0]: {type(gradients[0])}, should be dict")
        # print("gradients looks like:", gradients)
        avg_gradients = {}
        if not self.enable_v_threshold:
            # Existing averaging logic:
            for key in gradients[0].keys():
                if random.random() < self.drop_rate:  # x% probability to zero out gradients
                    avg_gradients[key] = torch.zeros_like(gradients[0][key])
                    # print(f"Gradient '{key}' zeroed out randomly.")
                else:
                    avg_gradients[key] = torch.stack([grad[key] for grad in gradients]).mean(dim=0)
        else:  # v-threshold is enabled
            for grad_dict in gradients:
                for key in grad_dict:
                    tensor = grad_dict[key]

                    # Update max absolute value
                    # self.overall_max_abs_value = max(self.overall_max_abs_value, tensor.abs().max().item())
                    self.median_tracker += tensor.mean().item()
                    self.counter += 1

                    # Count parameters
                    self.total_params += tensor.numel()

                    # Zero out small values
                    mask = tensor.abs() < self.v
                    self.zeroed_params += mask.sum().item()
                    tensor[mask] = 0  # In-place modification

            # right gradients are set to 0 in-place, calculate average
            for key in gradients[0].keys():
                avg_gradients[key] = torch.stack([grad[key] for grad in gradients]).mean(dim=0)

            print(
                f"current v: {self.v}",
                f"iter_count {self.iter_count}",
                f"zeroed_param {self.zeroed_params}",
                f"total params: {self.total_params}",
                f"drop rate: {self.zeroed_params / self.total_params}",
                flush=True,
            )

            self.iter_count += 1
            if self.iter_count >= self.update_v_per:
                # clear, and update v accordingly
                self.iter_count = 0
                actual_drop_rate = self.zeroed_params / self.total_params
                if actual_drop_rate < self.drop_rate:
                    # increase v, zero more gradients, increase drop rate
                    # see https://github.com/ZechenM/bound-tolerance-research/issues/9
                    # self.v += self.overall_max_abs_value * 0.0001
                    real_median = self.median_tracker / self.counter
                    self.v += abs(real_median) * 0.5

                else:  # actual_drop_rate >= self.drop_rate
                    # decrease v, allow more grdients to get passed, decrease drop rate
                    self.v *= self.AIMD_decrease
                self.overall_max_abs_value = 0
                self.total_params = 0
                self.zeroed_params = 0

        # Send averaged gradients back to all workers
        for conn in self.connections:
            tcp_sock = conn
            ip, port = self.conn_addr_map[conn]
            if self.protocol == "MLT":
                socks = {"tcp": tcp_sock, "udp": self.UDP_socket}
                receiver = {"ip": ip, "port": port}

                # Before serializing and send the tensor data, 2 IMPORTANT STEPS
                # STEP 0: send N signal as the server will NEVER send the E signal
                try:
                    tcp_sock.sendall(b"N")
                except Exception as e:
                    print(f"Failed to send N signal to worker {self.conn_addr_map[conn]}: {e}")
                    continue
                # STEP 1: send how many sub-tensors (subgradients / key-val pairs) will be sent
                num_subgradients = len(avg_gradients)
                try:
                    tcp_sock.sendall(struct.pack("!I", num_subgradients))
                except Exception as e:
                    print(f"Failed to send number of subgradients to worker {self.conn_addr_map[conn]}: {e}")
                    continue

                for key, tensor in avg_gradients.items():
                    # Serialize each tensor to custom binary format
                    averaged_tensor_bytes = mlt.serialize_gradient_to_custom_binary(tcp_sock, key, tensor)
                    mlt.send_data_MLT(socks, receiver, averaged_tensor_bytes)
            elif self.protocol == "TCP":
                self.send_data_TCP(tcp_sock, pickle.dumps(avg_gradients))
            else:
                raise TypeError(f"protocol {self.protocol} is not supported (TCP | MLT)")
            # print(f"Sent averaged gradients to worker {self.conn_addr_map[conn]}")

    def run_server(self) -> None:
        while self.is_listening:
            # server accepts connections from all the workers
            self.worker_connection_handler()

            # server receives gradients from all workers and sends back averaged gradients
            self.recv_send()

            # close all worker connections
            self.close_worker()

    def print_menu_options(self) -> None:
        print("Enter 'q' to close this server.")

    # close the server
    def close_server(self) -> None:
        """Close the server"""
        self.is_listening = False
        self.server_socket.close()
        print("Server closed.")

    def handle_user_input(self) -> None:
        self.print_menu_options()
        user_input = input()
        if len(user_input) == 0:
            print("Invalid option.")
            return
        user_input = user_input[0].lower() + user_input[1:]
        # case: close the server
        if user_input[0] == "q":
            self.close_server()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="60001")
    args = parser.parse_args()

    server_host = str(args.host)
    server_port = int(args.port)
    print(f"Starting server at port with {server_port}...")
    server = Server(server_host, server_port)
