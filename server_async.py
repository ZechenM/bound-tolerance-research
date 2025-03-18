import asyncio
import pickle
import struct

import torch

# from typing import Any, Dict, List, Tuple
import config

print(f"Compression Method: {config.compression_method}")

DEBUG_MODE = 0


class AsyncServer:
    def __init__(self, host="localhost", port=60000, num_workers=3):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.connections = []
        self.conn_addr_map = {}

    async def recv_all(self, reader, size):
        """Helper function to receive all data asynchronously"""
        data = b""
        while len(data) < size:
            chunk = await reader.read(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    async def worker_connection_handler(self, reader, writer) -> None:
        """Handle a single worker connection"""
        addr = writer.get_extra_info("peername")
        print(f"Worker connected from {addr}")
        self.conn_addr_map[writer] = addr
        self.connections.append((reader, writer))

        if len(self.connections) == self.num_workers:
            print("All workers connected. Starting gradient exchange...")
            await self.recv_send()

    async def recv_send(self):
        """Receive gradients from all workers concurrently and send back averaged gradients"""
        tasks = [self.recv_worker_grad(reader) for reader, _ in self.connections]
        gradients = await asyncio.gather(*tasks)

        if not gradients:
            print("Failed to receive gradients.")
            return

        print("All gradients received.")

        # Compute the average gradients
        avg_gradients = {}
        for key in gradients[0].keys():
            avg_gradients[key] = torch.stack([grad[key] for grad in gradients]).mean(dim=0)

        if DEBUG_MODE:
            for param in avg_gradients:
                print(f"the size of {param}: {len(avg_gradients[param])}")
                print(f"{param} after averaging:")
                for weight in avg_gradients[param]:
                    print(weight)

        # Compress the averaged gradients
        compressed_avg_gradients = config.compress(avg_gradients)
        avg_gradients_data = pickle.dumps(compressed_avg_gradients)

        # Send averaged gradients back to all workers concurrently
        send_tasks = [self.send_avg_grad(writer, avg_gradients_data) for _, writer in self.connections]
        await asyncio.gather(*send_tasks)

        # Close worker connections
        for _, writer in self.connections:
            writer.close()
            await writer.wait_closed()
        self.connections.clear()
        self.conn_addr_map.clear()
        print("Closed all worker connections.")

    async def recv_worker_grad(self, reader):
        """Receive a single worker's gradient asynchronously"""
        # Receive the size of the incoming data
        size_data = await self.recv_all(reader, 4)
        if not size_data:
            print("Failed to receive data size.")
            return None
        size = struct.unpack("!I", size_data)[0]

        # Receive the actual data
        data = await self.recv_all(reader, size)
        if not data:
            print("Failed to receive data.")
            return None

        # Send ACK
        writer = next(writer for r, writer in self.connections if r == reader)
        writer.write(b"A")
        await writer.drain()

        compressed_grad = pickle.loads(data)
        grad = config.decompress(compressed_grad)

        print(f"Received gradients from worker {self.conn_addr_map[writer]}")
        return grad

    async def send_avg_grad(self, writer, avg_gradients_data):
        """Send averaged gradients to a worker asynchronously"""
        writer.write(struct.pack("!I", len(avg_gradients_data)))
        writer.write(avg_gradients_data)
        await writer.drain()
        print(f"Sent averaged gradients to worker {self.conn_addr_map[writer]}")

    async def run_server(self) -> None:
        """Start the asyncio server"""
        server = await asyncio.start_server(self.worker_connection_handler, self.host, self.port)
        addr = server.sockets[0].getsockname()
        print(f"Server listening on {addr}...")

        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    server = AsyncServer()
    asyncio.run(server.run_server())
