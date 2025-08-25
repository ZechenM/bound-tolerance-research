import datetime
import os
import time

from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.topo import Topo

PYTHON = "/root/shared/.venv/bin/python3.12"


class WorkerServerTopo(Topo):
    """
    Custom Topology for a distributed ML setup:
    3 workers and 1 server connected via 1 switch.
    Links between workers and swtich have (varying?) loss rates
    """

    def __init__(self, **opts):
        "Create custom topology"
        # Initialize topology
        Topo.__init__(self, **opts)

        # add nodes (hosts: servers and workers; switch)
        server = self.addHost("server")
        worker0 = self.addHost("worker0")
        worker1 = self.addHost("worker1")
        worker2 = self.addHost("worker2")

        # using default openVswitch (OVS) kernel switch
        s1 = self.addSwitch("s1")

        # Define link characteristics
        link_opts_base = dict(bw=1000, delay="1ms", loss=0.1, max_queue_size=1000, use_htb=True)
        link_opts_server = dict(bw=1000, delay="1ms", loss=0.1, max_queue_size=1000, use_htb=True)

        info("*** Adding Worker Links with Loss:\n")
        info(f"* Worker i <-> Switch: {link_opts_base}\n")

        # add links between workers and the switch with specified options
        self.addLink(worker0, s1, **link_opts_base)
        self.addLink(worker1, s1, **link_opts_base)
        self.addLink(worker2, s1, **link_opts_base)

        info(f"*** Adding Server Link:\n* Server <-> Switch: {link_opts_base}\n")
        self.addLink(server, s1, **link_opts_server)


def run_experiment():
    """
    Creates the network, runs the distributed ML applications, and starts the CLI.
    """
    # --- Configuration ---
    # Assumes your project scripts are located here within the Mininet VM/environment
    # Adjust this path if your scripts are located elsewhere
    PROJECT_DIR = "/root/shared"

    SERVER_SCRIPT = f"{PROJECT_DIR}/server_multithreading.py"
    WORKER_SCRIPT = f"{PROJECT_DIR}/worker_multithreading.py"
    INITIAL_SERVER_PORT = 60001  # Default port used in your script

    # Define log paths within the Mininet nodes' filesystems
    # Using /tmp/ for simplicity, adjust if needed
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    LOG_DIR_BASE = f"{PROJECT_DIR}/mininet_logs/{timestamp}"
    SERVER_LOG = f"{LOG_DIR_BASE}/server.log"
    WORKER0_LOG = f"{LOG_DIR_BASE}/worker0.log"
    WORKER1_LOG = f"{LOG_DIR_BASE}/worker1.log"
    WORKER2_LOG = f"{LOG_DIR_BASE}/worker2.log"
    # --- End Configuration ---

    # Create an instance of the custom topology
    topo = WorkerServerTopo()

    # Create a Mininet network instance using the custom topology
    # Crucially, specify link=TCLink to enable loss, bw, delay configuration
    net = Mininet(
        topo=topo,
        switch=OVSKernelSwitch,  # Default switch type
        controller=RemoteController,  # No SDN controller needed for basic L2 switching
        link=TCLink,  # Use TCLink for traffic control
        autoSetMacs=True,
    )  # Automatically set MAC addresses

    info("*** Starting network\n")
    net.start()

    # Get handles to the nodes
    server_node = net.get("server")
    worker0_node = net.get("worker0")
    worker1_node = net.get("worker1")
    worker2_node = net.get("worker2")

    # Get server IP address for workers to connect to
    server_ip = server_node.IP()
    info(f"Server IP Address: {server_ip}\n")
    info(f"Server Port: {INITIAL_SERVER_PORT}\n")

    # Create log directory for all hosts
    info(f"*** Creating log directory '{LOG_DIR_BASE}' on nodes...\n")
    for node in [server_node, worker0_node, worker1_node, worker2_node]:
        # Optional: Ensure the project directory exists if needed by scripts
        node.cmd(f"mkdir -p {PROJECT_DIR}")  # Uncomment if scripts expect it
        node.cmd(f"mkdir -p {LOG_DIR_BASE}")

    # --- Start the distributed ML applications ---
    info("*** Starting Server Application...\n")
    # Command to start the server, redirecting output to its log file
    # Uses python -u for unbuffered output, runs in background (&)
    # TODO: figure out if "runs in background is needed or not"
    server_cmd = f"{PYTHON} -u {SERVER_SCRIPT} --host {server_ip} --port {INITIAL_SERVER_PORT}"
    info(f"Executing on server: {server_cmd}\n")

    # Create log file for redirecting output
    os.makedirs(os.path.dirname(SERVER_LOG), exist_ok=True)
    # server_node.popen(server_cmd.split(), stdout=open(SERVER_LOG, "w"), stderr=open(SERVER_LOG, "a"))
    server_proc = server_node.popen(
        server_cmd.split(),
        stdout=open(SERVER_LOG, "w"),
        stderr=open(SERVER_LOG, "a"),
    )

    # Give the server a moment to start up and bind to the port
    info("*** Waiting for server to initialize...\n")
    time.sleep(3)  # Adjust sleep time if server needs longer

    info("*** Starting Worker Applications...\n")
    # Using popen for better process management

    # Create log files for redirecting output
    for log_file in [WORKER0_LOG, WORKER1_LOG, WORKER2_LOG]:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Commands to start the workers, passing worker ID, server IP, and server port
    # Worker 0
    worker0_cmd = f"{PYTHON} -u {WORKER_SCRIPT} 0 {server_ip} {INITIAL_SERVER_PORT}"
    info(f"Executing on worker0: {worker0_cmd}\n")
    worker0_proc = worker0_node.popen(
        worker0_cmd.split(),
        stdout=open(WORKER0_LOG, "w"),
        stderr=open(WORKER0_LOG, "a"),
    )

    # Worker 1
    worker1_cmd = f"{PYTHON} -u {WORKER_SCRIPT} 1 {server_ip} {INITIAL_SERVER_PORT}"
    info(f"Executing on worker1: {worker1_cmd}\n")
    worker1_proc = worker1_node.popen(
        worker1_cmd.split(),
        stdout=open(WORKER1_LOG, "w"),
        stderr=open(WORKER1_LOG, "a"),
    )

    # Worker 2
    worker2_cmd = f"{PYTHON} -u {WORKER_SCRIPT} 2 {server_ip} {INITIAL_SERVER_PORT}"
    info(f"Executing on worker2: {worker2_cmd}\n")
    worker2_proc = worker2_node.popen(
        worker2_cmd.split(),
        stdout=open(WORKER2_LOG, "w"),
        stderr=open(WORKER2_LOG, "a"),
    )

    # --- End Application Start ---

    info("\n*** Running basic connectivity tests (pingall)...\n")
    # Optional: Run pingall to check basic connectivity after network start
    # Output might be interleaved with application logs if they print to stdout/stderr
    net.pingAll()

    info("\n*** Applications are running in the background.\n")
    info(f"*** Server log: {server_node.name}:{SERVER_LOG}\n")
    info(f"*** Worker 0 log: {worker0_node.name}:{WORKER0_LOG}\n")
    info(f"*** Worker 1 log: {worker1_node.name}:{WORKER1_LOG}\n")
    info(f"*** Worker 2 log: {worker2_node.name}:{WORKER2_LOG}\n")
    info("*** Starting Mininet CLI. Type 'exit' to quit and stop the network.\n")
    info(f"*** You can monitor logs using commands like: tail -f {WORKER0_LOG}\n")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        info("\n*** Keyboard interrupt received. Stopping network...")
    finally:
        # 2. Terminate all processes in the finally block
        print("Cleaning up processes...")
        if server_proc:
            print("Terminating server...")
            server_proc.terminate()
        if worker0_proc:
            print("Terminating worker 0...")
            worker0_proc.terminate()
        # Add termination for worker1_proc and worker2_proc as well
        if worker1_proc:
            print("Terminating worker 1...")
            worker1_proc.terminate()
        if worker2_proc:
            print("Terminating worker 2...")
            worker2_proc.terminate()

        info("*** Stopping network\n")
        net.stop()

    # # Start the Mininet Command Line Interface for interactive commands
    # CLI(net)    # TODO: implement network stopping?

    # info("*** Stopping network\n")
    # # Stop of network emulation and clean up resources
    # net.stop()


if __name__ == "__main__":
    start_time = time.time()
    topos = {"mltopo": (lambda: WorkerServerTopo())}
    setLogLevel("info")
    run_experiment()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")