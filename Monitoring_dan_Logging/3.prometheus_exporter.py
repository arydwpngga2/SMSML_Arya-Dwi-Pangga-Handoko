from prometheus_client import start_http_server, Gauge
import psutil
import time

cpu_usage = Gauge(
    "cpu_usage_percent",
    "CPU Usage"
)

memory_usage = Gauge(
    "memory_usage_percent",
    "Memory Usage"
)

disk_usage = Gauge(
    "disk_usage_percent",
    "Disk Usage"
)

network_sent = Gauge(
    "network_bytes_sent",
    "Network Sent"
)

network_recv = Gauge(
    "network_bytes_recv",
    "Network Received"
)

start_http_server(8000)

while True:
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage('/').percent)
    network = psutil.net_io_counters()
    network_sent.set(network.bytes_sent)
    network_recv.set(network.bytes_recv)

    time.sleep(5)