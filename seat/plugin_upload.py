#!/usr/bin/env python3
# coding=utf-8
"""
This script uploads a plugin package to the QGIS plugin repository.

Authors: A. Pasotti, V. Picavet
"""

import sys
import getpass
import logging
import xmlrpc.client
import argparse
from pathlib import Path

# Configuration
PROTOCOL = "https"
SERVER = "plugins.qgis.org"
PORT = "443"
ENDPOINT = "/plugins/RPC2/"
VERBOSE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def hide_password(url: str, start: int = 6) -> str:
    """Returns the URL with password part replaced with '*'."""
    start_pos = url.find(":", start) + 1
    end_pos = url.find("@")
    return f"{url[:start_pos]}{'*' * (end_pos - start_pos)}{url[end_pos:]}"


def upload_plugin(username: str, password: str, server: str, port: str, plugin_path: Path):
    """Uploads the plugin zip file to the server."""
    address = f"{PROTOCOL}://{username}:{password}@{server}:{port}{ENDPOINT}"
    logging.info("Connecting to: %s", hide_password(address))

    server_proxy = xmlrpc.client.ServerProxy(address, verbose=VERBOSE)

    try:
        with plugin_path.open("rb") as file_handle:
            plugin_id, version_id = server_proxy.plugin.upload(
                xmlrpc.client.Binary(file_handle.read())
            )
        logging.info("Plugin ID: %s", plugin_id)
        logging.info("Version ID: %s", version_id)
    except xmlrpc.client.ProtocolError as err:
        logging.error("A protocol error occurred")
        logging.error("URL: %s", hide_password(err.url))
        logging.error("HTTP/HTTPS headers: %s", err.headers)
        logging.error("Error code: %s", err.errcode)
        logging.error("Error message: %s", err.errmsg)
    except xmlrpc.client.Fault as err:
        logging.error("A fault occurred")
        logging.error("Fault code: %s", err.faultCode)
        logging.error("Fault string: %s", err.faultString)


def main():
    """Parse command line arguments and upload the plugin to QGIS repository."""
    parser = argparse.ArgumentParser(description="Upload a QGIS plugin to the repository.")
    parser.add_argument("plugin_zip", type=Path, help="Path to the plugin zip file")
    parser.add_argument("-u", "--username", help="Username for plugin site")
    parser.add_argument("-w", "--password", help="Password for plugin site")
    parser.add_argument("-s", "--server", default=SERVER, help=f"Server name (default: {SERVER})")
    parser.add_argument("-p", "--port", default=PORT, help=f"Server port (default: {PORT})")

    args = parser.parse_args()

    if not args.plugin_zip.exists():
        logging.error("File not found: %s", args.plugin_zip)
        sys.exit(1)

    username = args.username or input("Enter username: ").strip() or getpass.getuser()
    password = args.password or getpass.getpass("Enter password: ")

    upload_plugin(username, password, args.server, args.port, args.plugin_zip)


if __name__ == "__main__":
    main()
