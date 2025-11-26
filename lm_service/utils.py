# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import regex as re


def is_addr_ipv6(addr: str | None) -> bool:
    """
    Check if the given address is an IPv6 address

    Args:
        addr (str): The address to check

    Returns:
        bool: True if the address is an IPv6 address, False otherwise
    """
    if addr is None:
        return False
    # Support addresses with protocol prefix like "tcp://[::1]:8090"
    # Match protocol prefix (optional), IPv6 address in square brackets, and port (optional)
    ipv6_pattern = r"^(?:[a-zA-Z0-9]+://)?\[([0-9a-fA-F:]+)\](?::(\d+))?$"
    return bool(re.match(ipv6_pattern, addr))
