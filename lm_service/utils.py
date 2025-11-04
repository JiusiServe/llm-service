# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
import regex as re


def is_addr_ipv6(addr: str) -> bool:
    """
    Check if the given address is an IPv6 address

    Args:
        addr (str): The address to check

    Returns:
        bool: True if the address is an IPv6 address, False otherwise
    """
    ipv6_pattern = r"^\[(.*?)\]:(\d+)$"
    return bool(re.match(ipv6_pattern, addr))
