"""Errors raised when data breaks a contract."""


class DataContractError(Exception):
    """Input or output data does not meet its contract. The pipeline must stop."""
