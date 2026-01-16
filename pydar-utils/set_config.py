
from __future__ import annotations

import logging
import os

import toml
import yaml

log = logging.getLogger()

def config_to_env(config: dict):
    for key, value in config.items():
        key = f"PDAR_{key.upper()}"
        coalalesced_val = os.environ.get(key, value)
        if value != coalalesced_val:
            log.info(f"Existing value for {key} found: {coalalesced_val}")
        os.environ[key] = coalalesced_val

def load_config(config_file: str, load_to_env: bool = True) -> dict:
    config = dict()
    try:
        with open(config_file) as f:
            if 'toml' in config_file:
                config = toml.load(f)
            elif 'yml' in config_file or 'yaml' in config_file:
                config = yaml.safe_load(f)
            log.info(f"Loaded config from {config_file}")
    except Exception as error:
        log.error(f"Error loading config {config_file}: {error}")
        log.error(f"Default values will be used")
    if load_to_env:
        config_to_env(config)
    return config
