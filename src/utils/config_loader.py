"""
Configuration utilities for loading API keys and credentials securely.

This module provides functions to load configuration from JSON files and environment variables
while keeping sensitive information out of the codebase.
"""

import os
import json
from pathlib import Path

def load_config():
    """
    Load configuration from credentials.json file or environment variables.
    
    Returns:
        dict: Configuration dictionary with API keys and credentials
    """
    config = {}
    
    # Try to load from config file first
    config_path = Path(__file__).parent.parent / 'config' / 'credentials.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
    
    # Fill in missing values from environment variables
    if 'openai' not in config:
        config['openai'] = {}
    if not config['openai'].get('api_key'):
        config['openai']['api_key'] = os.getenv('OPENAI_API_KEY')
    
    if 'voyageai' not in config:
        config['voyageai'] = {}
    if not config['voyageai'].get('api_key'):
        config['voyageai']['api_key'] = os.getenv('VOYAGE_API_KEY')
    
    if 'deepl' not in config:
        config['deepl'] = {}
    if not config['deepl'].get('api_key'):
        config['deepl']['api_key'] = os.getenv('DEEPL_API_KEY')
    
    if 'google_cloud' not in config:
        config['google_cloud'] = {}
    if not config['google_cloud'].get('credentials_file'):
        config['google_cloud']['credentials_file'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not config['google_cloud'].get('project_id'):
        config['google_cloud']['project_id'] = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    return config

def get_api_key(service):
    """
    Get API key for a specific service.
    
    Args:
        service (str): Service name ('openai', 'voyageai', 'deepl')
    
    Returns:
        str: API key for the service
        
    Raises:
        ValueError: If API key is not found
    """
    config = load_config()
    
    if service not in config or not config[service].get('api_key'):
        raise ValueError(f"{service.upper()} API key not found. Please set it in config/credentials.json or as environment variable.")
    
    return config[service]['api_key']

def get_google_credentials():
    """
    Get Google Cloud credentials information.
    
    Returns:
        dict: Dictionary with 'credentials_file' and 'project_id'
        
    Raises:
        ValueError: If credentials are not found
    """
    config = load_config()
    
    if 'google_cloud' not in config:
        raise ValueError("Google Cloud configuration not found.")
    
    credentials_file = config['google_cloud'].get('credentials_file')
    if not credentials_file or not os.path.exists(credentials_file):
        raise ValueError("Google Cloud credentials file not found. Please set GOOGLE_APPLICATION_CREDENTIALS.")
    
    return {
        'credentials_file': credentials_file,
        'project_id': config['google_cloud'].get('project_id')
    }

def setup_google_credentials():
    """
    Set up Google Cloud credentials environment variable.
    """
    try:
        credentials = get_google_credentials()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials['credentials_file']
        if credentials['project_id']:
            os.environ['GOOGLE_CLOUD_PROJECT'] = credentials['project_id']
    except ValueError as e:
        print(f"Warning: Could not set up Google Cloud credentials: {e}")