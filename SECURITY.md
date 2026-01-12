# Security Configuration Guide

## 🔐 Secure API Key Management

This repository has been configured for secure credential management. Follow these steps to set up your API keys safely.

## ⚙️ Configuration Setup

### Option 1: JSON Configuration File (Recommended)

1. Copy the template:
   ```bash
   cp config/credentials_template.json config/credentials.json
   ```

2. Edit `config/credentials.json` with your actual keys:
   ```json
   {
       "openai": {
           "api_key": "sk-your-actual-openai-api-key"
       },
       "voyageai": {
           "api_key": "your-actual-voyageai-api-key"
       },
       "deepl": {
           "api_key": "your-actual-deepl-api-key"
       },
       "google_cloud": {
           "credentials_file": "path/to/your/google-credentials.json",
           "project_id": "your-google-cloud-project-id"
       }
   }
   ```

### Option 2: Environment Variables

1. Copy the template:
   ```bash
   cp config/.env_template .env
   ```

2. Edit `.env` with your actual keys:
   ```bash
   OPENAI_API_KEY=sk-your-actual-openai-api-key
   VOYAGE_API_KEY=your-actual-voyageai-api-key
   DEEPL_API_KEY=your-actual-deepl-api-key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
   ```

## 🛡️ Security Best Practices

1. **Never commit actual API keys** to git
2. **Use the config files** (`credentials.json` or `.env`) which are ignored by git
3. **Keep credentials files secure** with appropriate file permissions
4. **Rotate API keys regularly** especially if they might have been exposed
5. **Use environment variables in production** deployments

## 🔍 Verification

Run the security check script to verify your setup:
```bash
python setup_secure_config.py
```

This will:
- Create the config directory structure
- Check for old hardcoded credentials
- Provide setup guidance

## 📞 API Key Acquisition

### OpenAI API
- Sign up at: https://platform.openai.com/
- Generate API key in your dashboard

### VoyageAI API
- Sign up at: https://www.voyageai.com/
- Get your API key from the dashboard

### DeepL API
- Sign up at: https://www.deepl.com/pro-api
- Free tier available with usage limits

### Google Cloud Translation API
- Create project at: https://console.cloud.google.com/
- Enable Translation API
- Create service account and download JSON credentials

## 🚨 If You Have Issues

1. **Check file paths** in your configuration
2. **Verify API key format** (each service has different formats)
3. **Ensure config files exist** and are readable
4. **Check environment variables** if using that method
5. **Review error messages** for specific guidance

The configuration loader will provide detailed error messages to help you troubleshoot any issues.
   ```
   - Project ID

2. **DeepL API** (optional, for additional translation services)
   - API key from DeepL

## Setup Instructions

### Option 1: Using config/credentials.json (Recommended)

1. Copy the template file:
   ```bash
   cp config/credentials_template.json config/credentials.json
   ```

2. Edit `config/credentials.json` and fill in your API keys:
   ```json
   {
       "deepl": {
           "api_key": "your_actual_deepl_api_key_here"
       },
       "google_cloud": {
           "credentials_file": "path/to/your/google_service_account.json",
           "project_id": "your_google_cloud_project_id"
       }
   }
   ```

3. Place your Google Cloud service account JSON file in a secure location and update the path in the config.

### Option 2: Using Environment Variables

1. Copy the environment template:
   ```bash
   cp config/.env_template .env
   ```

2. Edit `.env` and fill in your values:
   ```env
   DEEPL_API_KEY=your_actual_deepl_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google_service_account.json
   GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
   ```
