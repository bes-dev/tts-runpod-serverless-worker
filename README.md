# Text-To-Speech Serverless RunPod Worker

## RunPod Endpoint

This repository contains the worker for the xTTSv2 AI Endpoints.

## Docker Image

```bash
docker build .
```
 or

 ```bash
 docker pull devbes/tts-runpod-serverless-worker:latest
 ```

## Continuous Deployment
This worker follows a modified version of the [worker template](https://github.com/runpod-workers/worker-template) where the Docker build workflow contains additional SD models to be built and pushed.

## API

```json
{
  "input": {
      "language": <language:str>,
      "voice": <url_of_voice_sample:str>,
      "text": <text>:str,
      "gpt_cond_len": <gpt_cond_len:int>,
      "max_ref_len": <max_ref_len:int>,
      "speed": <speed:float>
      "enhance_audio": <enhance_audio:bool>
  }
}
```
