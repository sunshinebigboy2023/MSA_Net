# MSA-Net

MSA-Net is a multimodal sentiment analysis web service. It combines:

- `MSA/`: Python multimodal sentiment analysis service
- `Net/user-center-backend-master/`: Spring Boot backend with login, registration and analysis APIs
- `Net/user-center-frontend-master/`: Web frontend for text/video sentiment analysis

## Start Locally

Use the helper script from the repository root:

```powershell
.\start-all.bat -Restart
```

Services:

- Frontend: `http://localhost:8001/`
- Backend: `http://localhost:8080/api`
- MSA service: `http://127.0.0.1:8000`

## Model And Data Files

Large local assets are intentionally not committed:

- `MSA/tools/`
- `MSA/models/`
- `MSA/dataset/`
- `MSA/outputs/`
- virtual environments, logs, build outputs and uploaded videos

Place required model checkpoints and external tools back under the ignored local directories before running full multimodal inference.

## Validation

Recent local checks:

```powershell
cd MSA
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service_scenarios tests.test_analysis_service_media

cd ..\Net\user-center-backend-master
.\mvnw.cmd test "-Dtest=AnalysisControllerTest,AnalysisServiceImplTest,HttpMsaClientTest,MsaPropertiesTest"

cd ..\user-center-frontend-master
npm.cmd run build
```
