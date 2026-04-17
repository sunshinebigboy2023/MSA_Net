# 多模态情感分析服务后端

Spring Boot backend for MSA-Net. It keeps user login and registration, saves uploaded videos, and forwards analysis jobs to the Python MSA service.

## APIs

- `POST /api/user/register`
- `POST /api/user/login`
- `GET /api/user/current`
- `POST /api/analysis/analyze`
- `GET /api/analysis/task/{taskId}`
- `GET /api/analysis/result/{taskId}`

## Local Config

Main configuration lives in `src/main/resources/application.yml`.

- Backend port: `8080`
- API context path: `/api`
- MSA service URL: `http://127.0.0.1:8000`
- Uploaded videos: `runtime/uploads`

## Tests

```powershell
.\mvnw.cmd test "-Dtest=AnalysisControllerTest,AnalysisServiceImplTest,HttpMsaClientTest,MsaPropertiesTest"
```
