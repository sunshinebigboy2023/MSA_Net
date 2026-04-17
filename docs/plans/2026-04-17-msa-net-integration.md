# MSA Net Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a logged-in web page in `MSA-Net` that lets users submit text, video, or both to the existing local MSA service for multimodal sentiment analysis.

**Architecture:** Keep MSA as an independent Python HTTP service. Add a Spring Boot proxy layer in `Net/user-center-backend-master` that accepts authenticated browser requests, stores uploaded videos, forwards JSON to MSA, and exposes task/result polling endpoints. Replace the current frontend welcome page with an Ant Design Pro analysis workspace.

**Tech Stack:** Java 8, Spring Boot 2.6, JUnit 5, MockMvc, Umi 3, React 17, TypeScript, Ant Design 4, existing `umi-request` wrapper.

---

### Task 1: Backend DTOs And Configuration

**Files:**
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/config/MsaProperties.java`
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/request/AnalysisSubmitRequest.java`
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/analysis/AnalysisTaskResponse.java`
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/analysis/AnalysisResultResponse.java`
- Modify: `Net/user-center-backend-master/src/main/resources/application.yml`
- Test: `Net/user-center-backend-master/src/test/java/com/yupi/usercenter/config/MsaPropertiesTest.java`

**Step 1: Write the failing test**

Create `MsaPropertiesTest`:

```java
package com.yupi.usercenter.config;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class MsaPropertiesTest {
    @Test
    void defaultsPointToLocalMsaAndRuntimeUploads() {
        MsaProperties properties = new MsaProperties();

        Assertions.assertEquals("http://127.0.0.1:8000", properties.getBaseUrl());
        Assertions.assertEquals("runtime/uploads", properties.getUploadDir());
    }
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
cd Net/user-center-backend-master
.\mvnw.cmd -Dtest=MsaPropertiesTest test
```

Expected: FAIL because `MsaProperties` does not exist.

**Step 3: Write minimal implementation**

Create `MsaProperties`:

```java
package com.yupi.usercenter.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "msa")
public class MsaProperties {
    private String baseUrl = "http://127.0.0.1:8000";
    private String uploadDir = "runtime/uploads";
}
```

Create request DTO:

```java
package com.yupi.usercenter.model.domain.request;

import lombok.Data;
import java.io.Serializable;

@Data
public class AnalysisSubmitRequest implements Serializable {
    private String text;
    private String language;
}
```

Create task DTO:

```java
package com.yupi.usercenter.model.domain.analysis;

import lombok.Data;
import java.io.Serializable;

@Data
public class AnalysisTaskResponse implements Serializable {
    private String taskId;
    private String status;
    private String error;
}
```

Create result DTO:

```java
package com.yupi.usercenter.model.domain.analysis;

import lombok.Data;
import java.io.Serializable;
import java.util.List;
import java.util.Map;

@Data
public class AnalysisResultResponse implements Serializable {
    private String taskId;
    private String status;
    private List<String> usedModalities;
    private List<String> missingModalities;
    private String emotionLabel;
    private String sentimentPolarity;
    private Double score;
    private Double confidence;
    private String message;
    private String error;
    private String transcript;
    private Map<String, String> featureStatus;
    private String language;
    private String modelDataset;
    private String modelCondition;
    private Map<String, Object> rawInputs;
    private Long processingTimeMs;
}
```

Add to `application.yml`:

```yaml
msa:
  base-url: http://127.0.0.1:8000
  upload-dir: runtime/uploads
spring:
  servlet:
    multipart:
      max-file-size: 500MB
      max-request-size: 500MB
```

Preserve the existing datasource/session settings when inserting YAML.

**Step 4: Run test to verify it passes**

Run:

```powershell
.\mvnw.cmd -Dtest=MsaPropertiesTest test
```

Expected: PASS.

**Step 5: Commit**

Stage exact files only:

```powershell
git add Net/user-center-backend-master/src/main/java/com/yupi/usercenter/config/MsaProperties.java Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/request/AnalysisSubmitRequest.java Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/analysis/AnalysisTaskResponse.java Net/user-center-backend-master/src/main/java/com/yupi/usercenter/model/domain/analysis/AnalysisResultResponse.java Net/user-center-backend-master/src/main/resources/application.yml Net/user-center-backend-master/src/test/java/com/yupi/usercenter/config/MsaPropertiesTest.java
git commit -m "feat: add MSA analysis backend DTOs"
```

---

### Task 2: MSA HTTP Client

**Files:**
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/MsaClient.java`
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/HttpMsaClient.java`
- Test: `Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/HttpMsaClientTest.java`

**Step 1: Write the failing test**

Use a small local HTTP server to avoid calling real MSA:

```java
package com.yupi.usercenter.service.impl;

import com.sun.net.httpserver.HttpServer;
import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.Map;

class HttpMsaClientTest {
    private HttpServer server;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void submitAnalysisPostsJsonToMsaAnalyze() throws Exception {
        final String[] body = new String[1];
        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/analyze", exchange -> {
            body[0] = new String(exchange.getRequestBody().readAllBytes(), "UTF-8");
            byte[] response = "{\"taskId\":\"task-1\",\"status\":\"PENDING\"}".getBytes("UTF-8");
            exchange.sendResponseHeaders(202, response.length);
            try (OutputStream output = exchange.getResponseBody()) {
                output.write(response);
            }
        });
        server.start();

        MsaProperties properties = new MsaProperties();
        properties.setBaseUrl("http://127.0.0.1:" + server.getAddress().getPort());
        HttpMsaClient client = new HttpMsaClient(properties);

        Map<String, Object> payload = new HashMap<>();
        payload.put("text", "我很高兴");
        payload.put("language", "zh");
        AnalysisTaskResponse result = client.submitAnalysis(payload);

        Assertions.assertEquals("task-1", result.getTaskId());
        Assertions.assertTrue(body[0].contains("\"text\":\"我很高兴\""));
        Assertions.assertTrue(body[0].contains("\"language\":\"zh\""));
    }
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
.\mvnw.cmd -Dtest=HttpMsaClientTest test
```

Expected: FAIL because `HttpMsaClient` and `MsaClient` do not exist.

**Step 3: Write minimal implementation**

Create `MsaClient`:

```java
package com.yupi.usercenter.service;

import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import java.util.Map;

public interface MsaClient {
    AnalysisTaskResponse submitAnalysis(Map<String, Object> payload);
    AnalysisTaskResponse getTask(String taskId);
    AnalysisResultResponse getResult(String taskId);
}
```

Create `HttpMsaClient` with `RestTemplate`, Jackson JSON serialization, and these endpoints:

```java
@Service
public class HttpMsaClient implements MsaClient {
    private final MsaProperties properties;
    private final RestTemplate restTemplate = new RestTemplate();

    public HttpMsaClient(MsaProperties properties) {
        this.properties = properties;
    }

    @Override
    public AnalysisTaskResponse submitAnalysis(Map<String, Object> payload) {
        return restTemplate.postForObject(properties.getBaseUrl() + "/analyze", payload, AnalysisTaskResponse.class);
    }

    @Override
    public AnalysisTaskResponse getTask(String taskId) {
        return restTemplate.getForObject(properties.getBaseUrl() + "/task/" + taskId, AnalysisTaskResponse.class);
    }

    @Override
    public AnalysisResultResponse getResult(String taskId) {
        return restTemplate.getForObject(properties.getBaseUrl() + "/result/" + taskId, AnalysisResultResponse.class);
    }
}
```

Add null checks and wrap client exceptions in `BusinessException(ErrorCode.SYSTEM_ERROR, "MSA 服务不可用，请确认 Python 服务已启动")`.

**Step 4: Run test to verify it passes**

Run:

```powershell
.\mvnw.cmd -Dtest=HttpMsaClientTest test
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/MsaClient.java Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/HttpMsaClient.java Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/HttpMsaClientTest.java
git commit -m "feat: add MSA HTTP client"
```

---

### Task 3: Backend Analysis Service

**Files:**
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/AnalysisService.java`
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/AnalysisServiceImpl.java`
- Test: `Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/AnalysisServiceImplTest.java`

**Step 1: Write the failing test**

Test text-only payload and empty validation:

```java
package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.exception.BusinessException;
import com.yupi.usercenter.model.domain.User;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import com.yupi.usercenter.service.MsaClient;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;

import java.util.Map;

class AnalysisServiceImplTest {
    @Test
    void submitTextOnlyForwardsTextAndLanguage() {
        CapturingClient client = new CapturingClient();
        AnalysisServiceImpl service = new AnalysisServiceImpl(client, new MsaProperties());
        User user = new User();
        user.setId(7L);

        AnalysisTaskResponse response = service.submit("我很高兴", "zh", null, user);

        Assertions.assertEquals("task-1", response.getTaskId());
        Assertions.assertEquals("我很高兴", client.payload.get("text"));
        Assertions.assertEquals("zh", client.payload.get("language"));
    }

    @Test
    void submitWithoutTextOrVideoFails() {
        AnalysisServiceImpl service = new AnalysisServiceImpl(new CapturingClient(), new MsaProperties());
        User user = new User();
        user.setId(7L);

        Assertions.assertThrows(BusinessException.class, () -> service.submit(" ", "auto", null, user));
    }

    private static class CapturingClient implements MsaClient {
        Map<String, Object> payload;

        @Override
        public AnalysisTaskResponse submitAnalysis(Map<String, Object> payload) {
            this.payload = payload;
            AnalysisTaskResponse response = new AnalysisTaskResponse();
            response.setTaskId("task-1");
            response.setStatus("PENDING");
            return response;
        }

        @Override
        public AnalysisTaskResponse getTask(String taskId) {
            return null;
        }

        @Override
        public com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse getResult(String taskId) {
            return null;
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
.\mvnw.cmd -Dtest=AnalysisServiceImplTest test
```

Expected: FAIL because service classes do not exist.

**Step 3: Write minimal implementation**

Create `AnalysisService`:

```java
package com.yupi.usercenter.service;

import com.yupi.usercenter.model.domain.User;
import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import org.springframework.web.multipart.MultipartFile;

public interface AnalysisService {
    AnalysisTaskResponse submit(String text, String language, MultipartFile video, User currentUser);
    AnalysisTaskResponse getTask(String taskId);
    AnalysisResultResponse getResult(String taskId);
}
```

Create `AnalysisServiceImpl`:

```java
@Service
public class AnalysisServiceImpl implements AnalysisService {
    private final MsaClient msaClient;
    private final MsaProperties properties;

    public AnalysisServiceImpl(MsaClient msaClient, MsaProperties properties) {
        this.msaClient = msaClient;
        this.properties = properties;
    }

    @Override
    public AnalysisTaskResponse submit(String text, String language, MultipartFile video, User currentUser) {
        boolean hasText = StringUtils.isNotBlank(text);
        boolean hasVideo = video != null && !video.isEmpty();
        if (!hasText && !hasVideo) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR, "请至少输入文本或上传视频");
        }
        Map<String, Object> payload = new HashMap<>();
        if (hasText) {
            payload.put("text", text.trim());
        }
        String normalizedLanguage = normalizeLanguage(language);
        if (StringUtils.isNotBlank(normalizedLanguage)) {
            payload.put("language", normalizedLanguage);
        }
        if (hasVideo) {
            payload.put("videoFile", saveVideo(video, currentUser).toAbsolutePath().toString());
        }
        return msaClient.submitAnalysis(payload);
    }
}
```

Add helpers:

- `normalizeLanguage`: returns `zh`, `en`, or `null` for auto/blank.
- `saveVideo`: creates `uploadDir/userId`, preserves a safe extension from the original filename, writes the file with `MultipartFile.transferTo`.
- `getTask` and `getResult`: delegate to `msaClient`.

**Step 4: Run test to verify it passes**

Run:

```powershell
.\mvnw.cmd -Dtest=AnalysisServiceImplTest test
```

Expected: PASS.

**Step 5: Add video-save test**

Add:

```java
@Test
void submitVideoSavesFileAndForwardsAbsolutePath() {
    CapturingClient client = new CapturingClient();
    MsaProperties properties = new MsaProperties();
    properties.setUploadDir("target/test-uploads");
    AnalysisServiceImpl service = new AnalysisServiceImpl(client, properties);
    User user = new User();
    user.setId(9L);
    MockMultipartFile video = new MockMultipartFile("video", "sample.mp4", "video/mp4", new byte[]{1, 2, 3});

    service.submit("", "en", video, user);

    String videoFile = String.valueOf(client.payload.get("videoFile"));
    Assertions.assertTrue(videoFile.endsWith(".mp4"));
    Assertions.assertTrue(new java.io.File(videoFile).exists());
}
```

Run it, verify RED if save helper missing, implement, then verify GREEN.

**Step 6: Commit**

```powershell
git add Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/AnalysisService.java Net/user-center-backend-master/src/main/java/com/yupi/usercenter/service/impl/AnalysisServiceImpl.java Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/AnalysisServiceImplTest.java
git commit -m "feat: add analysis submission service"
```

---

### Task 4: Backend Controller

**Files:**
- Create: `Net/user-center-backend-master/src/main/java/com/yupi/usercenter/controller/AnalysisController.java`
- Test: `Net/user-center-backend-master/src/test/java/com/yupi/usercenter/controller/AnalysisControllerTest.java`

**Step 1: Write the failing test**

Use `@WebMvcTest(AnalysisController.class)`:

```java
package com.yupi.usercenter.controller;

import com.yupi.usercenter.model.domain.User;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import com.yupi.usercenter.service.AnalysisService;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mock.web.MockHttpSession;
import org.springframework.test.web.servlet.MockMvc;

import static com.yupi.usercenter.contant.UserConstant.USER_LOGIN_STATE;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(AnalysisController.class)
class AnalysisControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private AnalysisService analysisService;

    @Test
    void analyzeRequiresLoginAndReturnsTask() throws Exception {
        User user = new User();
        user.setId(1L);
        MockHttpSession session = new MockHttpSession();
        session.setAttribute(USER_LOGIN_STATE, user);
        AnalysisTaskResponse response = new AnalysisTaskResponse();
        response.setTaskId("task-1");
        response.setStatus("PENDING");
        Mockito.when(analysisService.submit(Mockito.eq("hello"), Mockito.eq("en"), Mockito.isNull(), Mockito.eq(user)))
                .thenReturn(response);

        mockMvc.perform(multipart("/analysis/analyze")
                        .param("text", "hello")
                        .param("language", "en")
                        .session(session))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value(0))
                .andExpect(jsonPath("$.data.taskId").value("task-1"));
    }
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
.\mvnw.cmd -Dtest=AnalysisControllerTest test
```

Expected: FAIL because controller does not exist.

**Step 3: Write minimal implementation**

Create `AnalysisController`:

```java
package com.yupi.usercenter.controller;

@RestController
@RequestMapping("/analysis")
public class AnalysisController {
    @Resource
    private AnalysisService analysisService;

    @PostMapping("/analyze")
    public BaseResponse<AnalysisTaskResponse> analyze(
            @RequestParam(value = "text", required = false) String text,
            @RequestParam(value = "language", required = false) String language,
            @RequestPart(value = "video", required = false) MultipartFile video,
            HttpServletRequest request) {
        User currentUser = requireLogin(request);
        return ResultUtils.success(analysisService.submit(text, language, video, currentUser));
    }

    @GetMapping("/task/{taskId}")
    public BaseResponse<AnalysisTaskResponse> getTask(@PathVariable String taskId, HttpServletRequest request) {
        requireLogin(request);
        return ResultUtils.success(analysisService.getTask(taskId));
    }

    @GetMapping("/result/{taskId}")
    public BaseResponse<AnalysisResultResponse> getResult(@PathVariable String taskId, HttpServletRequest request) {
        requireLogin(request);
        return ResultUtils.success(analysisService.getResult(taskId));
    }

    private User requireLogin(HttpServletRequest request) {
        Object userObj = request.getSession().getAttribute(USER_LOGIN_STATE);
        if (!(userObj instanceof User)) {
            throw new BusinessException(ErrorCode.NOT_LOGIN);
        }
        return (User) userObj;
    }
}
```

**Step 4: Run test to verify it passes**

Run:

```powershell
.\mvnw.cmd -Dtest=AnalysisControllerTest test
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add Net/user-center-backend-master/src/main/java/com/yupi/usercenter/controller/AnalysisController.java Net/user-center-backend-master/src/test/java/com/yupi/usercenter/controller/AnalysisControllerTest.java
git commit -m "feat: expose analysis backend API"
```

---

### Task 5: Frontend API Types And Request Functions

**Files:**
- Modify: `Net/user-center-frontend-master/src/services/ant-design-pro/typings.d.ts`
- Modify: `Net/user-center-frontend-master/src/services/ant-design-pro/api.ts`

**Step 1: Write the failing test or type check target**

Because this project mainly uses generated service files, use TypeScript as the safety check. Add intended calls in `api.ts` first as test target comments are not enough, so create types before implementation is complete and run:

```powershell
cd Net/user-center-frontend-master
npm run tsc
```

Expected before implementation: TypeScript fails when `submitAnalysis`, `getAnalysisTask`, and `getAnalysisResult` are imported by the page in Task 6.

**Step 2: Implement API types**

Add to `typings.d.ts`:

```ts
type AnalysisTask = {
  taskId: string;
  status: string;
  error?: string;
};

type AnalysisResult = {
  taskId: string;
  status?: string;
  usedModalities?: string[];
  missingModalities?: string[];
  emotionLabel?: string;
  sentimentPolarity?: string;
  score?: number;
  confidence?: number;
  message?: string;
  error?: string;
  transcript?: string;
  featureStatus?: Record<string, string>;
  language?: string;
  modelDataset?: string;
  modelCondition?: string;
  rawInputs?: Record<string, any>;
  processingTimeMs?: number;
};
```

Add to `api.ts`:

```ts
export async function submitAnalysis(body: FormData, options?: { [key: string]: any }) {
  return request<API.AnalysisTask>('/api/analysis/analyze', {
    method: 'POST',
    data: body,
    ...(options || {}),
  });
}

export async function getAnalysisTask(taskId: string, options?: { [key: string]: any }) {
  return request<API.AnalysisTask>(`/api/analysis/task/${taskId}`, {
    method: 'GET',
    ...(options || {}),
  });
}

export async function getAnalysisResult(taskId: string, options?: { [key: string]: any }) {
  return request<API.AnalysisResult>(`/api/analysis/result/${taskId}`, {
    method: 'GET',
    ...(options || {}),
  });
}
```

Do not set `Content-Type` manually for `FormData`; the browser must provide the boundary.

**Step 3: Commit with frontend page task**

Commit this together with Task 6 after the type check has a real consumer.

---

### Task 6: Frontend Analysis Workspace

**Files:**
- Replace: `Net/user-center-frontend-master/src/pages/Welcome.tsx`
- Replace: `Net/user-center-frontend-master/src/pages/Welcome.less`
- Modify: `Net/user-center-frontend-master/config/routes.ts`
- Modify: `Net/user-center-frontend-master/config/defaultSettings.ts` if product name still says user center

**Step 1: Write the failing check**

After importing the new API functions in `Welcome.tsx`, run:

```powershell
cd Net/user-center-frontend-master
npm run tsc
```

Expected: FAIL until API functions/types from Task 5 exist.

**Step 2: Implement page behavior**

Replace the old welcome content with:

- `Input.TextArea` for `text`.
- `Upload.Dragger` with `beforeUpload={() => false}` for one local video file.
- `Segmented` or `Radio.Group` for `auto`, `zh`, `en`.
- Primary submit button.
- Status timeline/progress.
- Result section.

Core submit logic:

```ts
const handleAnalyze = async () => {
  if (!text.trim() && !videoFile) {
    message.warning('请输入文本或上传视频');
    return;
  }
  const formData = new FormData();
  if (text.trim()) {
    formData.append('text', text.trim());
  }
  if (language !== 'auto') {
    formData.append('language', language);
  }
  if (videoFile) {
    formData.append('video', videoFile as RcFile);
  }

  setSubmitting(true);
  setResult(undefined);
  const task = await submitAnalysis(formData);
  setTask(task);
  pollTask(task.taskId);
};
```

Polling:

```ts
const pollTask = async (taskId: string) => {
  const timer = window.setInterval(async () => {
    const current = await getAnalysisTask(taskId);
    setTask(current);
    if (current.status === 'SUCCESS') {
      window.clearInterval(timer);
      setResult(await getAnalysisResult(taskId));
      setSubmitting(false);
    }
    if (current.status === 'FAILED') {
      window.clearInterval(timer);
      setSubmitting(false);
      message.error(current.error || '分析失败');
    }
  }, 1500);
};
```

Render result fields:

- `sentimentPolarity`
- `score`
- `confidence`
- `usedModalities`
- `language`
- `modelDataset`
- `modelCondition`
- `transcript`
- `processingTimeMs`

**Step 3: Implement visual design**

Use a restrained, professional analysis dashboard:

- Full-width page band with a two-column responsive layout.
- Left side: input controls.
- Right side: status and result.
- Use cards only for the form panel and result panel.
- Avoid nested cards.
- Keep text Chinese-facing and concise.
- Use icons for upload/analyze/status actions where Ant Design provides them.

**Step 4: Run type check**

Run:

```powershell
npm run tsc
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add Net/user-center-frontend-master/src/services/ant-design-pro/typings.d.ts Net/user-center-frontend-master/src/services/ant-design-pro/api.ts Net/user-center-frontend-master/src/pages/Welcome.tsx Net/user-center-frontend-master/src/pages/Welcome.less Net/user-center-frontend-master/config/routes.ts Net/user-center-frontend-master/config/defaultSettings.ts
git commit -m "feat: add multimodal analysis workspace"
```

---

### Task 7: End-To-End Local Verification

**Files:**
- Modify only if verification finds a bug covered by a failing test first.

**Step 1: Start MSA**

Run from `MSA`:

```powershell
cd MSA
python -m msa_service.controller.http_server --host 127.0.0.1 --port 8000
```

Expected: `Serving on http://127.0.0.1:8000`.

**Step 2: Start backend**

Run:

```powershell
cd Net/user-center-backend-master
.\mvnw.cmd spring-boot:run
```

Expected: backend starts on `http://localhost:8080/api`.

**Step 3: Start frontend**

Run:

```powershell
cd Net/user-center-frontend-master
npm run start:dev
```

Expected: Umi dev server starts, usually `http://localhost:8000` or another available port.

**Step 4: Manual smoke test**

Use the browser:

- Register or login.
- Open `/welcome`.
- Submit text only: `我很高兴`.
- Submit video only with language set to Chinese.
- Submit text + video.

Expected: page shows task progress and final MSA result.

**Step 5: Fix any bug with TDD**

For each bug:

- Write a failing backend or frontend test/check first.
- Verify RED.
- Implement minimal fix.
- Verify GREEN.
- Commit exact files.

---

### Task 8: Upload To MSA-Net

**Files:**
- Include only integration code, docs, and tests from this plan.
- Do not include `MSA/dataset`, `MSA/models`, `MSA/tools`, `MSA/temp`, `MSA/outputs`, uploaded videos, or runtime files.
- Do not include unrelated existing deletions under old `MoMKE/` or `user-center-*` root paths unless the user explicitly asks.

**Step 1: Inspect status**

Run:

```powershell
git status --short --branch
```

Expected: only planned files should be staged or ready to stage. Unrelated dirty files may exist; leave them untouched.

**Step 2: Run final checks**

Run:

```powershell
cd Net/user-center-backend-master
.\mvnw.cmd test
cd ..\user-center-frontend-master
npm run tsc
```

Expected: both pass.

**Step 3: Push to MSA-Net**

Run from repo root:

```powershell
git push origin main
```

Target remote:

```text
https://github.com/sunshinebigboy2023/MSA_Net.git
```

Expected: `main -> main`.
