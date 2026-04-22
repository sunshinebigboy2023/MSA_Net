# UI And Sentiment Accuracy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair and polish the login and analysis UI while improving practical sentiment judgment accuracy through transcript fusion, language routing visibility, and confidence warnings.

**Architecture:** Keep the current React 17, Umi, Ant Design Pro frontend and Spring Boot proxy API. The frontend submits `FormData` to `/api/analysis/analyze`; the backend forwards supported fields to the Python MSA service, which performs feature extraction, language routing, and inference.

**Tech Stack:** React, TypeScript, Ant Design Pro, Less, Spring Boot, Python unittest.

---

### Task 1: Verify Current Frontend Failure

**Files:**
- Read: `../Net/user-center-frontend-master/src/pages/Welcome.tsx`
- Read: `../Net/user-center-frontend-master/src/pages/user/Login/index.tsx`

**Step 1: Run the failing build**

Run:

```powershell
npm.cmd run build
```

from `../Net/user-center-frontend-master`.

Expected: FAIL because corrupted TSX has unterminated strings or malformed tags.

### Task 2: Repair And Polish Login UI

**Files:**
- Modify: `../Net/user-center-frontend-master/src/pages/user/Login/index.tsx`
- Modify: `../Net/user-center-frontend-master/src/pages/user/Login/index.less`

**Step 1: Implement the page**

Replace corrupted text with readable Chinese copy. Keep account login behavior. Add a brand panel, clean form title, short trust notes, and responsive styling.

**Step 2: Verify**

Run:

```powershell
npm.cmd run build
```

Expected: build should progress beyond login TSX parsing.

### Task 3: Repair And Polish Analysis UI

**Files:**
- Modify: `../Net/user-center-frontend-master/src/pages/Welcome.tsx`
- Modify: `../Net/user-center-frontend-master/src/pages/Welcome.less`
- Modify: `../Net/user-center-frontend-master/src/services/ant-design-pro/typings.d.ts`

**Step 1: Implement the page**

Replace corrupted text with readable Chinese copy. Add transcript enhancement switch, language segmented selection, upload state, progress timeline feel, result cards, warnings, transcript, feature status, dataset, condition, and processing time.

**Step 2: Verify**

Run:

```powershell
npm.cmd run build
```

Expected: PASS or expose the next concrete compile error.

### Task 4: Ensure Accuracy Flag Is Sent

**Files:**
- Modify: `../Net/user-center-frontend-master/src/pages/Welcome.tsx`
- Test: existing Spring test `../Net/user-center-backend-master/src/test/java/com/yupi/usercenter/service/impl/AnalysisServiceImplTest.java`
- Test: existing Python test `tests/test_analysis_service_scenarios.py`

**Step 1: Write or confirm failing behavior**

Confirm the frontend appends `enhanceTextWithTranscript=true` when text and video are present and the toggle is enabled.

**Step 2: Verify backend and MSA behavior**

Run:

```powershell
.\mvnw.cmd test "-Dtest=AnalysisServiceImplTest#submitTextAndVideoForwardsTranscriptEnhancementFlag"
```

from `../Net/user-center-backend-master`.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service_scenarios.AnalysisServiceScenarioTests.test_video_with_manual_text_can_fuse_asr_transcript
```

from `MSA`.

Expected: PASS.

### Task 5: Final Verification

**Files:**
- Verify all modified files.

**Step 1: Frontend build**

Run:

```powershell
npm.cmd run build
```

from `../Net/user-center-frontend-master`.

**Step 2: Backend focused tests**

Run:

```powershell
.\mvnw.cmd test "-Dtest=AnalysisControllerTest,AnalysisServiceImplTest,HttpMsaClientTest,MsaPropertiesTest"
```

from `../Net/user-center-backend-master`.

**Step 3: Python focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_service_scenarios tests.test_language_model_routing
```

from `MSA`.
