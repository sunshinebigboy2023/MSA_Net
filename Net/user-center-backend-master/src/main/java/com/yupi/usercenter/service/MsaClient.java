package com.yupi.usercenter.service;

import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;

import java.util.Map;

public interface MsaClient {

    AnalysisTaskResponse submitAnalysis(Map<String, Object> payload);

    AnalysisTaskResponse getTask(String taskId);

    AnalysisResultResponse getResult(String taskId);
}
