package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.common.ErrorCode;
import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.exception.BusinessException;
import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import com.yupi.usercenter.service.MsaClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
public class HttpMsaClient implements MsaClient {

    private final MsaProperties properties;

    private final RestTemplate restTemplate;

    @Autowired
    public HttpMsaClient(MsaProperties properties) {
        this(properties, new RestTemplate());
    }

    HttpMsaClient(MsaProperties properties, RestTemplate restTemplate) {
        this.properties = properties;
        this.restTemplate = restTemplate;
    }

    @Override
    public AnalysisTaskResponse submitAnalysis(Map<String, Object> payload) {
        return request(() -> restTemplate.postForObject(url("/analyze"), payload, AnalysisTaskResponse.class));
    }

    @Override
    public AnalysisTaskResponse getTask(String taskId) {
        return request(() -> restTemplate.getForObject(url("/task/" + taskId), AnalysisTaskResponse.class));
    }

    @Override
    public AnalysisResultResponse getResult(String taskId) {
        return request(() -> restTemplate.getForObject(url("/result/" + taskId), AnalysisResultResponse.class));
    }

    private String url(String path) {
        String baseUrl = properties.getBaseUrl();
        if (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        return baseUrl + path;
    }

    private <T> T request(MsaRequest<T> request) {
        try {
            T response = request.execute();
            if (response == null) {
                throw new BusinessException(ErrorCode.SYSTEM_ERROR, "MSA service returned empty response");
            }
            return response;
        } catch (BusinessException e) {
            throw e;
        } catch (RestClientException e) {
            throw new BusinessException(ErrorCode.SYSTEM_ERROR, "MSA service unavailable. Please start the Python service.");
        }
    }

    private interface MsaRequest<T> {
        T execute();
    }
}
