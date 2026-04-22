package com.yupi.usercenter.service.impl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTask;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

@Service
public class AnalysisCacheService {

    private static final long TASK_CACHE_MINUTES = 30L;
    private static final long RESULT_CACHE_HOURS = 6L;

    private final StringRedisTemplate redisTemplate;

    private final ObjectMapper objectMapper;

    public AnalysisCacheService(StringRedisTemplate redisTemplate, ObjectMapper objectMapper) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = objectMapper;
    }

    public void cacheTask(AnalysisTask task) {
        if (task == null || task.getTaskId() == null) {
            return;
        }
        try {
            redisTemplate.opsForValue().set(taskKey(task.getTaskId()), objectMapper.writeValueAsString(toTaskResponse(task)), TASK_CACHE_MINUTES, TimeUnit.MINUTES);
        } catch (JsonProcessingException ignored) {
        }
    }

    public AnalysisTaskResponse getCachedTask(String taskId) {
        String value = redisTemplate.opsForValue().get(taskKey(taskId));
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.readValue(value, AnalysisTaskResponse.class);
        } catch (IOException e) {
            return null;
        }
    }

    public void cacheResult(String taskId, AnalysisResultResponse result) {
        if (taskId == null || result == null) {
            return;
        }
        try {
            redisTemplate.opsForValue().set(resultKey(taskId), objectMapper.writeValueAsString(result), RESULT_CACHE_HOURS, TimeUnit.HOURS);
        } catch (JsonProcessingException ignored) {
        }
    }

    public AnalysisResultResponse getCachedResult(String taskId) {
        String value = redisTemplate.opsForValue().get(resultKey(taskId));
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.readValue(value, AnalysisResultResponse.class);
        } catch (IOException e) {
            return null;
        }
    }

    public static AnalysisTaskResponse toTaskResponse(AnalysisTask task) {
        AnalysisTaskResponse response = new AnalysisTaskResponse();
        response.setTaskId(task.getTaskId());
        response.setStatus(task.getStatus());
        response.setError(task.getError());
        return response;
    }

    private static String taskKey(String taskId) {
        return "msa:task:" + taskId;
    }

    private static String resultKey(String taskId) {
        return "msa:result:" + taskId;
    }
}
