package com.yupi.usercenter.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yupi.usercenter.common.ErrorCode;
import com.yupi.usercenter.exception.BusinessException;
import com.yupi.usercenter.mapper.AnalysisTaskMapper;
import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTask;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import com.yupi.usercenter.model.domain.request.AnalysisCallbackRequest;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class AnalysisTaskService {

    private final AnalysisTaskMapper taskMapper;

    private final AnalysisCacheService cacheService;

    private final ObjectMapper objectMapper;

    public AnalysisTaskService(AnalysisTaskMapper taskMapper, AnalysisCacheService cacheService, ObjectMapper objectMapper) {
        this.taskMapper = taskMapper;
        this.cacheService = cacheService;
        this.objectMapper = objectMapper;
    }

    public AnalysisTask createQueuedTask(Long userId, Map<String, Object> payload) {
        AnalysisTask task = new AnalysisTask();
        task.setTaskId(UUID.randomUUID().toString());
        task.setUserId(userId == null ? 0L : userId);
        task.setStatus(AnalysisTask.STATUS_QUEUED);
        task.setPayload(toJson(payload));
        task.setRetryCount(0);
        task.setCreateTime(new Date());
        task.setUpdateTime(new Date());
        taskMapper.insert(task);
        return task;
    }

    public AnalysisTask getByTaskId(String taskId) {
        return taskMapper.selectOne(new QueryWrapper<AnalysisTask>().eq("taskId", taskId).last("limit 1"));
    }

    public AnalysisTaskResponse getTaskResponse(String taskId) {
        AnalysisTaskResponse cached = cacheService.getCachedTask(taskId);
        if (cached != null) {
            return cached;
        }
        AnalysisTask task = requireTask(taskId);
        AnalysisTaskResponse response = AnalysisCacheService.toTaskResponse(task);
        cacheService.cacheTask(task);
        return response;
    }

    public AnalysisResultResponse getResultResponse(String taskId) {
        AnalysisResultResponse cached = cacheService.getCachedResult(taskId);
        if (cached != null) {
            return cached;
        }
        AnalysisTask task = requireTask(taskId);
        if (task.getResult() == null) {
            AnalysisResultResponse response = new AnalysisResultResponse();
            response.setTaskId(task.getTaskId());
            response.setStatus(task.getStatus());
            response.setError(task.getError());
            return response;
        }
        try {
            AnalysisResultResponse response = objectMapper.readValue(task.getResult(), AnalysisResultResponse.class);
            cacheService.cacheResult(taskId, response);
            return response;
        } catch (IOException e) {
            throw new BusinessException(ErrorCode.SYSTEM_ERROR, "Analysis result is not valid JSON");
        }
    }

    public AnalysisTask completeFromCallback(AnalysisCallbackRequest request) {
        AnalysisTask task = requireTask(request.getTaskId());
        task.setStatus(request.getStatus());
        task.setError(request.getError());
        task.setProcessingTimeMs(request.getProcessingTimeMs());
        task.setUpdateTime(new Date());
        if (request.getResult() != null) {
            task.setResult(toJson(request.getResult()));
        }
        taskMapper.update(task, new QueryWrapper<AnalysisTask>().eq("taskId", task.getTaskId()));
        cacheService.cacheTask(task);
        if (AnalysisTask.STATUS_SUCCESS.equals(task.getStatus())) {
            cacheService.cacheResult(task.getTaskId(), mapToResult(task.getTaskId(), task.getStatus(), request.getResult(), request.getProcessingTimeMs()));
        }
        return task;
    }

    public void markRunning(String taskId) {
        AnalysisTask task = requireTask(taskId);
        task.setStatus(AnalysisTask.STATUS_RUNNING);
        task.setUpdateTime(new Date());
        taskMapper.update(task, new QueryWrapper<AnalysisTask>().eq("taskId", taskId));
        cacheService.cacheTask(task);
    }

    public void markFailed(String taskId, String status, String error) {
        AnalysisTask task = requireTask(taskId);
        task.setStatus(status);
        task.setError(error);
        task.setUpdateTime(new Date());
        taskMapper.update(task, new QueryWrapper<AnalysisTask>().eq("taskId", taskId));
        cacheService.cacheTask(task);
    }

    public void incrementRetry(String taskId) {
        AnalysisTask task = requireTask(taskId);
        int retryCount = task.getRetryCount() == null ? 0 : task.getRetryCount();
        task.setRetryCount(retryCount + 1);
        task.setStatus(AnalysisTask.STATUS_RETRYING);
        task.setUpdateTime(new Date());
        taskMapper.update(task, new QueryWrapper<AnalysisTask>().eq("taskId", taskId));
        cacheService.cacheTask(task);
    }

    public List<AnalysisTask> listStaleTasks(Date before, int limit) {
        return taskMapper.selectList(new QueryWrapper<AnalysisTask>()
                .in("status", Arrays.asList(AnalysisTask.STATUS_QUEUED, AnalysisTask.STATUS_RUNNING, AnalysisTask.STATUS_RETRYING))
                .lt("updateTime", before)
                .last("limit " + limit));
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> payloadAsMap(AnalysisTask task) {
        try {
            return objectMapper.readValue(task.getPayload(), Map.class);
        } catch (IOException e) {
            throw new BusinessException(ErrorCode.SYSTEM_ERROR, "Analysis task payload is not valid JSON");
        }
    }

    private AnalysisTask requireTask(String taskId) {
        AnalysisTask task = getByTaskId(taskId);
        if (task == null) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR, "Analysis task not found");
        }
        return task;
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new BusinessException(ErrorCode.SYSTEM_ERROR, "Failed to serialize analysis task");
        }
    }

    private AnalysisResultResponse mapToResult(String taskId, String status, Map<String, Object> result, Long processingTimeMs) {
        AnalysisResultResponse response = objectMapper.convertValue(result, AnalysisResultResponse.class);
        response.setTaskId(taskId);
        response.setStatus(status);
        response.setProcessingTimeMs(processingTimeMs);
        return response;
    }
}
