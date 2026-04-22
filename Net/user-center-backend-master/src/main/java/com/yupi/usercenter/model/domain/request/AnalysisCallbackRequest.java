package com.yupi.usercenter.model.domain.request;

import lombok.Data;

import java.io.Serializable;
import java.util.Map;

@Data
public class AnalysisCallbackRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    private String taskId;

    private String status;

    private Map<String, Object> result;

    private String error;

    private Long processingTimeMs;
}
