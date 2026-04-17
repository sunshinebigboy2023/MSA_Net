package com.yupi.usercenter.model.domain.analysis;

import lombok.Data;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

@Data
public class AnalysisResultResponse implements Serializable {

    private static final long serialVersionUID = 1L;

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
