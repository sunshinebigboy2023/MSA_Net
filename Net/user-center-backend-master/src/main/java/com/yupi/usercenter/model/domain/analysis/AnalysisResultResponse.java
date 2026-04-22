package com.yupi.usercenter.model.domain.analysis;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
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

    private String textSource;

    private Map<String, String> featureStatus;

    private List<String> warnings;

    private String language;

    private String modelDataset;

    private String modelCondition;

    private Map<String, Object> rawInputs;

    private Long processingTimeMs;
}
