package com.yupi.usercenter.model.domain.analysis;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AnalysisTaskMessage implements Serializable {

    private static final long serialVersionUID = 1L;

    private String taskId;

    private Long userId;

    private Map<String, Object> payload;

    private Integer retryCount;
}
