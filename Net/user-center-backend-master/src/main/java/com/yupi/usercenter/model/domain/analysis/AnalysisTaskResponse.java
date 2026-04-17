package com.yupi.usercenter.model.domain.analysis;

import lombok.Data;

import java.io.Serializable;

@Data
public class AnalysisTaskResponse implements Serializable {

    private static final long serialVersionUID = 1L;

    private String taskId;

    private String status;

    private String error;
}
