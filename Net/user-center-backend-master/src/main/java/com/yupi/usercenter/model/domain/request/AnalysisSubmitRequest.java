package com.yupi.usercenter.model.domain.request;

import lombok.Data;

import java.io.Serializable;

@Data
public class AnalysisSubmitRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    private String text;

    private String language;

    private Boolean enhanceTextWithTranscript;
}
