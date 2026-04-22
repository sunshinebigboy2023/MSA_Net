package com.yupi.usercenter.model.domain.analysis;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serializable;
import java.util.Date;

@Data
@TableName("analysis_task")
public class AnalysisTask implements Serializable {

    private static final long serialVersionUID = 1L;

    public static final String STATUS_QUEUED = "QUEUED";
    public static final String STATUS_RUNNING = "RUNNING";
    public static final String STATUS_SUCCESS = "SUCCESS";
    public static final String STATUS_FAILED = "FAILED";
    public static final String STATUS_RETRYING = "RETRYING";
    public static final String STATUS_DEAD_LETTER = "DEAD_LETTER";

    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskId;

    private Long userId;

    private String status;

    private String payload;

    private String result;

    private String error;

    private Integer retryCount;

    private Long processingTimeMs;

    private Date createTime;

    private Date updateTime;
}
