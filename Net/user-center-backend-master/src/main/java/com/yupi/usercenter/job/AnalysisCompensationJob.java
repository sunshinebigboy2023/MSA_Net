package com.yupi.usercenter.job;

import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.model.domain.analysis.AnalysisTask;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskMessage;
import com.yupi.usercenter.service.impl.AnalysisQueueProducer;
import com.yupi.usercenter.service.impl.AnalysisTaskService;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Date;
import java.util.List;

@Component
public class AnalysisCompensationJob {

    private final AnalysisTaskService taskService;

    private final AnalysisQueueProducer queueProducer;

    private final MsaProperties properties;

    public AnalysisCompensationJob(AnalysisTaskService taskService, AnalysisQueueProducer queueProducer, MsaProperties properties) {
        this.taskService = taskService;
        this.queueProducer = queueProducer;
        this.properties = properties;
    }

    @Scheduled(fixedDelay = 60000)
    public void requeueStaleTasks() {
        long staleMillis = properties.getStaleTaskSeconds() * 1000L;
        Date before = new Date(System.currentTimeMillis() - staleMillis);
        List<AnalysisTask> staleTasks = taskService.listStaleTasks(before, 50);
        for (AnalysisTask task : staleTasks) {
            int retryCount = task.getRetryCount() == null ? 0 : task.getRetryCount();
            if (retryCount >= properties.getMaxRetryCount()) {
                taskService.markFailed(task.getTaskId(), AnalysisTask.STATUS_DEAD_LETTER, "Task exceeded retry budget");
                continue;
            }
            taskService.incrementRetry(task.getTaskId());
            queueProducer.publishRetry(new AnalysisTaskMessage(
                    task.getTaskId(),
                    task.getUserId(),
                    taskService.payloadAsMap(task),
                    retryCount + 1));
        }
    }
}
