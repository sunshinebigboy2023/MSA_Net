package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.model.domain.User;
import com.yupi.usercenter.model.domain.analysis.AnalysisTask;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskMessage;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import java.util.Map;

class AnalysisServiceImplAsyncTest {

    @Test
    void submitUsesDurableQueueWhenAsyncServicesArePresent() {
        AnalysisTaskService taskService = Mockito.mock(AnalysisTaskService.class);
        AnalysisQueueProducer queueProducer = Mockito.mock(AnalysisQueueProducer.class);
        AnalysisRateLimitService rateLimitService = Mockito.mock(AnalysisRateLimitService.class);
        AnalysisCacheService cacheService = Mockito.mock(AnalysisCacheService.class);
        MsaProperties properties = new MsaProperties();
        AnalysisServiceImpl service = new AnalysisServiceImpl(
                null,
                properties,
                taskService,
                queueProducer,
                rateLimitService,
                cacheService);
        User user = new User();
        user.setId(7L);
        AnalysisTask task = new AnalysisTask();
        task.setTaskId("task-1");
        task.setStatus(AnalysisTask.STATUS_QUEUED);
        Mockito.when(rateLimitService.allowSubmit(7L)).thenReturn(true);
        Mockito.when(taskService.createQueuedTask(Mockito.eq(7L), Mockito.anyMap())).thenReturn(task);

        AnalysisTaskResponse response = service.submit("happy", "en", false, null, user);

        Assertions.assertEquals("task-1", response.getTaskId());
        Assertions.assertEquals(AnalysisTask.STATUS_QUEUED, response.getStatus());
        ArgumentCaptor<AnalysisTaskMessage> messageCaptor = ArgumentCaptor.forClass(AnalysisTaskMessage.class);
        Mockito.verify(queueProducer).publish(messageCaptor.capture());
        Assertions.assertEquals("task-1", messageCaptor.getValue().getTaskId());
        Assertions.assertEquals(7L, messageCaptor.getValue().getUserId());
        Assertions.assertEquals("happy", messageCaptor.getValue().getPayload().get("text"));
        Mockito.verify(cacheService).cacheTask(task);
    }
}
