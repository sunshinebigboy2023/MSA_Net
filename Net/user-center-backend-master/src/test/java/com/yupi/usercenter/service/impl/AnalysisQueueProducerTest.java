package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.config.RabbitMqConfig;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskMessage;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.amqp.rabbit.core.RabbitTemplate;

import java.util.HashMap;
import java.util.Map;

class AnalysisQueueProducerTest {

    @Test
    void publishSendsMessageToAnalysisExchange() {
        RabbitTemplate rabbitTemplate = Mockito.mock(RabbitTemplate.class);
        AnalysisQueueProducer producer = new AnalysisQueueProducer(rabbitTemplate);
        Map<String, Object> payload = new HashMap<>();
        payload.put("text", "happy");
        AnalysisTaskMessage message = new AnalysisTaskMessage("task-1", 7L, payload, 0);

        producer.publish(message);

        Mockito.verify(rabbitTemplate).convertAndSend(
                RabbitMqConfig.ANALYSIS_EXCHANGE,
                RabbitMqConfig.ANALYSIS_ROUTING_KEY,
                message);
    }
}
