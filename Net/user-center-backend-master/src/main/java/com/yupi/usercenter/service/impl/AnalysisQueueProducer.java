package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.config.RabbitMqConfig;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskMessage;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;

@Service
public class AnalysisQueueProducer {

    private final RabbitTemplate rabbitTemplate;

    public AnalysisQueueProducer(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = rabbitTemplate;
    }

    public void publish(AnalysisTaskMessage message) {
        rabbitTemplate.convertAndSend(
                RabbitMqConfig.ANALYSIS_EXCHANGE,
                RabbitMqConfig.ANALYSIS_ROUTING_KEY,
                message);
    }

    public void publishRetry(AnalysisTaskMessage message) {
        rabbitTemplate.convertAndSend(
                RabbitMqConfig.ANALYSIS_EXCHANGE,
                RabbitMqConfig.ANALYSIS_RETRY_ROUTING_KEY,
                message);
    }
}
