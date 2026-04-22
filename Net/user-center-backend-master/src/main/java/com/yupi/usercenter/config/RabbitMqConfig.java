package com.yupi.usercenter.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.QueueBuilder;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitMqConfig {

    public static final String ANALYSIS_EXCHANGE = "msa.analysis.exchange";
    public static final String ANALYSIS_QUEUE = "msa.analysis.queue";
    public static final String ANALYSIS_RETRY_QUEUE = "msa.analysis.retry.queue";
    public static final String ANALYSIS_DLQ = "msa.analysis.dlq";
    public static final String ANALYSIS_DLX = "msa.analysis.dlx";
    public static final String ANALYSIS_ROUTING_KEY = "msa.analysis.dispatch";
    public static final String ANALYSIS_RETRY_ROUTING_KEY = "msa.analysis.retry";
    public static final String ANALYSIS_DLQ_ROUTING_KEY = "msa.analysis.dead";

    @Bean
    public DirectExchange analysisExchange() {
        return new DirectExchange(ANALYSIS_EXCHANGE, true, false);
    }

    @Bean
    public DirectExchange analysisDeadLetterExchange() {
        return new DirectExchange(ANALYSIS_DLX, true, false);
    }

    @Bean
    public Queue analysisQueue() {
        return QueueBuilder.durable(ANALYSIS_QUEUE)
                .withArgument("x-dead-letter-exchange", ANALYSIS_DLX)
                .withArgument("x-dead-letter-routing-key", ANALYSIS_DLQ_ROUTING_KEY)
                .build();
    }

    @Bean
    public Queue analysisRetryQueue(MsaProperties properties) {
        return QueueBuilder.durable(ANALYSIS_RETRY_QUEUE)
                .withArgument("x-message-ttl", properties.getRetryDelayMs())
                .withArgument("x-dead-letter-exchange", ANALYSIS_EXCHANGE)
                .withArgument("x-dead-letter-routing-key", ANALYSIS_ROUTING_KEY)
                .build();
    }

    @Bean
    public Queue analysisDeadLetterQueue() {
        return QueueBuilder.durable(ANALYSIS_DLQ).build();
    }

    @Bean
    public Binding analysisBinding(Queue analysisQueue, DirectExchange analysisExchange) {
        return BindingBuilder.bind(analysisQueue).to(analysisExchange).with(ANALYSIS_ROUTING_KEY);
    }

    @Bean
    public Binding analysisRetryBinding(Queue analysisRetryQueue, DirectExchange analysisExchange) {
        return BindingBuilder.bind(analysisRetryQueue).to(analysisExchange).with(ANALYSIS_RETRY_ROUTING_KEY);
    }

    @Bean
    public Binding analysisDeadLetterBinding(Queue analysisDeadLetterQueue, DirectExchange analysisDeadLetterExchange) {
        return BindingBuilder.bind(analysisDeadLetterQueue).to(analysisDeadLetterExchange).with(ANALYSIS_DLQ_ROUTING_KEY);
    }

    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory, MessageConverter jsonMessageConverter) {
        RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        rabbitTemplate.setMessageConverter(jsonMessageConverter);
        return rabbitTemplate;
    }
}
