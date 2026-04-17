package com.yupi.usercenter.service.impl;

import com.sun.net.httpserver.HttpServer;
import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

class HttpMsaClientTest {

    private HttpServer server;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void submitAnalysisPostsJsonToMsaAnalyze() throws Exception {
        final String[] body = new String[1];
        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/analyze", exchange -> {
            body[0] = readBody(exchange.getRequestBody());
            byte[] response = "{\"taskId\":\"task-1\",\"status\":\"PENDING\"}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
            exchange.sendResponseHeaders(202, response.length);
            try (OutputStream output = exchange.getResponseBody()) {
                output.write(response);
            }
        });
        server.start();

        MsaProperties properties = new MsaProperties();
        properties.setBaseUrl("http://127.0.0.1:" + server.getAddress().getPort());
        HttpMsaClient client = new HttpMsaClient(properties);

        Map<String, Object> payload = new HashMap<>();
        payload.put("text", "happy");
        payload.put("language", "en");
        AnalysisTaskResponse result = client.submitAnalysis(payload);

        Assertions.assertEquals("task-1", result.getTaskId());
        Assertions.assertTrue(body[0].contains("\"text\":\"happy\""));
        Assertions.assertTrue(body[0].contains("\"language\":\"en\""));
    }

    private static String readBody(InputStream input) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int read;
        while ((read = input.read(buffer)) != -1) {
            output.write(buffer, 0, read);
        }
        return new String(output.toByteArray(), StandardCharsets.UTF_8);
    }
}
