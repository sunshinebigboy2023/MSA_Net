package com.yupi.usercenter.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "msa")
public class MsaProperties {

    private String baseUrl = "http://127.0.0.1:8000";

    private String uploadDir = "runtime/uploads";
}
