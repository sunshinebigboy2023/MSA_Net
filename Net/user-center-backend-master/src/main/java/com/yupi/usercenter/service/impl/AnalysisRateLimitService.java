package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.config.MsaProperties;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.data.redis.core.script.RedisScript;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
public class AnalysisRateLimitService {

    private static final RedisScript<Long> RATE_LIMIT_SCRIPT = new DefaultRedisScript<>(
            "local current = redis.call('INCR', KEYS[1]) " +
                    "if current == 1 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end " +
                    "if current > tonumber(ARGV[1]) then return 0 end " +
                    "return 1",
            Long.class);

    private final StringRedisTemplate redisTemplate;

    private final MsaProperties properties;

    public AnalysisRateLimitService(StringRedisTemplate redisTemplate, MsaProperties properties) {
        this.redisTemplate = redisTemplate;
        this.properties = properties;
    }

    public boolean allowSubmit(Long userId) {
        Long safeUserId = userId == null ? 0L : userId;
        Long result = redisTemplate.execute(
                RATE_LIMIT_SCRIPT,
                Collections.singletonList("msa:rate:submit:" + safeUserId),
                String.valueOf(properties.getSubmitLimit()),
                String.valueOf(properties.getSubmitWindowSeconds()));
        return Long.valueOf(1L).equals(result);
    }
}
