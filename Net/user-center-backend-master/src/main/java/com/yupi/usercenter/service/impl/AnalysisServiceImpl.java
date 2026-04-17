package com.yupi.usercenter.service.impl;

import com.yupi.usercenter.common.ErrorCode;
import com.yupi.usercenter.config.MsaProperties;
import com.yupi.usercenter.exception.BusinessException;
import com.yupi.usercenter.model.domain.User;
import com.yupi.usercenter.model.domain.analysis.AnalysisResultResponse;
import com.yupi.usercenter.model.domain.analysis.AnalysisTaskResponse;
import com.yupi.usercenter.service.AnalysisService;
import com.yupi.usercenter.service.MsaClient;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

@Service
public class AnalysisServiceImpl implements AnalysisService {

    private final MsaClient msaClient;

    private final MsaProperties properties;

    public AnalysisServiceImpl(MsaClient msaClient, MsaProperties properties) {
        this.msaClient = msaClient;
        this.properties = properties;
    }

    @Override
    public AnalysisTaskResponse submit(String text, String language, MultipartFile video, User currentUser) {
        boolean hasText = StringUtils.isNotBlank(text);
        boolean hasVideo = video != null && !video.isEmpty();
        if (!hasText && !hasVideo) {
            throw new BusinessException(ErrorCode.PARAMS_ERROR, "Please enter text or upload a video");
        }

        Map<String, Object> payload = new HashMap<>();
        if (hasText) {
            payload.put("text", text.trim());
        }

        String normalizedLanguage = normalizeLanguage(language);
        if (StringUtils.isNotBlank(normalizedLanguage)) {
            payload.put("language", normalizedLanguage);
        }

        if (hasVideo) {
            payload.put("videoFile", saveVideo(video, currentUser).toAbsolutePath().toString());
        }

        return msaClient.submitAnalysis(payload);
    }

    @Override
    public AnalysisTaskResponse getTask(String taskId) {
        return msaClient.getTask(taskId);
    }

    @Override
    public AnalysisResultResponse getResult(String taskId) {
        return msaClient.getResult(taskId);
    }

    private String normalizeLanguage(String language) {
        if (StringUtils.isBlank(language)) {
            return null;
        }
        String value = language.trim().toLowerCase(Locale.ROOT);
        if ("zh".equals(value) || "en".equals(value)) {
            return value;
        }
        return null;
    }

    private Path saveVideo(MultipartFile video, User currentUser) {
        long userId = currentUser == null || currentUser.getId() == null ? 0L : currentUser.getId();
        Path uploadDir = Paths.get(properties.getUploadDir(), String.valueOf(userId)).toAbsolutePath().normalize();
        try {
            Files.createDirectories(uploadDir);
            Path target = uploadDir.resolve(UUID.randomUUID() + extensionOf(video.getOriginalFilename()));
            try (InputStream inputStream = video.getInputStream()) {
                Files.copy(inputStream, target, StandardCopyOption.REPLACE_EXISTING);
            }
            return target;
        } catch (IOException e) {
            throw new BusinessException(ErrorCode.SYSTEM_ERROR, "保存上传视频失败，请检查上传目录权限和磁盘空间");
        }
    }

    private String extensionOf(String filename) {
        if (StringUtils.isBlank(filename)) {
            return ".mp4";
        }
        String cleanName = Paths.get(filename).getFileName().toString();
        int dotIndex = cleanName.lastIndexOf('.');
        if (dotIndex < 0 || dotIndex == cleanName.length() - 1) {
            return ".mp4";
        }
        String extension = cleanName.substring(dotIndex).toLowerCase(Locale.ROOT);
        if (extension.length() > 10) {
            return ".mp4";
        }
        return extension;
    }
}
