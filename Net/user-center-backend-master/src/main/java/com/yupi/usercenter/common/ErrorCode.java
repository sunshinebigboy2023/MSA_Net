package com.yupi.usercenter.common;

public enum ErrorCode {

    SUCCESS(0, "ok", ""),
    PARAMS_ERROR(40000, "\u8bf7\u6c42\u53c2\u6570\u9519\u8bef", ""),
    NULL_ERROR(40001, "\u8bf7\u6c42\u6570\u636e\u4e3a\u7a7a", ""),
    NOT_LOGIN(40100, "\u672a\u767b\u5f55", ""),
    NO_AUTH(40101, "\u65e0\u6743\u9650", ""),
    SYSTEM_ERROR(50000, "\u7cfb\u7edf\u5185\u90e8\u5f02\u5e38", "");

    private final int code;

    private final String message;

    private final String description;

    ErrorCode(int code, String message, String description) {
        this.code = code;
        this.message = message;
        this.description = description;
    }

    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }

    public String getDescription() {
        return description;
    }
}
