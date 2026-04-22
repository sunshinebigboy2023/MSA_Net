create database if not exists yupi;

use yupi;

create table if not exists user
(
    id           bigint auto_increment comment 'id' primary key,
    username     varchar(256)                       null comment 'username',
    userAccount  varchar(256)                       null comment 'account',
    avatarUrl    varchar(1024)                      null comment 'avatar url',
    gender       tinyint                            null comment 'gender',
    userPassword varchar(512)                       not null comment 'password hash',
    phone        varchar(128)                       null comment 'phone',
    email        varchar(512)                       null comment 'email',
    userStatus   int      default 0                 not null comment '0 normal',
    createTime   datetime default CURRENT_TIMESTAMP null comment 'created time',
    updateTime   datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP comment 'updated time',
    isDelete     tinyint  default 0                 not null comment 'logical delete flag',
    userRole     int      default 0                 not null comment '0 user, 1 admin',
    planetCode   varchar(512)                       null comment 'registration code'
)
    comment 'user';

create table if not exists analysis_task
(
    id               bigint auto_increment comment 'id' primary key,
    taskId           varchar(64)                        not null comment 'external task id',
    userId           bigint                             not null comment 'submitter user id',
    status           varchar(32)                        not null comment 'QUEUED/RUNNING/SUCCESS/FAILED',
    payload          longtext                           not null comment 'analysis request payload json',
    result           longtext                           null comment 'analysis result json',
    error            varchar(1024)                      null comment 'failure reason',
    retryCount       int      default 0                 not null comment 'retry count',
    processingTimeMs bigint                             null comment 'worker processing time',
    createTime       datetime default CURRENT_TIMESTAMP null comment 'created time',
    updateTime       datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP comment 'updated time',
    isDelete         tinyint  default 0                 not null comment 'logical delete flag',
    unique key uk_taskId (taskId),
    key idx_user_status (userId, status),
    key idx_status_updateTime (status, updateTime)
)
    comment 'multimodal analysis task';
