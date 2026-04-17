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
