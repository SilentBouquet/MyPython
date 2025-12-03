-- 遥感图像识别系统数据库架构

-- 用户表：存储系统用户的基本信息和权限
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 用户唯一标识符
    username VARCHAR(50) NOT NULL UNIQUE, -- 用户名（唯一）
    email VARCHAR(100) NOT NULL UNIQUE, -- 用户邮箱（唯一）
    password VARCHAR(255) NOT NULL, -- 密码（加密存储）
    organization VARCHAR(100) NOT NULL, -- 所属机构/组织
    department VARCHAR(100), -- 所属部门
    license_number VARCHAR(50) NOT NULL, -- 许可证号码或用户编号
    research_field VARCHAR(50), -- 研究领域
    platforms JSON, -- 用户使用的平台信息（JSON格式）
    usage_purpose TEXT, -- 使用系统的目的
    last_login DATETIME, -- 最后登录时间
    created_at DATETIME NOT NULL, -- 账号创建时间
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- 信息更新时间
    name VARCHAR(100), -- 用户真实姓名
    phone VARCHAR(20), -- 联系电话
    bio TEXT, -- 个人简介
    is_admin BOOLEAN DEFAULT FALSE -- 是否为管理员（1为是，0为否）
);

-- 图像和视频表：存储用户上传的图像和视频文件元数据
CREATE TABLE images (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 文件唯一标识符
    user_id INT NOT NULL, -- 上传用户的ID（外键）
    original_filename VARCHAR(255) NOT NULL, -- 原始文件名
    storage_filename VARCHAR(255) NOT NULL, -- 存储在服务器上的文件名
    upload_path VARCHAR(255) NOT NULL, -- 文件存储路径
    file_size INT, -- 文件大小（字节）
    width INT, -- 图像宽度（像素）
    height INT, -- 图像高度（像素）
    file_type VARCHAR(10) DEFAULT 'image', -- 文件类型（image或video）
    duration FLOAT, -- 视频时长（秒）
    frame_count INT, -- 视频帧数
    metadata JSON, -- 其他元数据（JSON格式）
    location_lat DECIMAL(10, 8), -- 拍摄位置纬度
    location_lng DECIMAL(11, 8), -- 拍摄位置经度
    created_at DATETIME NOT NULL, -- 上传时间
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- 外键约束：删除用户时级联删除其上传的文件
);

-- 模型表：存储系统中可用的分析模型
CREATE TABLE models (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 模型唯一标识符
    name VARCHAR(100) NOT NULL, -- 模型名称
    type VARCHAR(50) NOT NULL, -- 模型类型（如landcover、building等）
    description TEXT, -- 模型描述
    path VARCHAR(255) NOT NULL, -- 模型文件存储路径
    example_image VARCHAR(255), -- 示例输入图像路径
    result_example_image VARCHAR(255), -- 示例输出结果图像路径
    creator_id INT, -- 模型创建者的用户ID（外键）
    is_shared BOOLEAN DEFAULT FALSE, -- 是否共享给其他用户（1为是，0为否）
    is_system BOOLEAN DEFAULT FALSE, -- 是否为系统默认模型（1为是，0为否）
    accuracy DECIMAL(5, 2), -- 模型精度
    usage_count INT DEFAULT 0, -- 使用次数计数
    supports_video BOOLEAN DEFAULT TRUE, -- 是否支持视频处理（1为是，0为否）
    created_at DATETIME, -- 模型创建时间
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- 模型更新时间
    FOREIGN KEY (creator_id) REFERENCES users(id) ON DELETE SET NULL -- 外键约束：删除创建者时将creator_id设为NULL
);

-- 收藏模型表：记录用户收藏的模型
CREATE TABLE favorite_models (
    user_id INT NOT NULL, -- 用户ID（外键）
    model_id INT NOT NULL, -- 模型ID（外键）
    created_at DATETIME NOT NULL, -- 收藏时间
    PRIMARY KEY (user_id, model_id), -- 联合主键：同一用户不能重复收藏同一模型
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, -- 外键约束：删除用户时级联删除其收藏记录
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE -- 外键约束：删除模型时级联删除收藏记录
);

-- 处理历史表：记录用户使用模型处理文件的历史记录
CREATE TABLE processing_history (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 处理记录唯一标识符
    user_id INT NOT NULL, -- 处理发起者的用户ID（外键）
    image_id INT, -- 被处理的图像/视频ID（外键，可为空）
    model_id INT NOT NULL, -- 使用的模型ID（外键）
    result_path VARCHAR(255), -- 处理结果存储路径
    accuracy DECIMAL(5,2), -- 处理结果精度
    processing_time FLOAT, -- 处理耗时（秒）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 处理开始时间
    object_count INT DEFAULT 0, -- 检测到的目标数量
    categories TEXT, -- 检测到的类别列表（逗号分隔）
    file_type ENUM('image', 'video', 'realtime') DEFAULT 'image', -- 处理的文件类型
    status ENUM('pending', 'processing', 'completed', 'error') DEFAULT 'completed', -- 处理状态
    progress INT DEFAULT 100, -- 处理进度（百分比）
    data_size BIGINT, -- 处理的数据量（字节）
    FOREIGN KEY (user_id) REFERENCES users(id), -- 外键约束：删除用户时保留处理记录（user_id保留但关联断开）
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE SET NULL, -- 外键约束：删除图像时将image_id设为NULL
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE -- 外键约束：删除模型时级联删除相关处理记录
);

-- 通知表：存储系统向用户发送的通知消息
CREATE TABLE notifications (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 通知唯一标识符
    user_id INT NOT NULL, -- 接收通知的用户ID（外键）
    title VARCHAR(100) NOT NULL, -- 通知标题
    content TEXT NOT NULL, -- 通知内容
    type VARCHAR(50) NOT NULL, -- 通知类型（如info、warning、error等）
    is_read BOOLEAN DEFAULT FALSE, -- 是否已读（1为已读，0为未读）
    created_at DATETIME NOT NULL, -- 通知创建时间
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- 外键约束：删除用户时级联删除其通知
);

-- 权限申请表：记录用户提交的权限申请
CREATE TABLE permission_requests (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 申请唯一标识符
    user_id INT NOT NULL, -- 申请用户的ID（外键）
    request_type VARCHAR(50) NOT NULL, -- 申请类型（如model_publish、data_export等）
    request_reason TEXT NOT NULL, -- 申请原因说明
    additional_info JSON, -- 额外信息（JSON格式）
    status VARCHAR(20) DEFAULT 'pending', -- 申请状态（pending、approved、rejected）
    created_at DATETIME NOT NULL, -- 申请提交时间
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- 申请更新时间
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- 外键约束：删除用户时级联删除其申请记录
);

-- 处理作业表：记录长时间运行的处理任务（如视频处理）
CREATE TABLE processing_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 作业唯一标识符
    user_id INT NOT NULL, -- 作业发起者的用户ID（外键）
    history_id INT NOT NULL, -- 关联的处理历史ID（外键）
    status VARCHAR(20) NOT NULL, -- 作业状态（queued、processing、completed、failed）
    progress INT DEFAULT 0, -- 作业进度（百分比）
    message TEXT, -- 状态信息或错误消息
    start_time DATETIME, -- 作业开始时间
    end_time DATETIME, -- 作业结束时间
    created_at DATETIME NOT NULL, -- 作业创建时间
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- 作业更新时间
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, -- 外键约束：删除用户时级联删除其作业记录
    FOREIGN KEY (history_id) REFERENCES processing_history(id) ON DELETE CASCADE -- 外键约束：删除处理历史时级联删除相关作业
);

-- 模型公开申请表：记录用户申请将模型公开的请求
CREATE TABLE model_publish_requests (
    id INT AUTO_INCREMENT PRIMARY KEY, -- 申请唯一标识符
    model_id INT NOT NULL, -- 申请公开的模型ID（外键）
    user_id INT NOT NULL, -- 申请用户的ID（外键）
    reason TEXT, -- 申请公开的原因说明
    status ENUM('pending', 'approved', 'rejected') NOT NULL DEFAULT 'pending', -- 申请状态
    admin_comment TEXT, -- 管理员审核意见
    created_at DATETIME NOT NULL, -- 申请提交时间
    updated_at DATETIME, -- 申请更新时间
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE, -- 外键约束：删除模型时级联删除相关申请
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE -- 外键约束：删除用户时级联删除其申请记录
);