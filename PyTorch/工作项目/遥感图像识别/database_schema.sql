-- 用户表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    organization VARCHAR(100) NOT NULL,
    department VARCHAR(100),
    license_number VARCHAR(50) NOT NULL,
    research_field VARCHAR(50),
    platforms JSON,
    usage_purpose TEXT,
    last_login DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    name VARCHAR(100),
    phone VARCHAR(20),
    bio TEXT
);

-- 图像表 (同时存储图像和视频)
CREATE TABLE images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    storage_filename VARCHAR(255) NOT NULL,
    upload_path VARCHAR(255) NOT NULL,
    file_size INT,
    width INT,
    height INT,
    file_type VARCHAR(10) DEFAULT 'image', -- 文件类型: image, video
    duration FLOAT,                         -- 视频时长(秒)
    frame_count INT,                        -- 视频帧数
    metadata JSON,
    location_lat DECIMAL(10, 8),
    location_lng DECIMAL(11, 8),
    created_at DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 模型表
CREATE TABLE models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    path VARCHAR(255) NOT NULL,
    example_image VARCHAR(255),
    result_example_image VARCHAR(255),
    creator_id INT,
    is_shared BOOLEAN DEFAULT FALSE,
    is_system BOOLEAN DEFAULT FALSE,
    accuracy DECIMAL(5, 2),
    usage_count INT DEFAULT 0,
    supports_video BOOLEAN DEFAULT TRUE,   -- 模型是否支持视频处理
    created_at DATETIME,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (creator_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 收藏模型表
CREATE TABLE favorite_models (
    user_id INT NOT NULL,
    model_id INT NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (user_id, model_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

-- 处理历史表
CREATE TABLE processing_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    image_id INT,
    model_id INT NOT NULL,
    result_path VARCHAR(255),
    accuracy DECIMAL(5,2),
    processing_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    object_count INT DEFAULT 0,
    categories TEXT,
    file_type ENUM('image', 'video', 'realtime') DEFAULT 'image',
    status ENUM('pending', 'processing', 'completed', 'error') DEFAULT 'completed',
    progress INT DEFAULT 100,
    data_size BIGINT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE SET NULL,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

-- 通知表
CREATE TABLE notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    type VARCHAR(50) NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 权限申请表
CREATE TABLE permission_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    request_type VARCHAR(50) NOT NULL,
    request_reason TEXT NOT NULL,
    additional_info JSON,
    status VARCHAR(20) DEFAULT 'pending',
    created_at DATETIME NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 处理作业表 (用于长时间运行的视频处理作业)
CREATE TABLE processing_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    history_id INT NOT NULL,              -- 关联的处理历史ID
    status VARCHAR(20) NOT NULL,          -- 作业状态: queued, processing, completed, failed
    progress INT DEFAULT 0,               -- 处理进度(0-100)
    message TEXT,                         -- 状态信息或错误消息
    start_time DATETIME,                  -- 开始处理时间
    end_time DATETIME,                    -- 结束处理时间
    created_at DATETIME NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (history_id) REFERENCES processing_history(id) ON DELETE CASCADE
);